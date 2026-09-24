"""CPU-only replay: real vLLM output processing, no model or training results."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace


def toy_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "alpha": 1, "beta": 2, "STOP": 3, "[EOS]": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]")


def replay_output(tokenizer, token_ids, *, final_only, detokenize, stop=None, finish="length", logprobs=True):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.engine.output_processor import OutputProcessor

    params = SamplingParams(
        max_tokens=len(token_ids), temperature=0.7, seed=42,
        logprobs=0 if logprobs else None, detokenize=detokenize, stop=stop,
        output_kind=RequestOutputKind.FINAL_ONLY if final_only else RequestOutputKind.CUMULATIVE)
    request = EngineCoreRequest(
        request_id="CPU_REPLAY_NOT_CANONICAL", prompt_token_ids=tokenizer.encode("alpha", add_special_tokens=False),
        mm_inputs=None, mm_hashes=None, mm_placeholders=None, sampling_params=params,
        pooling_params=None, eos_token_id=tokenizer.eos_token_id, arrival_time=time.time(),
        lora_request=None, cache_salt=None, data_parallel_rank=None)
    processor = OutputProcessor(SimpleNamespace(get_lora_tokenizer=lambda _: tokenizer), log_stats=False)
    processor.add_request(request, prompt=None)
    final = None
    deliveries = 0
    started = time.perf_counter()
    for i, token_id in enumerate(token_ids):
        terminal = {"length": FinishReason.LENGTH, "stop": FinishReason.STOP}[finish] if i == len(token_ids) - 1 else None
        core = EngineCoreOutput(
            request_id=request.request_id, new_token_ids=[token_id],
            new_logprobs=([[token_id]], [[-0.5]], [1]) if logprobs else None,
            finish_reason=terminal)
        outputs = processor.process_outputs([core]).request_outputs
        deliveries += len(outputs)
        if outputs:
            final = outputs[-1]
            if final.finished:
                break
    elapsed = time.perf_counter() - started
    assert final is not None and final.finished
    output = final.outputs[0]
    numeric = None if output.logprobs is None else [
        row[token].logprob for token, row in zip(output.token_ids, output.logprobs, strict=True)]
    contract = {"token_ids": list(output.token_ids), "logprobs": numeric,
                "finish_reason": output.finish_reason, "stop_reason": output.stop_reason}
    return {"elapsed_s": elapsed, "deliveries": deliveries, "contract": contract,
            "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()}


async def replay_admission(batcher_class, *, slow_seconds=0.2):
    batcher = batcher_class(max_batch_size=2, max_requests_per_turn=4, coalesce_ms=0)
    started = asyncio.Event()
    dispatch_times = {}

    async def slow(route):
        started.set()
        await asyncio.sleep(slow_seconds)
        return (route, "slow")

    async def late(route):
        dispatch_times["late"] = time.perf_counter()
        return (route, "late")

    first = asyncio.create_task(batcher.submit("orchestrator", slow))
    await started.wait()
    await asyncio.sleep(0.01)
    submitted = time.perf_counter()
    second = asyncio.create_task(batcher.submit("worker0", late))
    results = await asyncio.gather(first, second)
    return {"late_queue_wait_s": dispatch_times["late"] - submitted, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-batcher", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    from verl.experimental.agent_loop.route_homogeneous_batcher import RouteHomogeneousBatcher
    import sys
    import vllm

    spec = importlib.util.spec_from_file_location("baseline_route_batcher", args.baseline_batcher)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    else:
        tokenizer = toy_tokenizer()
    seed_tokens = tokenizer.encode("alpha beta ", add_special_tokens=False)
    report = {"scope": "CPU_SYNTHETIC_REPLAY_NOT_MODEL_OR_TRAINING_BENCHMARK",
              "vllm": vllm.__version__, "tokenizer": str(args.tokenizer or "toy_wordlevel"),
              "baseline_source_sha256": hashlib.sha256(args.baseline_batcher.read_bytes()).hexdigest(),
              "admission": {}, "output_processing": []}
    for label, cls in (("before", baseline.RouteHomogeneousBatcher), ("after", RouteHomogeneousBatcher)):
        samples = [asyncio.run(replay_admission(cls)) for _ in range(args.repeats)]
        assert all(sample["results"] == [("orchestrator", "slow"), ("worker0", "late")] for sample in samples)
        report["admission"][label] = {"samples": samples,
            "median_late_queue_wait_s": statistics.median(s["late_queue_wait_s"] for s in samples)}
    for length in (512, 5120):
        tokens = (seed_tokens * (length // len(seed_tokens) + 1))[:length]
        modes = (("before", False, True), ("final_only", True, True), ("final_token_only", True, False))
        samples = {name: [] for name, _, _ in modes}
        reference = None
        # Warm both paths, then interleave samples to reduce ordering bias.
        for iteration in range(args.repeats + 1):
            for name, final_only, detokenize in modes:
                result = replay_output(tokenizer, tokens, final_only=final_only, detokenize=detokenize)
                reference = reference or result["contract"]
                assert result["contract"] == reference
                if iteration:
                    samples[name].append({key: value for key, value in result.items() if key != "contract"})
        report["output_processing"].append({"tokens": length, "modes": {
            name: {"median_cpu_s": statistics.median(s["elapsed_s"] for s in rows), "samples": rows}
            for name, rows in samples.items()}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
