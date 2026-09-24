"""Checkpoint evaluator with independent, LoRA-capable TP1 servers."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import threading
import time
import json

from dashboard import evaluate_checkpoints as evaluator

SingleServer = evaluator.VLLMServerManager
SingleEngine = evaluator.LoRAOpenAIEngine
REPLICA_COUNT = 4


def record_progress(event):
    path = os.environ.get("UNITYMAS_EVAL_PROGRESS_PATH")
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}")
        temporary.write_text(json.dumps({"event": event, "time": time.time()}))
        temporary.replace(target)


class ReplicaServers:
    def __init__(self, **kwargs):
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if (not 1 <= REPLICA_COUNT <= 8 or len(devices) != REPLICA_COUNT
                or len(set(devices)) != REPLICA_COUNT or any(not d.strip() for d in devices)
                or kwargs["tensor_parallel_size"] != 1 or kwargs["data_parallel_size"] != REPLICA_COUNT):
            raise ValueError("Replica evaluation requires distinct GPUs, TP1, and matching replica count")
        self.servers = []
        for index, device in enumerate(devices):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": device,
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            self.servers.append(SingleServer(**{**kwargs, "data_parallel_size": 1,
                "port": kwargs["port"] + index, "process_env": env}))

    def start(self):
        try:
            with ThreadPoolExecutor(max_workers=REPLICA_COUNT) as pool:
                futures = [pool.submit(server.start) for server in self.servers]
                for future in futures:
                    future.result()
        except BaseException:
            self.stop()
            raise
        print(f"[TP1ReplicaEval] {REPLICA_COUNT} independent LoRA servers ready", flush=True)
        record_progress("servers_ready")

    def load_lora(self, name, path):
        for index, server in enumerate(self.servers):
            server.load_lora(name, path)
            print(f"[TP1ReplicaEval] adapter_loaded replica={index} name={name} path={path}", flush=True)

    def unload_all_loras(self):
        errors = []
        for server in self.servers:
            try:
                server.unload_all_loras()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Replica adapter cleanup failed: {errors}")

    def stop(self):
        errors = []
        for server in self.servers:
            try:
                server.stop()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"Replica process cleanup failed: {errors}")


class ReplicaEngine(SingleEngine):
    def __init__(self, *, base_url, **kwargs):
        from urllib.parse import urlsplit, urlunsplit
        super().__init__(base_url=base_url, **kwargs)
        url = urlsplit(base_url)
        self._urls = [urlunsplit((url.scheme, f"{url.hostname}:{url.port + i}", url.path, "", "")) for i in range(REPLICA_COUNT)]
        self._client_type = type(self.client)
        self._clients = [self.client] + [self._client_type(base_url=u, api_key="EMPTY") for u in self._urls[1:]]
        self._inflight = [0] * REPLICA_COUNT
        self._request_counts = [0] * REPLICA_COUNT

    async def get_model_response(self, messages, **kwargs):
        if not self._clients:
            self._clients = [self._client_type(base_url=u, api_key="EMPTY") for u in self._urls]
        index = min(range(len(self._inflight)), key=lambda i: self._inflight[i])
        self._inflight[index] += 1
        self._request_counts[index] += 1
        # Route selection mutates SingleEngine.model. Keep model/client local
        # to this request while preserving its parser, limits and sampling.
        request_engine = copy.copy(self)
        request_engine.client = self._clients[index]
        try:
            result = await SingleEngine.get_model_response(request_engine, messages, **kwargs)
            record_progress("request_complete")
            return result
        finally:
            self._inflight[index] -= 1

    async def close_clients(self):
        await asyncio.gather(*(client.close() for client in self._clients))
        self._clients = []
        print(f"[TP1ReplicaEval] completed_requests={self._request_counts}", flush=True)


def main():
    global REPLICA_COUNT
    args = evaluator.parse_args()
    REPLICA_COUNT = args.data_parallel
    original_evaluate = evaluator.evaluate_checkpoint

    async def evaluate_and_close(**kwargs):
        engine = kwargs["engine"]
        try:
            result = await original_evaluate(**kwargs)
            if args.max_samples is not None:
                destination = Path(args.trajectory_output_dir) / "NONCANONICAL_RESULTS.jsonl"
                evaluator.save_results_to_json([result], str(destination), include_problem_results=True)
            return result
        finally:
            await engine.close_clients()

    evaluator.VLLMServerManager = ReplicaServers
    evaluator.LoRAOpenAIEngine = ReplicaEngine
    evaluator.evaluate_checkpoint = evaluate_and_close
    evaluator.main(args)


if __name__ == "__main__":
    main()
