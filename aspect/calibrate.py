#!/usr/bin/env python3
"""Build a LoRA-SB update-aligned basis from a pinned training set.

The implementation follows CERT-Lab/lora-sb: estimate the first AdamW update
with ``-effective_lr * sign(sum_gradients)``, compute a rank-r low-rank SVD,
freeze the orthonormal A/B factors, and initialize R from the singular values.
Only the compact A/B/R artifact and a reproducibility manifest are persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from verl.workers.actor.agent_lorasb import (
    OFFICIAL_LORASB_COMMIT,
    OFFICIAL_LORASB_REPOSITORY,
    POLICY_MODE,
    SVD_BACKEND,
    load_basis_artifact,
    lorasb_core_tensor_key,
    truncated_svd_lorasb_factors,
)
from verl.workers.actor.agent_loraxs import basis_tensor_keys, sha256_file

from aspect.data.contracts import OFFICIAL_DATASET_CONTRACT
from aspect.data.sources import _load_mount, _sha256_file, _text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--task", choices=("math", "code"), default="math")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--allow-nonunit-adapter-scale", action="store_true")
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=15)
    parser.add_argument("--calibration-samples", type=int, default=50)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--model-revision")
    return parser.parse_args()


def first_adamw_update_approximation(
    gradient_sum: torch.Tensor,
    *,
    effective_lr: float,
) -> torch.Tensor:
    """Approximate AdamW's first update as used by the official LoRA-SB code."""

    if effective_lr <= 0:
        raise ValueError(f"effective_lr must be positive, got {effective_lr}")
    return -float(effective_lr) * torch.sign(gradient_sum)


def summarize_calibration_statistics(
    *,
    example_loss_sum: float,
    token_nll_sum: float,
    example_count: int,
    target_token_count: int,
) -> dict[str, int | float]:
    """Return auditable supervised-calibration loss summaries."""

    if example_count <= 0 or target_token_count <= 0:
        raise ValueError(
            "LoRA-SB calibration statistics require positive example and target-token counts"
        )
    values = (float(example_loss_sum), float(token_nll_sum))
    if any(not torch.isfinite(torch.tensor(value)).item() for value in values):
        raise ValueError("LoRA-SB calibration losses must be finite")
    return {
        "example_count": int(example_count),
        "target_token_count": int(target_token_count),
        "example_loss_mean": float(example_loss_sum) / int(example_count),
        "token_nll_mean": float(token_nll_sum) / int(target_token_count),
    }


def summarize_singular_spectrum(
    module_stats: dict[str, dict[str, Any]],
    *,
    rank: int,
) -> list[dict[str, int | float]]:
    """Aggregate the per-module top-r spectrum without storing model tensors."""

    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if not module_stats:
        return []
    values_by_rank: list[list[float]] = [[] for _ in range(rank)]
    for module_name, stats in module_stats.items():
        values = stats.get("top_r_singular_values")
        if not isinstance(values, list) or len(values) != rank:
            raise ValueError(
                f"Module {module_name!r} does not contain exactly {rank} singular values"
            )
        for rank_index, value in enumerate(values):
            scalar = float(value)
            if not torch.isfinite(torch.tensor(scalar)).item() or scalar < 0.0:
                raise ValueError(
                    f"Module {module_name!r} has an invalid singular value {value!r}"
                )
            values_by_rank[rank_index].append(scalar)

    total_energy = sum(
        value * value for rank_values in values_by_rank for value in rank_values
    )
    cumulative_energy = 0.0
    spectrum: list[dict[str, int | float]] = []
    for rank_index, rank_values in enumerate(values_by_rank, start=1):
        cumulative_energy += sum(value * value for value in rank_values)
        spectrum.append(
            {
                "rank": rank_index,
                "singular_value_mean": sum(rank_values) / len(rank_values),
                "cumulative_energy_ratio": (
                    cumulative_energy / total_energy if total_energy > 0.0 else 0.0
                ),
            }
        )
    return spectrum


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA-SB basis calibration requires CUDA")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _validate_contract(args: argparse.Namespace) -> None:
    if args.rank <= 0:
        raise ValueError(f"rank must be positive, got {args.rank}")
    if args.lora_alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {args.lora_alpha}")
    if args.lora_alpha != args.rank and not getattr(
        args, "allow_nonunit_adapter_scale", False
    ):
        raise ValueError(
            "Non-unit LoRA-SB adapter scale requires the explicit "
            "--allow-nonunit-adapter-scale diagnostic gate; "
            f"got alpha={args.lora_alpha}, rank={args.rank}"
        )
    if args.actor_lr <= 0 or args.warmup_steps <= 0:
        raise ValueError("actor_lr and warmup_steps must be positive")
    if args.calibration_samples <= 0 or args.max_seq_length <= 0:
        raise ValueError("calibration_samples and max_seq_length must be positive")


def _validate_existing(root: Path, args: argparse.Namespace) -> bool:
    if not root.exists():
        return False
    _, manifest = load_basis_artifact(root)
    expected = {
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "actor_learning_rate": args.actor_lr,
        "warmup_steps": args.warmup_steps,
        "calibration_samples": args.calibration_samples,
        "max_seq_length": args.max_seq_length,
        "n_iter": args.n_iter,
        "random_state": args.random_state,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if args.expected_config_sha256 and (
        manifest.get("model_config_sha256") != args.expected_config_sha256
    ):
        mismatches["model_config_sha256"] = (
            manifest.get("model_config_sha256"),
            args.expected_config_sha256,
        )
    if args.task == "code" and manifest.get("calibration_task") != "code":
        mismatches["calibration_task"] = (
            manifest.get("calibration_task"),
            "code",
        )
    if mismatches:
        raise ValueError(f"Existing AW-LoRA-SB basis has a different contract: {mismatches}")
    return True


def _load_math_calibration_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    raw, source = _load_mount(
        Path(args.train_root),
        "train",
        config_hint=OFFICIAL_DATASET_CONTRACT["dapo"]["config"],
    )
    if source is None:
        raise ValueError("LoRA-SB calibration requires the exact mounted DAPO source file")
    source_sha = _sha256_file(source)
    expected_sha = OFFICIAL_DATASET_CONTRACT["dapo"]["source_sha256"]
    if source_sha != expected_sha:
        raise ValueError(
            f"DAPO source SHA256 mismatch: expected {expected_sha}, got {source_sha}"
        )
    expected_rows = OFFICIAL_DATASET_CONTRACT["dapo"]["raw_rows"]
    if len(raw) != expected_rows:
        raise ValueError(f"DAPO row-count mismatch: expected {expected_rows}, got {len(raw)}")
    required = {"prompt", "solution"}
    missing = required.difference(raw.column_names)
    if missing:
        raise ValueError(f"DAPO calibration data is missing columns {sorted(missing)}")

    train = raw.train_test_split(test_size=0.1, seed=42)["train"]
    if args.calibration_samples > len(train):
        raise ValueError(
            f"Requested {args.calibration_samples} calibration samples from {len(train)} rows"
        )
    selected = train.select(range(args.calibration_samples))
    rows = [
        {
            "question": _text(example["prompt"]),
            "solution": str(example["solution"]),
        }
        for example in selected
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        )
    return rows, {
        "task": "math",
        "source_file": str(source.resolve()),
        "source_sha256": source_sha,
        "raw_rows": len(raw),
        "train_rows": len(train),
        "selected_rows_sha256": digest.hexdigest(),
    }


def _load_code_calibration_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    from datasets import load_dataset

    from aspect.data.code import (
        CALIBRATION_ROWS,
        PACKAGE_FORMAT,
        resolve_package_root,
    )
    from aspect.data.code import (
        sha256_file as code_sha256_file,
    )

    package_root = resolve_package_root(Path(args.train_root))
    manifest_path = package_root / "manifest.json"
    calibration_path = package_root / "calibration.parquet"
    success_path = package_root / "_SUCCESS"
    if not manifest_path.is_file() or not calibration_path.is_file() or not success_path.is_file():
        raise ValueError(f"Incomplete pinned DeepCoder package at {package_root}")
    package_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if package_manifest.get("format") != PACKAGE_FORMAT:
        raise ValueError(
            f"Unexpected DeepCoder package format: {package_manifest.get('format')}"
        )
    expected_sha = package_manifest["files"]["calibration.parquet"]
    actual_sha = code_sha256_file(calibration_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"DeepCoder calibration SHA mismatch: expected {expected_sha}, got {actual_sha}"
        )
    calibration = load_dataset(
        "parquet", data_files=[str(calibration_path)], split="train"
    )
    if len(calibration) != CALIBRATION_ROWS:
        raise ValueError(
            f"DeepCoder calibration rows changed: {len(calibration)}"
        )
    if args.calibration_samples > len(calibration):
        raise ValueError(
            f"Requested {args.calibration_samples} Code calibration rows from "
            f"{len(calibration)}"
        )
    selected = calibration.select(range(args.calibration_samples))
    rows = [
        {
            "question": str(example["question"]),
            "solution": str(example["solution"]),
        }
        for example in selected
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
            + b"\n"
        )
    return rows, {
        "task": "code",
        "package_root": str(package_root.resolve()),
        "package_manifest_sha256": code_sha256_file(manifest_path),
        "calibration_file": str(calibration_path.resolve()),
        "calibration_file_sha256": actual_sha,
        "available_rows": len(calibration),
        "selected_rows_sha256": digest.hexdigest(),
    }


def _load_calibration_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if args.task == "code":
        return _load_code_calibration_rows(args)
    return _load_math_calibration_rows(args)


def _supervised_example(tokenizer, row: dict[str, str], max_length: int) -> dict[str, torch.Tensor]:
    user_messages = [{"role": "user", "content": row["question"]}]
    full_messages = [
        *user_messages,
        {"role": "assistant", "content": row["solution"]},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        user_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_length = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids, strict=False):
        if prompt_token != full_token:
            break
        prompt_length += 1
    if prompt_length != len(prompt_ids):
        raise ValueError(
            "Calibration chat template changed: generation prompt is not a prefix "
            "of the supervised conversation"
        )

    target_ids = full_ids[prompt_length:]
    if not target_ids:
        raise ValueError("Calibration example has no assistant target tokens")

    # Code questions can exceed the calibration context by themselves. Keep the
    # supervised answer first, then fill the remaining context with the nearest
    # (rightmost) prompt tokens. This preserves a real gradient for every pinned
    # calibration row instead of silently turning long examples into zero loss.
    retained_target = target_ids[:max_length]
    prompt_budget = max_length - len(retained_target)
    retained_prompt = prompt_ids[-prompt_budget:] if prompt_budget else []
    full_ids = retained_prompt + retained_target
    labels = [-100] * len(retained_prompt) + retained_target
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def _candidate_modules(peft_model: torch.nn.Module) -> list[tuple[str, torch.nn.Module, torch.nn.Parameter]]:
    candidates = []
    for name, module in peft_model.named_modules():
        if not (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and "default" in module.lora_A
            and "default" in module.lora_B
        ):
            continue
        base_layer = module.get_base_layer() if hasattr(module, "get_base_layer") else module.base_layer
        weight = base_layer.weight
        if weight.ndim != 2:
            raise ValueError(f"LoRA-SB target {name} is not a matrix: {tuple(weight.shape)}")
        candidates.append((name, module, weight))
    if not candidates:
        raise RuntimeError("PEFT selected no all-linear modules for LoRA-SB calibration")
    return candidates


def _atomic_write_artifact(
    target: Path,
    factors: dict[str, torch.Tensor],
    manifest: dict[str, Any],
) -> None:
    from safetensors.torch import save_file

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    temp.mkdir(parents=True)
    try:
        basis_path = temp / "basis.safetensors"
        save_file(factors, str(basis_path))
        manifest = {
            **manifest,
            "basis_file_size_bytes": basis_path.stat().st_size,
            "basis_sha256": sha256_file(basis_path),
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temp / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        if target.exists():
            raise FileExistsError(f"Basis target appeared during build: {target}")
        os.replace(temp, target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _build(args: argparse.Namespace) -> None:
    from peft import LoraConfig, TaskType, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    rank, world_size, local_rank, device = _distributed_context()
    target = Path(args.output_dir)
    work_root = Path(args.work_dir)

    reuse = _validate_existing(target, args)
    if target.exists() and not reuse:
        raise FileExistsError(f"Incomplete basis artifact already exists: {target}")
    if reuse:
        if rank == 0:
            print(f"[aw-lorasb-basis] reuse={target}", flush=True)
        return

    model_path = Path(args.model_path)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Mounted model is missing config.json: {model_path}")
    config_sha = sha256_file(config_path)
    if args.expected_config_sha256 and config_sha != args.expected_config_sha256:
        raise ValueError(
            f"Model config SHA mismatch: expected {args.expected_config_sha256}, got {config_sha}"
        )

    rows, data_manifest = _load_calibration_rows(args)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    model.config.use_cache = False
    peft_model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
            bias="none",
        ),
    )
    candidates = _candidate_modules(peft_model)
    for parameter in peft_model.parameters():
        parameter.requires_grad_(False)
    for _, _, weight in candidates:
        weight.requires_grad_(True)
    peft_model.train()

    gradient_sums = {
        name: torch.zeros_like(weight, dtype=torch.float32, device=device)
        for name, _, weight in candidates
    }
    calibration_totals = torch.zeros(4, dtype=torch.float64, device=device)
    local_rows = rows[rank::world_size]
    started = time.time()
    for sample_index, row in enumerate(local_rows, start=1):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in _supervised_example(
                tokenizer,
                row,
                args.max_seq_length,
            ).items()
        }
        outputs = peft_model(**batch)
        loss_value = float(outputs.loss.detach().to(dtype=torch.float32).item())
        target_token_count = int(batch["labels"][:, 1:].ne(-100).sum().item())
        if not torch.isfinite(torch.tensor(loss_value)).item() or target_token_count <= 0:
            raise RuntimeError(
                "LoRA-SB calibration produced a non-finite loss or no predicted target tokens"
            )
        calibration_totals += torch.tensor(
            [loss_value, loss_value * target_token_count, 1.0, target_token_count],
            dtype=torch.float64,
            device=device,
        )
        outputs.loss.backward()
        for name, _, weight in candidates:
            if weight.grad is None:
                raise RuntimeError(f"LoRA-SB calibration produced no gradient for {name}")
            gradient_sums[name].add_(weight.grad.detach().to(dtype=torch.float32))
            weight.grad = None
        if sample_index == 1 or sample_index == len(local_rows):
            print(
                f"[aw-lorasb-basis][rank={rank}] samples={sample_index}/{len(local_rows)} "
                f"elapsed_s={time.time() - started:.1f}",
                flush=True,
            )

    if dist.is_initialized():
        for name, _, _ in candidates:
            dist.all_reduce(gradient_sums[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(calibration_totals, op=dist.ReduceOp.SUM)
    calibration_statistics = summarize_calibration_statistics(
        example_loss_sum=float(calibration_totals[0].item()),
        token_nll_sum=float(calibration_totals[1].item()),
        example_count=int(calibration_totals[2].item()),
        target_token_count=int(calibration_totals[3].item()),
    )

    assigned_names = {
        name
        for module_index, (name, _, _) in enumerate(candidates)
        if module_index % world_size == rank
    }
    for name in tuple(gradient_sums):
        if name not in assigned_names:
            del gradient_sums[name]
    torch.cuda.empty_cache()

    effective_lr = args.actor_lr / args.warmup_steps
    local_factors: dict[str, torch.Tensor] = {}
    local_stats: dict[str, dict[str, Any]] = {}
    for module_index, (name, _, _) in enumerate(candidates):
        if module_index % world_size != rank:
            continue
        update = first_adamw_update_approximation(
            gradient_sums[name],
            effective_lr=effective_lr,
        )
        fixed_a, fixed_b, initial_core = truncated_svd_lorasb_factors(
            update,
            rank=args.rank,
            n_iter=args.n_iter,
            random_state=args.random_state,
        )
        key_a, key_b = basis_tensor_keys(name)
        local_factors[key_a] = fixed_a.to(device="cpu", dtype=torch.bfloat16)
        local_factors[key_b] = fixed_b.to(device="cpu", dtype=torch.bfloat16)
        local_factors[lorasb_core_tensor_key(name)] = initial_core.to(
            device="cpu", dtype=torch.bfloat16
        )
        full_energy = float(update.square().sum().item())
        singular_values = torch.diagonal(initial_core).detach().to(dtype=torch.float32)
        captured_energy = float(singular_values.square().sum().item())
        fixed_a_f32 = fixed_a.detach().to(dtype=torch.float32)
        fixed_b_f32 = fixed_b.detach().to(dtype=torch.float32)
        identity = torch.eye(args.rank, dtype=torch.float32, device=device)
        a_orthogonality_error = float(
            torch.linalg.vector_norm(fixed_a_f32 @ fixed_a_f32.T - identity).item()
        )
        b_orthogonality_error = float(
            torch.linalg.vector_norm(fixed_b_f32.T @ fixed_b_f32 - identity).item()
        )
        local_stats[name] = {
            "update_frobenius_norm": full_energy**0.5,
            "captured_update_frobenius_norm": captured_energy**0.5,
            "residual_update_frobenius_norm": max(full_energy - captured_energy, 0.0) ** 0.5,
            "rank_r_capture_ratio": captured_energy / max(full_energy, 1e-30),
            "top_r_singular_values": [float(value) for value in singular_values.tolist()],
            "a_orthogonality_error": a_orthogonality_error,
            "b_orthogonality_error": b_orthogonality_error,
        }
        del gradient_sums[name], update, fixed_a, fixed_b, initial_core

    work_root.mkdir(parents=True, exist_ok=True)
    shard_path = work_root / f"rank_{rank:02d}.safetensors"
    stats_path = work_root / f"rank_{rank:02d}.json"
    save_file(local_factors, str(shard_path))
    stats_path.write_text(json.dumps(local_stats, sort_keys=True) + "\n", encoding="utf-8")
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        factors: dict[str, torch.Tensor] = {}
        module_stats: dict[str, dict[str, float]] = {}
        for shard_rank in range(world_size):
            shard = work_root / f"rank_{shard_rank:02d}.safetensors"
            shard_stats = work_root / f"rank_{shard_rank:02d}.json"
            factors.update(load_file(str(shard), device="cpu"))
            module_stats.update(json.loads(shard_stats.read_text(encoding="utf-8")))
        module_names = [name for name, _, _ in candidates]
        expected_keys = {
            key
            for name in module_names
            for key in (*basis_tensor_keys(name), lorasb_core_tensor_key(name))
        }
        if set(factors) != expected_keys:
            raise RuntimeError(
                "LoRA-SB factor key mismatch: "
                f"missing={sorted(expected_keys.difference(factors))[:5]}, "
                f"extras={sorted(set(factors).difference(expected_keys))[:5]}"
            )
        update_energy = sum(
            float(stats["update_frobenius_norm"]) ** 2
            for stats in module_stats.values()
        )
        captured_energy = sum(
            float(stats["captured_update_frobenius_norm"]) ** 2
            for stats in module_stats.values()
        )
        singular_values = [
            float(value)
            for stats in module_stats.values()
            for value in stats["top_r_singular_values"]
        ]
        singular_spectrum = summarize_singular_spectrum(
            module_stats,
            rank=args.rank,
        )
        orthogonality_errors = [
            float(stats[key])
            for stats in module_stats.values()
            for key in ("a_orthogonality_error", "b_orthogonality_error")
        ]
        aggregate_update_statistics = {
            "update_frobenius_norm": update_energy**0.5,
            "captured_update_frobenius_norm": captured_energy**0.5,
            "residual_update_frobenius_norm": max(
                update_energy - captured_energy, 0.0
            )
            ** 0.5,
            "rank_r_capture_ratio": captured_energy / max(update_energy, 1e-30),
            "singular_value_max": max(singular_values, default=0.0),
            "singular_value_mean": (
                sum(singular_values) / len(singular_values) if singular_values else 0.0
            ),
            "singular_spectrum": singular_spectrum,
            "basis_orthogonality_error_max": max(orthogonality_errors, default=0.0),
        }
        manifest = {
            "policy_mode": POLICY_MODE,
            "diagnostics_schema_version": 1,
            "created_unix": time.time(),
            "model_path_at_creation": str(model_path),
            "model_revision": args.model_revision,
            "model_config_sha256": config_sha,
            "rank": args.rank,
            "lora_alpha": args.lora_alpha,
            "adapter_scale": args.lora_alpha / args.rank,
            "target_modules": "all-linear",
            "target_module_count": len(module_names),
            "module_names": module_names,
            "tensor_count": len(factors),
            "basis_value_count": sum(tensor.numel() for tensor in factors.values()),
            "basis_dtype": "bfloat16",
            "reconstruction_type": "update_approximation_svd",
            "svd_backend": SVD_BACKEND,
            "n_iter": args.n_iter,
            "random_state": args.random_state,
            "actor_learning_rate": args.actor_lr,
            "warmup_steps": args.warmup_steps,
            "effective_first_step_learning_rate": effective_lr,
            "update_approximation": "-effective_lr*sign(sum_supervised_gradients)",
            "calibration_samples": args.calibration_samples,
            "calibration_statistics": calibration_statistics,
            "max_seq_length": args.max_seq_length,
            "calibration_objective": (
                "causal_lm_supervised_loss_on_deepcoder_reference_solution"
                if args.task == "code"
                else "causal_lm_supervised_loss_on_dapo_solution"
            ),
            "calibration_task": args.task,
            "calibration_data": data_manifest,
            "module_statistics": module_stats,
            "aggregate_update_statistics": aggregate_update_statistics,
            "core_initialization": "R=diag(top_r_singular_values)",
            "official_lorasb_repository": OFFICIAL_LORASB_REPOSITORY,
            "official_lorasb_commit": OFFICIAL_LORASB_COMMIT,
            "world_size": world_size,
            "elapsed_s": time.time() - started,
        }
        _atomic_write_artifact(target, factors, manifest)
        print(
            "[aw-lorasb-basis] complete="
            + json.dumps(
                {
                    "path": str(target),
                    "target_module_count": len(module_names),
                    "calibration_samples": args.calibration_samples,
                    "elapsed_s": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        shutil.rmtree(work_root, ignore_errors=True)
    if dist.is_initialized():
        dist.barrier()


def main() -> None:
    args = _parse_args()
    _validate_contract(args)
    _build(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[aw-lorasb-basis] ERROR: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
