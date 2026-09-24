"""Agent-wise LoRA-SB on top of the routed square-core implementation.

LoRA-SB approximates the first full-finetuning update ``Delta W`` and uses its
rank-r SVD to initialize an orthonormal frozen basis and a trainable core::

    Delta W ~= B_fixed @ R_initial @ A_fixed
    B_fixed.T @ B_fixed = I
    A_fixed @ A_fixed.T = I

For the multi-agent workflow every route receives an independent copy of
``R_initial``.  The frozen base model and frozen A/B factors are shared; only
the route-local R matrices are optimized.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from verl.workers.actor.agent_loraxs import (
    AgentLoRAXSOutputProjection as AgentLoRAXSOutputProjection,
)
from verl.workers.actor.agent_loraxs import (
    InstallationReport,
    canonical_module_name,
    collect_agent_loraxs_core_state,
    install_agent_loraxs,
    iter_agent_loraxs_layers,
    load_agent_loraxs_core_state_,
    materialize_agent_loraxs_adapters,
    materialize_agent_loraxs_checkpoint,
    set_agent_loraxs_route,
    sha256_file,
    validate_routes,
)
from verl.workers.actor.agent_loraxs import (
    load_basis_artifact as _load_basis_artifact,
)
from verl.workers.actor.agent_loraxs import (
    save_core_checkpoint_atomic as _save_core_checkpoint_atomic,
)
from verl.workers.actor.agent_loraxs import (
    save_core_state_atomic as _save_core_state_atomic,
)

POLICY_MODE = "agentwise_lorasb"
SVD_BACKEND = "torch_svd_lowrank_official_lorasb"
OFFICIAL_LORASB_REPOSITORY = "CERT-Lab/lora-sb"
OFFICIAL_LORASB_COMMIT = "4feb81c243e6e762c64b736c968b232caf7b44c4"


def extract_lorasb_initial_core_state(
    basis_tensors: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Copy the immutable LoRA-SB ``R_0`` tensors to CPU for diagnostics."""

    initial = {
        str(key): value.detach().to(device="cpu", dtype=torch.float32).clone()
        for key, value in basis_tensors.items()
        if str(key).endswith(".lora_R.weight")
    }
    if not initial:
        raise ValueError("AW-LoRA-SB basis contains no initial R tensors")
    if any(tensor.ndim != 2 for tensor in initial.values()):
        raise ValueError("AW-LoRA-SB initial R tensors must be matrices")
    return initial


def load_agent_lorasb_checkpoint_(
    model: nn.Module,
    checkpoint_path: str | os.PathLike[str],
    *,
    basis_manifest: Mapping[str, object],
    expected_global_step: int,
    expected_routes: Sequence[str],
) -> dict[str, object]:
    """Load one atomic R-only checkpoint after validating its full contract."""

    from safetensors.torch import load_file

    root = Path(checkpoint_path)
    manifest_path = root / "manifest.json"
    cores_path = root / "cores.safetensors"
    success_path = root / "_SUCCESS"
    missing = [
        str(path)
        for path in (manifest_path, cores_path, success_path)
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete AW-LoRA-SB warm-start checkpoint at {root}: {missing}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    routes = tuple(str(route) for route in manifest.get("routes", ()))
    expected_routes = tuple(str(route) for route in expected_routes)
    checks = {
        "policy_mode": manifest.get("policy_mode") == POLICY_MODE,
        "global_step": int(manifest.get("global_step", -1))
        == int(expected_global_step),
        "routes": routes == expected_routes,
        "rank": int(manifest.get("rank", -1))
        == int(basis_manifest.get("rank", -2)),
        "lora_alpha": int(manifest.get("lora_alpha", -1))
        == int(basis_manifest.get("lora_alpha", -2)),
        "basis_sha256": manifest.get("basis_sha256")
        == basis_manifest.get("basis_sha256"),
        "cores_sha256": manifest.get("cores_sha256") == sha256_file(cores_path),
        "storage_contract": manifest.get("storage_contract")
        == "R_only_no_base_no_optimizer_no_scheduler_no_dataloader",
    }
    if not all(checks.values()):
        raise ValueError(
            "AW-LoRA-SB warm-start checkpoint contract mismatch: "
            f"{json.dumps(checks, sort_keys=True)}"
        )

    state = dict(load_file(str(cores_path), device="cpu"))
    expected: dict[str, torch.Size] = {}
    has_meta_core = False
    for name, _, _, output_projection in iter_agent_lorasb_layers(model):
        for route in output_projection.routes:
            key = f"{name}.cores.{route}.weight"
            weight = output_projection.core_router.cores[route].weight
            expected[key] = weight.shape
            has_meta_core = has_meta_core or weight.is_meta
    if not expected:
        raise ValueError("Model does not contain installed AW-LoRA-SB layers")
    missing_keys = set(expected).difference(state)
    extra_keys = set(state).difference(expected)
    if missing_keys or extra_keys:
        raise ValueError(
            "AW-LoRA-SB warm-start core keys do not match the installed model: "
            f"missing={sorted(missing_keys)[:5]}, extras={sorted(extra_keys)[:5]}"
        )
    shape_mismatches = {
        key: (tuple(expected[key]), tuple(state[key].shape))
        for key in expected
        if state[key].shape != expected[key]
    }
    if shape_mismatches:
        raise ValueError(
            "AW-LoRA-SB warm-start core shapes do not match the installed model: "
            f"{dict(list(shape_mismatches.items())[:5])}"
        )

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    process_rank = torch.distributed.get_rank() if distributed else 0
    if has_meta_core and process_rank == 0:
        raise RuntimeError("AW-LoRA-SB source rank unexpectedly contains meta R cores")

    local_tensor_exact: bool | None = None
    if not has_meta_core:
        load_agent_lorasb_core_state_(model, state)
        loaded = collect_agent_lorasb_core_state(model)
        local_tensor_exact = set(loaded) == set(state) and all(
            torch.equal(
                loaded[key].detach().to(device="cpu"),
                state[key].to(dtype=loaded[key].dtype),
            )
            for key in state
        )
        if not local_tensor_exact:
            raise RuntimeError("AW-LoRA-SB warm-start R cores failed exact load audit")

    source_rank_audit = [local_tensor_exact if process_rank == 0 else None]
    if distributed:
        torch.distributed.broadcast_object_list(source_rank_audit, src=0)
    if source_rank_audit[0] is not True:
        raise RuntimeError("AW-LoRA-SB source-rank warm-start audit did not pass")

    return {
        "checkpoint_path": str(root),
        "global_step": int(expected_global_step),
        "routes": list(routes),
        "basis_sha256": manifest.get("basis_sha256"),
        "cores_sha256": manifest.get("cores_sha256"),
        "tensor_count": len(state),
        "tensor_exact": True,
        "source_rank": 0,
        "source_rank_tensor_exact": True,
        "local_pre_fsdp_load_applied": not has_meta_core,
        "meta_rank_deferred_to_fsdp_sync": has_meta_core,
        "optimizer_state_restored": False,
        "scheduler_state_restored": False,
        "dataloader_state_restored": False,
    }


def _core_state_by_route(
    state: Mapping[str, torch.Tensor],
    routes: Sequence[str],
) -> dict[str, dict[str, torch.Tensor]]:
    routed: dict[str, dict[str, torch.Tensor]] = {route: {} for route in routes}
    for raw_key, tensor in state.items():
        key = str(raw_key)
        matched_route = next(
            (
                route
                for route in routes
                if key.endswith(f".cores.{route}.weight")
            ),
            None,
        )
        if matched_route is None:
            raise ValueError(f"Unrecognized AW-LoRA-SB core key {key!r}")
        module_name = key[: -len(f".cores.{matched_route}.weight")]
        if module_name in routed[matched_route]:
            raise ValueError(
                f"Duplicate AW-LoRA-SB core for route={matched_route!r}, "
                f"module={module_name!r}"
            )
        routed[matched_route][module_name] = tensor.detach().to(
            device="cpu", dtype=torch.float64
        )
    return routed


def _effective_count(shares: Sequence[float], eps: float) -> float:
    positive = [float(share) for share in shares if float(share) > eps]
    return math.exp(-sum(share * math.log(share) for share in positive)) if positive else 0.0


@torch.no_grad()
def compute_agent_lorasb_mechanism_metrics(
    core_state: Mapping[str, torch.Tensor],
    initial_core_state: Mapping[str, torch.Tensor],
    *,
    routes: Sequence[str],
    adapter_scale: float,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Summarize route differentiation without serializing any dense update.

    LoRA-SB's frozen bases are orthonormal, so
    ``||scale * B R A||_F = |scale| * ||R||_F``.  The effective-update norms
    below are therefore exact weight-space Frobenius norms, not proxies.
    """

    normalized_routes = validate_routes(routes)
    if not math.isfinite(float(adapter_scale)):
        raise ValueError(f"adapter_scale must be finite, got {adapter_scale}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    initial = {
        str(key)[: -len(".lora_R.weight")]: tensor.detach().to(
            device="cpu", dtype=torch.float64
        )
        for key, tensor in initial_core_state.items()
        if str(key).endswith(".lora_R.weight")
    }
    if not initial:
        raise ValueError("AW-LoRA-SB diagnostics require immutable R_0 tensors")
    routed = _core_state_by_route(core_state, normalized_routes)
    expected_modules = set(initial)
    for route, modules in routed.items():
        if set(modules) != expected_modules:
            raise ValueError(
                f"AW-LoRA-SB route {route!r} has a different module set: "
                f"missing={sorted(expected_modules.difference(modules))[:5]}, "
                f"extra={sorted(set(modules).difference(expected_modules))[:5]}"
            )

    initial_norm_sq = sum(float(tensor.square().sum().item()) for tensor in initial.values())
    initial_norm = math.sqrt(max(initial_norm_sq, 0.0))
    route_norms: dict[str, float] = {}
    route_drift_norms: dict[str, float] = {}
    metrics: dict[str, float] = {
        "paper_diag/lorasb/core/initial_norm": initial_norm,
        "paper_diag/lorasb/core/module_count": float(len(initial)),
        "paper_diag/lorasb/core/route_count": float(len(normalized_routes)),
    }

    for route in normalized_routes:
        norm_sq = 0.0
        drift_sq = 0.0
        initial_dot = 0.0
        for module_name, initial_tensor in initial.items():
            current = routed[route][module_name]
            if current.shape != initial_tensor.shape:
                raise ValueError(
                    f"AW-LoRA-SB core shape mismatch for {module_name!r}: "
                    f"current={tuple(current.shape)}, initial={tuple(initial_tensor.shape)}"
                )
            norm_sq += float(current.square().sum().item())
            drift_sq += float((current - initial_tensor).square().sum().item())
            initial_dot += float((current * initial_tensor).sum().item())
        norm = math.sqrt(max(norm_sq, 0.0))
        drift = math.sqrt(max(drift_sq, 0.0))
        route_norms[route] = norm
        route_drift_norms[route] = drift
        prefix = f"paper_diag/lorasb/core/route/{route}"
        metrics[f"{prefix}/norm"] = norm
        metrics[f"{prefix}/drift_norm"] = drift
        metrics[f"{prefix}/relative_drift"] = drift / max(initial_norm, eps)
        metrics[f"{prefix}/cosine_to_initial"] = (
            max(-1.0, min(1.0, initial_dot / (norm * initial_norm)))
            if norm > eps and initial_norm > eps
            else 0.0
        )

    pair_cosines: list[float] = []
    pair_distances: list[float] = []
    for left_index, left_route in enumerate(normalized_routes):
        for right_route in normalized_routes[left_index + 1 :]:
            dot = 0.0
            distance_sq = 0.0
            for module_name in initial:
                left = routed[left_route][module_name]
                right = routed[right_route][module_name]
                dot += float((left * right).sum().item())
                distance_sq += float((left - right).square().sum().item())
            left_norm = route_norms[left_route]
            right_norm = route_norms[right_route]
            cosine = (
                max(-1.0, min(1.0, dot / (left_norm * right_norm)))
                if left_norm > eps and right_norm > eps
                else 0.0
            )
            distance = math.sqrt(max(distance_sq, 0.0))
            pair = f"{left_route}__{right_route}"
            prefix = f"paper_diag/lorasb/core/pair/{pair}"
            metrics[f"{prefix}/cosine"] = cosine
            metrics[f"{prefix}/distance"] = distance
            metrics[f"{prefix}/relative_distance"] = distance / max(
                0.5 * (left_norm + right_norm), eps
            )
            pair_cosines.append(cosine)
            pair_distances.append(distance)
    metrics["paper_diag/lorasb/core/pair_count"] = float(len(pair_cosines))
    metrics["paper_diag/lorasb/core/pair_cosine_mean"] = (
        sum(pair_cosines) / len(pair_cosines) if pair_cosines else 0.0
    )
    metrics["paper_diag/lorasb/core/pair_distance_mean"] = (
        sum(pair_distances) / len(pair_distances) if pair_distances else 0.0
    )

    effective_norms = {
        route: abs(float(adapter_scale)) * norm for route, norm in route_norms.items()
    }
    effective_delta_norms = {
        route: abs(float(adapter_scale)) * drift
        for route, drift in route_drift_norms.items()
    }
    total_effective_norm = sum(effective_norms.values())
    shares = {
        route: norm / total_effective_norm if total_effective_norm > eps else 0.0
        for route, norm in effective_norms.items()
    }
    for route in normalized_routes:
        prefix = f"paper_diag/lorasb/effective_update/route/{route}"
        metrics[f"{prefix}/norm"] = effective_norms[route]
        metrics[f"{prefix}/delta_norm"] = effective_delta_norms[route]
        metrics[f"{prefix}/contribution_share"] = shares[route]
    metrics["paper_diag/lorasb/effective_update/norm_sum"] = total_effective_norm
    metrics["paper_diag/lorasb/effective_update/delta_norm_sum"] = sum(
        effective_delta_norms.values()
    )
    metrics["paper_diag/lorasb/effective_update/dominance"] = max(
        shares.values(), default=0.0
    )
    metrics["paper_diag/lorasb/effective_update/effective_routes"] = _effective_count(
        tuple(shares.values()), eps
    )
    return metrics


def summarize_lorasb_basis_manifest(manifest: Mapping[str, object]) -> dict[str, float]:
    """Flatten persisted calibration/SVD evidence into logger-safe scalars."""

    metrics: dict[str, float] = {}
    for source, target in (
        ("basis_file_size_bytes", "basis_file_size_bytes"),
        ("basis_value_count", "basis_value_count"),
        ("target_module_count", "target_module_count"),
    ):
        value = manifest.get(source)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics[f"paper_diag/lorasb/storage/{target}"] = float(value)
    calibration = manifest.get("calibration_statistics", {})
    if isinstance(calibration, Mapping):
        aliases = {
            "example_loss_mean": "calibration_loss",
            "token_nll_mean": "calibration_token_nll",
            "example_count": "calibration_example_count",
            "target_token_count": "calibration_target_token_count",
        }
        for source, target in aliases.items():
            value = calibration.get(source)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[f"paper_diag/lorasb/basis/{target}"] = float(value)

    aggregate = manifest.get("aggregate_update_statistics", {})
    if isinstance(aggregate, Mapping):
        aliases = {
            "update_frobenius_norm": "update_approximation_norm",
            "captured_update_frobenius_norm": "captured_update_norm",
            "residual_update_frobenius_norm": "residual_update_norm",
            "rank_r_capture_ratio": "explained_energy",
            "singular_value_max": "singular_value_max",
            "singular_value_mean": "singular_value_mean",
            "basis_orthogonality_error_max": "orthogonality_error_max",
        }
        for source, target in aliases.items():
            value = aggregate.get(source)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                metrics[f"paper_diag/lorasb/basis/{target}"] = float(value)
        spectrum = aggregate.get("singular_spectrum")
        if isinstance(spectrum, list):
            for rank_index, rank_stats in enumerate(spectrum, start=1):
                if not isinstance(rank_stats, Mapping):
                    continue
                for source, target in (
                    ("singular_value_mean", "singular_value_mean"),
                    ("cumulative_energy_ratio", "cumulative_energy_ratio"),
                ):
                    value = rank_stats.get(source)
                    if isinstance(value, (int, float)) and math.isfinite(
                        float(value)
                    ):
                        metrics[
                            f"paper_diag/lorasb/basis/spectrum/rank_{rank_index:03d}/{target}"
                        ] = float(value)
    return metrics


def lorasb_core_tensor_key(module_name: str) -> str:
    return f"{canonical_module_name(module_name)}.lora_R.weight"


def truncated_svd_lorasb_factors(
    update: torch.Tensor,
    *,
    rank: int,
    n_iter: int = 10,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return LoRA-SB ``A``, ``B`` and ``R`` for a full-weight update.

    This follows the official implementation's ``torch.svd_lowrank`` path.
    For a linear weight/update shaped ``[out_features, in_features]`` the
    returned tensors have shapes ``A=[r, in]``, ``B=[out, r]`` and
    ``R=[r, r]``.  The formal default uses ``lora_alpha == rank`` so the
    adapter scale is one.  Explicit scale diagnostics keep the same factors
    and apply ``lora_alpha / rank`` through the standard PEFT path.
    """

    if update.ndim != 2:
        raise ValueError(f"Weight update must be two-dimensional, got {tuple(update.shape)}")
    if rank <= 0 or rank > min(update.shape):
        raise ValueError(f"Invalid rank {rank} for update shape {tuple(update.shape)}")
    if n_iter <= 0:
        raise ValueError(f"n_iter must be positive, got {n_iter}")

    source_dtype = update.dtype
    matrix = update.detach().to(dtype=torch.float32)
    cuda_devices = [matrix.device.index] if matrix.is_cuda else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(random_state))
        left, singular_values, right = torch.svd_lowrank(
            matrix,
            q=int(rank),
            niter=int(n_iter),
        )

    fixed_a = right.T.to(dtype=source_dtype).contiguous()
    fixed_b = left.to(dtype=source_dtype).contiguous()
    initial_core = torch.diag(singular_values).to(dtype=source_dtype).contiguous()
    return fixed_a, fixed_b, initial_core


def install_agent_lorasb(
    peft_model: nn.Module,
    basis_tensors: Mapping[str, torch.Tensor],
    *,
    routes: Sequence[str],
    rank: int,
    adapter_name: str = "default",
) -> InstallationReport:
    """Install frozen LoRA-SB bases and independently trainable route cores."""

    factor_tensors: dict[str, torch.Tensor] = {}
    core_tensors: dict[str, torch.Tensor] = {}
    for key, value in basis_tensors.items():
        if key.endswith(".lora_R.weight"):
            core_tensors[key] = value
        else:
            factor_tensors[key] = value

    if not core_tensors:
        raise ValueError("AW-LoRA-SB basis contains no singular-value R initializers")
    return install_agent_loraxs(
        peft_model,
        factor_tensors,
        routes=routes,
        rank=rank,
        core_tensors=core_tensors,
        policy_mode=POLICY_MODE,
        algorithm_label="AW-LoRA-SB",
        adapter_name=adapter_name,
    )


def load_basis_artifact(path: str | os.PathLike[str]) -> tuple[dict[str, torch.Tensor], dict]:
    return _load_basis_artifact(path, expected_policy_mode=POLICY_MODE)


def materialize_agent_lorasb_checkpoint(
    basis_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    base_model_name_or_path: str,
) -> dict[str, object]:
    return materialize_agent_loraxs_checkpoint(
        basis_path,
        checkpoint_path,
        output_path,
        base_model_name_or_path=base_model_name_or_path,
        expected_policy_mode=POLICY_MODE,
        algorithm_label="AW-LoRA-SB",
    )


def save_core_checkpoint_atomic(
    model: nn.Module,
    target_dir: str | os.PathLike[str],
    *,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return _save_core_checkpoint_atomic(
        model,
        target_dir,
        metadata=metadata,
        policy_mode=POLICY_MODE,
    )


def save_core_state_atomic(
    state: Mapping[str, torch.Tensor],
    target_dir: str | os.PathLike[str],
    *,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return _save_core_state_atomic(
        state,
        target_dir,
        metadata=metadata,
        policy_mode=POLICY_MODE,
    )


# The routed module mechanics are identical; these aliases make the algorithm
# boundary explicit without duplicating the FSDP-safe implementation.
collect_agent_lorasb_core_state = collect_agent_loraxs_core_state
iter_agent_lorasb_layers = iter_agent_loraxs_layers
load_agent_lorasb_core_state_ = load_agent_loraxs_core_state_
materialize_agent_lorasb_adapters = materialize_agent_loraxs_adapters
set_agent_lorasb_route = set_agent_loraxs_route
