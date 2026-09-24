"""Agent-wise LoRA-XS modules and serialization helpers.

AW-LoRA-XS keeps one frozen SVD basis per adapted linear layer and gives each
workflow agent instance its own trainable square core::

    Delta W_i = scale * B_fixed @ R_i @ A_fixed

``A_fixed`` and ``B_fixed`` are buffers, not parameters.  Every ``R_i`` is a
separate parameter so an optimizer step for one route cannot apply gradients or
AdamW decay to another route.  The helpers in this module deliberately keep
the mathematical representation independent from vLLM: serving adapters are
materialized as ``A_fixed`` and ``B_fixed @ R_i`` only at synchronization time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

POLICY_MODE = "agentwise_loraxs"
DEFAULT_ADAPTER = "default"
CORE_INIT_STD = 1e-5
SVD_BACKEND = "vendored_sklearn_compatible_randomized_truncated_svd"
_ROUTE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_routes(routes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(route) for route in routes)
    if not normalized:
        raise ValueError("AW-LoRA-XS requires at least one agent route")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"AW-LoRA-XS routes must be unique: {normalized!r}")
    invalid = [route for route in normalized if _ROUTE_PATTERN.fullmatch(route) is None]
    if invalid:
        raise ValueError(
            "AW-LoRA-XS route names may contain only letters, digits, '_' and '-'; "
            f"got {invalid!r}"
        )
    return normalized


def canonical_module_name(name: str) -> str:
    """Remove FSDP wrapper path components from a module name."""

    parts = [part for part in str(name).split(".") if part != "_fsdp_wrapped_module"]
    return ".".join(parts)


def basis_tensor_keys(module_name: str) -> tuple[str, str]:
    canonical = canonical_module_name(module_name)
    return f"{canonical}.lora_A.weight", f"{canonical}.lora_B.weight"


def truncated_svd_loraxs_factors(
    weight: torch.Tensor,
    *,
    rank: int,
    n_iter: int = 10,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the official LoRA-XS randomized-SVD factor orientation.

    The official implementation applies ``TruncatedSVD`` to ``weight.T``.
    For a PyTorch linear weight ``W`` with shape ``[out, in]`` this returns
    ``A_fixed = Sigma V^T`` with shape ``[rank, in]`` and
    ``B_fixed = U`` with shape ``[out, rank]``.  Consequently
    ``B_fixed @ R @ A_fixed`` has the same orientation as ``W``.
    """

    if weight.ndim != 2:
        raise ValueError(f"Linear weight must be two-dimensional, got {tuple(weight.shape)}")
    if rank <= 0 or rank > min(weight.shape):
        raise ValueError(f"Invalid rank {rank} for weight shape {tuple(weight.shape)}")
    if n_iter <= 0:
        raise ValueError(f"n_iter must be positive, got {n_iter}")

    import numpy as np
    from scipy import linalg

    # This is a compact port of sklearn's randomized TruncatedSVD path.  It is
    # kept here because the training image intentionally has scipy but not
    # scikit-learn.  The operations, defaults (10 oversamples, LU power
    # normalization), RandomState stream and transform convention match the
    # LoRA-XS dependency, without introducing a network-time package install.
    source_dtype = weight.dtype
    transposed = weight.detach().to(device="cpu", dtype=torch.float32).T.contiguous().numpy()
    matrix = transposed
    transpose_algorithm = matrix.shape[0] < matrix.shape[1]
    if transpose_algorithm:
        matrix = matrix.T

    random = np.random.RandomState(int(random_state))
    projection = random.normal(size=(matrix.shape[1], int(rank) + 10)).astype(
        matrix.dtype,
        copy=False,
    )
    for _ in range(int(n_iter)):
        projection, _ = linalg.lu(
            matrix @ projection,
            permute_l=True,
            check_finite=False,
        )
        projection, _ = linalg.lu(
            matrix.T @ projection,
            permute_l=True,
            check_finite=False,
        )
    projection, _ = linalg.qr(
        matrix @ projection,
        mode="economic",
        check_finite=False,
    )
    compressed = projection.T @ matrix
    compressed_u, _, compressed_vt = linalg.svd(
        compressed,
        full_matrices=False,
        lapack_driver="gesdd",
        check_finite=False,
    )
    left = projection @ compressed_u

    if transpose_algorithm:
        components = left[:, : int(rank)].T
    else:
        components = compressed_vt[: int(rank), :]

    # TruncatedSVD calls randomized_svd(..., flip_sign=False), then resolves
    # signs from the returned right singular vectors (u_based_decision=False).
    # Repeating that two-stage convention is necessary to match the official
    # LoRA-XS A/B factors, not merely their reconstructed product.
    indices = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), indices])
    signs[signs == 0] = 1
    components *= signs[:, np.newaxis]
    reduced = transposed @ components.T
    fixed_a = torch.from_numpy(reduced.T.copy()).to(dtype=source_dtype)
    fixed_b = torch.from_numpy(components.T.copy()).to(dtype=source_dtype)
    return fixed_a.contiguous(), fixed_b.contiguous()


def _tensor_for_reference(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if reference.is_meta:
        return torch.empty(tensor.shape, dtype=reference.dtype, device="meta")
    return tensor.detach().to(device=reference.device, dtype=reference.dtype).contiguous()


def _stable_module_seed(seed: int, module_name: str) -> int:
    digest = hashlib.sha256(module_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big")
    return (int(seed) + offset) % (2**31 - 1)


class FixedInputProjection(nn.Module):
    """Frozen LoRA-XS input factor with shape ``[rank, in_features]``."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"A_fixed must be a matrix, got {tuple(weight.shape)}")
        self.register_buffer("weight", weight.detach().contiguous(), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight.dtype != inputs.dtype:
            weight = weight.to(dtype=inputs.dtype)
        return F.linear(inputs, weight)


class AgentLoRAXSCore(nn.Module):
    """One independently optimized ``R_i`` matrix."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError(f"R_i must be square, got {tuple(weight.shape)}")
        self.weight = nn.Parameter(weight.detach().contiguous())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight.dtype != inputs.dtype:
            weight = weight.to(dtype=inputs.dtype)
        return F.linear(inputs, weight)


class AgentLoRAXSCoreRouter(nn.Module):
    """Route activations through separate core parameters.

    Separate child modules are intentional.  With FSDP and AdamW, an inactive
    child then has ``grad is None`` and receives neither a gradient update nor
    decoupled weight decay.
    """

    def __init__(self, routes: Sequence[str], template: torch.Tensor):
        super().__init__()
        self.routes = validate_routes(routes)
        self.cores = nn.ModuleDict(
            {route: AgentLoRAXSCore(template.clone()) for route in self.routes}
        )
        self.active_route = self.routes[0]

    def set_route(self, route: str) -> None:
        if route not in self.cores:
            raise KeyError(
                f"Unknown AW-LoRA-XS route {route!r}; available routes: {self.routes}"
            )
        self.active_route = route

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.cores[self.active_route](inputs)


class AgentLoRAXSOutputProjection(nn.Module):
    """Trainable core followed by the frozen output factor."""

    def __init__(
        self,
        basis: torch.Tensor,
        routes: Sequence[str],
        core_template: torch.Tensor,
    ):
        super().__init__()
        if basis.ndim != 2:
            raise ValueError(f"B_fixed must be a matrix, got {tuple(basis.shape)}")
        if basis.shape[1] != core_template.shape[0]:
            raise ValueError(
                "B_fixed and R_i rank mismatch: "
                f"{tuple(basis.shape)} vs {tuple(core_template.shape)}"
            )
        self.register_buffer("basis", basis.detach().contiguous(), persistent=False)
        self.core_router = AgentLoRAXSCoreRouter(routes, core_template)
        self.bias = None

    @property
    def routes(self) -> tuple[str, ...]:
        return self.core_router.routes

    def set_route(self, route: str) -> None:
        self.core_router.set_route(route)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.core_router(inputs)
        basis = self.basis
        if basis.dtype != hidden.dtype:
            basis = basis.to(dtype=hidden.dtype)
        return F.linear(hidden, basis)


@dataclass(frozen=True)
class InstallationReport:
    policy_mode: str
    routes: tuple[str, ...]
    target_module_count: int
    rank: int
    trainable_parameter_count: int
    frozen_basis_value_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "policy_mode": self.policy_mode,
            "routes": list(self.routes),
            "target_module_count": self.target_module_count,
            "rank": self.rank,
            "trainable_parameter_count": self.trainable_parameter_count,
            "frozen_basis_value_count": self.frozen_basis_value_count,
        }


def _is_lora_linear(module: nn.Module, adapter_name: str = DEFAULT_ADAPTER) -> bool:
    return bool(
        hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and adapter_name in module.lora_A
        and adapter_name in module.lora_B
    )


def iter_agent_loraxs_layers(
    model: nn.Module,
    adapter_name: str = DEFAULT_ADAPTER,
) -> Iterable[tuple[str, nn.Module, FixedInputProjection, AgentLoRAXSOutputProjection]]:
    for name, module in model.named_modules():
        if not _is_lora_linear(module, adapter_name):
            continue
        input_projection = module.lora_A[adapter_name]
        output_projection = module.lora_B[adapter_name]
        if isinstance(input_projection, FixedInputProjection) and isinstance(
            output_projection, AgentLoRAXSOutputProjection
        ):
            yield canonical_module_name(name), module, input_projection, output_projection


def _initial_core_template(
    *,
    rank: int,
    seed: int,
    module_name: str,
    std: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    if reference.is_meta:
        return torch.empty((rank, rank), dtype=reference.dtype, device="meta")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_module_seed(seed, module_name))
    template = torch.empty((rank, rank), dtype=torch.float32, device="cpu")
    template.normal_(mean=0.0, std=float(std), generator=generator)
    return template.to(device=reference.device, dtype=reference.dtype)


def install_agent_loraxs(
    peft_model: nn.Module,
    basis_tensors: Mapping[str, torch.Tensor],
    *,
    routes: Sequence[str],
    rank: int,
    seed: int = 42,
    core_init_std: float = CORE_INIT_STD,
    core_tensors: Mapping[str, torch.Tensor] | None = None,
    policy_mode: str = POLICY_MODE,
    algorithm_label: str = "AW-LoRA-XS",
    adapter_name: str = DEFAULT_ADAPTER,
) -> InstallationReport:
    """Replace a vanilla PEFT adapter with frozen bases and private cores."""

    routes = validate_routes(routes)
    if int(rank) <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if core_tensors is None and float(core_init_std) <= 0:
        raise ValueError(f"core_init_std must be positive, got {core_init_std}")

    for parameter in peft_model.parameters():
        parameter.requires_grad_(False)

    candidates = [
        (name, module)
        for name, module in peft_model.named_modules()
        if _is_lora_linear(module, adapter_name)
    ]
    if not candidates:
        raise ValueError("No PEFT LoRA linear modules were found for AW-LoRA-XS")

    used_keys: set[str] = set()
    frozen_values = 0
    for raw_name, module in candidates:
        name = canonical_module_name(raw_name)
        key_a, key_b = basis_tensor_keys(name)
        if key_a not in basis_tensors or key_b not in basis_tensors:
            raise KeyError(
                f"SVD basis is missing {name!r}; required keys: {key_a!r}, {key_b!r}"
            )
        used_keys.update((key_a, key_b))

        old_a = module.lora_A[adapter_name].weight
        old_b = module.lora_B[adapter_name].weight
        fixed_a = _tensor_for_reference(basis_tensors[key_a], old_a)
        fixed_b = _tensor_for_reference(basis_tensors[key_b], old_b)
        if fixed_a.shape != old_a.shape or fixed_b.shape != old_b.shape:
            raise ValueError(
                f"Basis shape mismatch for {name}: expected A={tuple(old_a.shape)}, "
                f"B={tuple(old_b.shape)}; got A={tuple(fixed_a.shape)}, "
                f"B={tuple(fixed_b.shape)}"
            )
        if fixed_a.shape[0] != rank or fixed_b.shape[1] != rank:
            raise ValueError(
                f"Basis rank mismatch for {name}: configured {rank}, "
                f"A={tuple(fixed_a.shape)}, B={tuple(fixed_b.shape)}"
            )

        if core_tensors is None:
            core_template = _initial_core_template(
                rank=rank,
                seed=seed,
                module_name=name,
                std=core_init_std,
                reference=old_a,
            )
        else:
            core_key = f"{name}.lora_R.weight"
            if core_key not in core_tensors:
                raise KeyError(
                    f"{algorithm_label} basis is missing initial core {core_key!r}"
                )
            core_template = _tensor_for_reference(core_tensors[core_key], old_a)
            if tuple(core_template.shape) != (rank, rank):
                raise ValueError(
                    f"Initial core shape mismatch for {name}: expected {(rank, rank)}, "
                    f"got {tuple(core_template.shape)}"
                )
        module.lora_A[adapter_name] = FixedInputProjection(fixed_a)
        module.lora_B[adapter_name] = AgentLoRAXSOutputProjection(
            fixed_b,
            routes,
            core_template,
        )
        frozen_values += fixed_a.numel() + fixed_b.numel()

    extra_basis_keys = set(basis_tensors).difference(used_keys)
    if extra_basis_keys:
        raise ValueError(
            "SVD basis contains modules absent from this PEFT model; first extras: "
            f"{sorted(extra_basis_keys)[:5]}"
        )

    set_agent_loraxs_route(peft_model, routes[0])
    trainable = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    expected = len(candidates) * len(routes) * rank * rank
    if trainable_count != expected:
        raise RuntimeError(
            f"{algorithm_label} trainable parameter mismatch: expected {expected}, got {trainable_count}"
        )
    invalid_trainables = [
        name
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad and ".core_router.cores." not in name
    ]
    if invalid_trainables:
        raise RuntimeError(
            f"{algorithm_label} found trainable parameters outside R_i cores: "
            f"{invalid_trainables[:5]}"
        )

    peft_model._agent_loraxs_routes = routes
    peft_model._agent_loraxs_rank = int(rank)
    peft_model._agent_low_rank_policy_mode = str(policy_mode)
    return InstallationReport(
        policy_mode=str(policy_mode),
        routes=routes,
        target_module_count=len(candidates),
        rank=int(rank),
        trainable_parameter_count=trainable_count,
        frozen_basis_value_count=frozen_values,
    )


def set_agent_loraxs_route(model: nn.Module, route: str) -> None:
    layers = list(iter_agent_loraxs_layers(model))
    if not layers:
        raise ValueError("Model does not contain installed AW-LoRA-XS layers")
    for _, _, _, output_projection in layers:
        output_projection.set_route(route)
    model._agent_loraxs_active_route = route


def _unwrap_core(core: nn.Module) -> AgentLoRAXSCore:
    current = core
    seen: set[int] = set()
    while not isinstance(current, AgentLoRAXSCore):
        if id(current) in seen:
            break
        seen.add(id(current))
        wrapped = getattr(current, "_fsdp_wrapped_module", None)
        if wrapped is None:
            wrapped = getattr(current, "module", None)
        if wrapped is None:
            break
        current = wrapped
    if not isinstance(current, AgentLoRAXSCore):
        raise TypeError(f"Expected AgentLoRAXSCore, got {type(current)!r}")
    return current


@contextmanager
def _summon_agent_loraxs_model(model: nn.Module):
    """Summon a complete FSDP1 tree once, before reading any nested core.

    Summoning an individual child FSDP before the root's first forward marks
    that child as a root and corrupts FSDP lazy initialization.  All rollout
    materialization and checkpoint gathering therefore enter through the real
    root exactly once.
    """

    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        FSDP = None

    context = (
        FSDP.summon_full_params(model, recurse=True, writeback=False)
        if FSDP is not None and isinstance(model, FSDP)
        else nullcontext()
    )
    with context:
        yield


def _core_weight_in_summon_context(core: nn.Module) -> torch.Tensor:
    weight = _unwrap_core(core).weight.detach()
    if hasattr(weight, "full_tensor"):
        weight = weight.full_tensor()
    return weight.to("cpu").clone()


@torch.no_grad()
def collect_agent_loraxs_core_state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    with _summon_agent_loraxs_model(model):
        layers = list(iter_agent_loraxs_layers(model))
        if not layers:
            raise ValueError("Model does not contain installed AW-LoRA-XS layers")
        for name, _, _, output_projection in layers:
            for route in output_projection.routes:
                core = output_projection.core_router.cores[route]
                state[f"{name}.cores.{route}.weight"] = _core_weight_in_summon_context(
                    core
                )
    return state


@torch.no_grad()
def materialize_agent_loraxs_adapters(
    model: nn.Module,
) -> dict[str, OrderedDict[str, torch.Tensor]]:
    """Return standard PEFT A/B tensors for every vLLM route."""

    with _summon_agent_loraxs_model(model):
        layers = list(iter_agent_loraxs_layers(model))
        if not layers:
            raise ValueError("Model does not contain installed AW-LoRA-XS layers")
        routes = layers[0][3].routes
        adapters = {route: OrderedDict() for route in routes}
        for name, _, input_projection, output_projection in layers:
            key_a, key_b = basis_tensor_keys(name)
            fixed_a = input_projection.weight.detach().to("cpu").clone()
            fixed_b = output_projection.basis.detach().to("cpu", dtype=torch.float32)
            for route in routes:
                core = _core_weight_in_summon_context(
                    output_projection.core_router.cores[route]
                ).to(dtype=torch.float32)
                effective_b = (fixed_b @ core).to(dtype=input_projection.weight.dtype)
                adapters[route][key_a] = fixed_a
                adapters[route][key_b] = effective_b.contiguous()
    return adapters


def materialize_agent_loraxs_checkpoint(
    basis_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    base_model_name_or_path: str,
    expected_policy_mode: str = POLICY_MODE,
    algorithm_label: str = "AW-LoRA-XS",
) -> dict[str, object]:
    """Materialize an R-only checkpoint as temporary standard PEFT adapters.

    This path intentionally does not instantiate the frozen base model.  It
    reconstructs each route's serving factors directly from the immutable SVD
    artifact and the checkpointed cores::

        A_route = A_fixed
        B_route = B_fixed @ R_route

    The caller is expected to place ``output_path`` on node-local temporary
    storage and remove it after vLLM unloads the adapters.
    """

    from safetensors.torch import load_file, save_file

    basis_root = Path(basis_path)
    checkpoint_root = Path(checkpoint_path)
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite materialized adapters: {target}")

    basis_tensors, basis_manifest = load_basis_artifact(
        basis_root,
        expected_policy_mode=expected_policy_mode,
    )
    core_path = checkpoint_root / "cores.safetensors"
    manifest_path = checkpoint_root / "manifest.json"
    success_path = checkpoint_root / "_SUCCESS"
    if not core_path.is_file() or not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(
            f"Incomplete AW-LoRA-XS checkpoint at {checkpoint_root}; expected "
            "cores.safetensors, manifest.json and _SUCCESS"
        )

    checkpoint_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if checkpoint_manifest.get("policy_mode") != expected_policy_mode:
        raise ValueError(
            f"Unexpected checkpoint policy mode: {checkpoint_manifest.get('policy_mode')!r}"
        )
    actual_core_sha = sha256_file(core_path)
    if actual_core_sha != checkpoint_manifest.get("cores_sha256"):
        raise ValueError(
            f"{algorithm_label} core SHA mismatch: "
            f"expected {checkpoint_manifest.get('cores_sha256')}, got {actual_core_sha}"
        )
    if checkpoint_manifest.get("basis_sha256") != basis_manifest.get("basis_sha256"):
        raise ValueError(
            f"{algorithm_label} checkpoint/basis mismatch: "
            f"checkpoint={checkpoint_manifest.get('basis_sha256')}, "
            f"basis={basis_manifest.get('basis_sha256')}"
        )

    rank = int(checkpoint_manifest.get("rank", 0))
    lora_alpha = int(checkpoint_manifest.get("lora_alpha", 0))
    if rank != int(basis_manifest.get("rank", 0)) or rank <= 0:
        raise ValueError(
            f"{algorithm_label} rank mismatch: checkpoint={rank}, basis={basis_manifest.get('rank')}"
        )
    if lora_alpha != int(basis_manifest.get("lora_alpha", 0)) or lora_alpha <= 0:
        raise ValueError(
            f"{algorithm_label} alpha mismatch: "
            f"checkpoint={lora_alpha}, basis={basis_manifest.get('lora_alpha')}"
        )

    routes = validate_routes(checkpoint_manifest.get("routes", ()))
    module_names = tuple(str(name) for name in basis_manifest.get("module_names", ()))
    if not module_names:
        raise ValueError("AW-LoRA-XS basis manifest contains no module_names")
    core_tensors = dict(load_file(str(core_path), device="cpu"))
    expected_core_keys = {
        f"{module_name}.cores.{route}.weight"
        for module_name in module_names
        for route in routes
    }
    actual_core_keys = set(core_tensors)
    if actual_core_keys != expected_core_keys:
        missing = sorted(expected_core_keys.difference(actual_core_keys))[:5]
        extras = sorted(actual_core_keys.difference(expected_core_keys))[:5]
        raise ValueError(
            f"{algorithm_label} core key mismatch: missing={missing}, extras={extras}"
        )

    adapter_config = {
        "base_model_name_or_path": str(base_model_name_or_path),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "revision": None,
        "target_modules": basis_manifest.get("target_modules", "all-linear"),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    temp.mkdir(parents=True)
    try:
        adapter_summaries: dict[str, dict[str, object]] = {}
        for route in routes:
            adapter_state: OrderedDict[str, torch.Tensor] = OrderedDict()
            for module_name in module_names:
                key_a, key_b = basis_tensor_keys(module_name)
                if key_a not in basis_tensors or key_b not in basis_tensors:
                    raise KeyError(
                        f"SVD basis is missing tensors for module {module_name!r}"
                    )
                core_key = f"{module_name}.cores.{route}.weight"
                core = core_tensors[core_key]
                if tuple(core.shape) != (rank, rank):
                    raise ValueError(
                        f"Core shape mismatch for {core_key}: expected {(rank, rank)}, "
                        f"got {tuple(core.shape)}"
                    )
                fixed_a = basis_tensors[key_a]
                fixed_b = basis_tensors[key_b]
                if fixed_a.shape[0] != rank or fixed_b.shape[1] != rank:
                    raise ValueError(
                        f"Basis shape mismatch for {module_name}: "
                        f"A={tuple(fixed_a.shape)}, B={tuple(fixed_b.shape)}, rank={rank}"
                    )

                # Training materialization multiplies in fp32 and then casts to
                # the model/core dtype before handing the adapter to vLLM.
                adapter_dtype = core.dtype
                adapter_state[key_a] = fixed_a.to(dtype=adapter_dtype).contiguous()
                adapter_state[key_b] = (
                    fixed_b.to(dtype=torch.float32)
                    @ core.to(dtype=torch.float32)
                ).to(dtype=adapter_dtype).contiguous()

            adapter_dir = temp / f"lora_adapter_{route}"
            adapter_dir.mkdir()
            weights_path = adapter_dir / "adapter_model.safetensors"
            save_file(dict(adapter_state), str(weights_path))
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            adapter_summaries[route] = {
                "tensor_count": len(adapter_state),
                "weights_sha256": sha256_file(weights_path),
            }

        materialization_manifest = {
            "policy_mode": expected_policy_mode,
            "source_checkpoint": str(checkpoint_root),
            "global_step": int(checkpoint_manifest.get("global_step", -1)),
            "basis_sha256": basis_manifest["basis_sha256"],
            "cores_sha256": actual_core_sha,
            "routes": list(routes),
            "rank": rank,
            "lora_alpha": lora_alpha,
            "equation": "A=A_fixed; B=B_fixed@R_route; scale=lora_alpha/rank",
            "temporary_serving_artifact": True,
            "adapters": adapter_summaries,
        }
        (temp / "materialization_manifest.json").write_text(
            json.dumps(materialization_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, target)
        return materialization_manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def load_agent_loraxs_core_state_(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Load an R-only checkpoint into an unsharded model."""

    expected: set[str] = set()
    for name, _, _, output_projection in iter_agent_loraxs_layers(model):
        for route in output_projection.routes:
            key = f"{name}.cores.{route}.weight"
            expected.add(key)
            if key not in state:
                raise KeyError(f"AW-LoRA-XS checkpoint is missing {key!r}")
            core = _unwrap_core(output_projection.core_router.cores[route])
            value = state[key]
            if value.shape != core.weight.shape:
                raise ValueError(
                    f"Core shape mismatch for {key}: expected {tuple(core.weight.shape)}, "
                    f"got {tuple(value.shape)}"
                )
            with torch.no_grad():
                core.weight.copy_(value.to(device=core.weight.device, dtype=core.weight.dtype))
    extras = set(state).difference(expected)
    if extras:
        raise ValueError(f"Unexpected AW-LoRA-XS checkpoint keys: {sorted(extras)[:5]}")


def load_basis_artifact(
    path: str | os.PathLike[str],
    *,
    expected_policy_mode: str = POLICY_MODE,
) -> tuple[dict[str, torch.Tensor], dict]:
    from safetensors.torch import load_file

    root = Path(path)
    basis_path = root / "basis.safetensors"
    manifest_path = root / "manifest.json"
    success_path = root / "_SUCCESS"
    if not basis_path.is_file() or not manifest_path.is_file() or not success_path.is_file():
        raise FileNotFoundError(
            f"Incomplete AW-LoRA-XS basis artifact at {root}; expected basis.safetensors, "
            "manifest.json and _SUCCESS"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("policy_mode") != expected_policy_mode:
        raise ValueError(f"Unexpected basis policy mode: {manifest.get('policy_mode')!r}")
    actual_sha = sha256_file(basis_path)
    if actual_sha != manifest.get("basis_sha256"):
        raise ValueError(
            f"AW-LoRA-XS basis SHA mismatch: expected {manifest.get('basis_sha256')}, "
            f"got {actual_sha}"
        )
    return dict(load_file(str(basis_path), device="cpu")), manifest


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_core_checkpoint_atomic(
    model: nn.Module,
    target_dir: str | os.PathLike[str],
    *,
    metadata: Mapping[str, object],
    policy_mode: str = POLICY_MODE,
) -> dict[str, object]:
    """Persist only trainable R cores and a lightweight manifest."""

    return save_core_state_atomic(
        collect_agent_loraxs_core_state(model),
        target_dir,
        metadata=metadata,
        policy_mode=policy_mode,
    )


def save_core_state_atomic(
    state: Mapping[str, torch.Tensor],
    target_dir: str | os.PathLike[str],
    *,
    metadata: Mapping[str, object],
    policy_mode: str = POLICY_MODE,
) -> dict[str, object]:
    """Atomically persist an already gathered R-only state dictionary."""

    from safetensors.torch import save_file

    target = Path(target_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite AW-LoRA-XS checkpoint: {target}")
    temp = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        core_path = temp / "cores.safetensors"
        save_file(dict(state), str(core_path))
        manifest = {
            **dict(metadata),
            "policy_mode": policy_mode,
            "tensor_count": len(state),
            "trainable_value_count": sum(tensor.numel() for tensor in state.values()),
            "cores_file_size_bytes": core_path.stat().st_size,
            "cores_sha256": sha256_file(core_path),
            "storage_contract": "R_only_no_base_no_optimizer_no_scheduler_no_dataloader",
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temp / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        os.replace(temp, target)
        return manifest
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
