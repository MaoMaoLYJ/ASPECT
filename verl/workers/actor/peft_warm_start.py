"""Audited shared-policy adapter weight warm start, before FSDP wrapping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sp_checkpoint(path, *, step, rank, alpha):
    root = Path(path)
    manifest_path = root / "actor/peft_checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("global_step") != step
            or manifest.get("policy") != "sp"
            or manifest.get("checkpoint_format") != "peft_eval_v1"
            or manifest.get("adapter_routes") != [
                {"route": "default", "directory": "lora_adapter"}]):
        raise ValueError("SP warm-start checkpoint manifest mismatch")
    adapter = root / "actor/lora_adapter"
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    if config.get("r") != rank or config.get("lora_alpha") != alpha:
        raise ValueError("SP warm-start rank/alpha mismatch")
    if config.get("use_dora", False) or config.get("use_rslora", False):
        raise ValueError("SP warm start only supports the ordinary LoRA contract")
    weights = adapter / "adapter_model.safetensors"
    if weights.stat().st_size == 0:
        raise ValueError("Empty SP adapter")
    return {
        "global_step": step, "checkpoint": str(root),
        "adapter_path": str(adapter), "rank": rank, "alpha": alpha,
        "weights_sha256": sha256(weights),
        "config_sha256": sha256(config_path),
        "manifest_sha256": sha256(manifest_path),
        "continuity_claim": "weight_warm_start_not_exact_resume",
        "optimizer_state_restored": False, "dataloader_state_restored": False,
        "lora_A_B_trainable": True,
    }


def load_sp_checkpoint_(model, path, *, step, rank, alpha):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    source_rank = not distributed or torch.distributed.get_rank() == 0
    result = [None]
    if source_rank:
        try:
            report = audit_sp_checkpoint(path, step=step, rank=rank, alpha=alpha)
            state = load_file(str(Path(report["adapter_path"]) / "adapter_model.safetensors"))
            current = get_peft_model_state_dict(model, adapter_name="default")
            if set(state) != set(current) or not state:
                raise ValueError("SP warm-start tensor keys do not exactly match model")
            for key, value in state.items():
                if not ("lora_A" in key or "lora_B" in key):
                    raise ValueError(f"Unexpected SP tensor {key}")
                if value.shape != current[key].shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid SP tensor {key}")
            load_unsharded_lora_weights_(model, state)
            loaded = get_peft_model_state_dict(model, adapter_name="default")
            if any(not torch.equal(loaded[k].detach().cpu(), v.to(loaded[k].dtype))
                   for k, v in state.items()):
                raise ValueError("SP adapter failed exact post-load tensor audit")
            if any(not p.requires_grad for n, p in model.named_parameters()
                   if ".lora_A.default." in n or ".lora_B.default." in n):
                raise ValueError("SP adapter A/B unexpectedly frozen")
            report["tensors_loaded"] = len(state)
            report["loader"] = "audited_unsharded_lora_before_fsdp"
            report["passed"] = True
            result[0] = {"report": report}
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    # Nonzero ranks hold meta tensors; FSDP sync_module_states loads their weights.
    if distributed:
        torch.distributed.broadcast_object_list(result, src=0)
    if "error" in result[0]:
        raise ValueError(result[0]["error"])
    return result[0]["report"]


def load_unsharded_lora_weights_(model, state):
    """Load ordinary A/B tensors before FSDP, without PEFT's unrelated HF-TP path."""
    import torch
    config = model.peft_config["default"]
    if (config.bias != "none" or config.modules_to_save
            or config.use_dora or config.use_rslora):
        raise ValueError("Unsharded SP loader requires ordinary bias-free LoRA")
    if any(getattr(module, "_hf_device_mesh", None) is not None
           or getattr(module, "_tp_info", None) is not None for module in model.modules()):
        raise ValueError("SP warm start must precede HF tensor parallelism and FSDP")
    parameters = {name: value for name, value in model.named_parameters()
                  if name.endswith((".lora_A.default.weight", ".lora_B.default.weight"))}
    mapped = {}
    for name, value in state.items():
        if not name.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"Unexpected SP tensor {name}")
        mapped[name.removesuffix(".weight") + ".default.weight"] = value
    if not mapped or set(mapped) != set(parameters):
        raise ValueError("SP warm-start tensor keys do not exactly match parameters")
    # Validate the entire source before changing any parameter. Only rank0 owns
    # materialized full tensors here; FSDP sync_module_states handles other ranks.
    for name, value in mapped.items():
        target = parameters[name]
        if (target.is_meta or hasattr(target, "to_local") or value.ndim != 2
                or target.shape != value.shape or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid unsharded SP tensor {name}")
        if not target.requires_grad:
            raise ValueError(f"SP A/B must remain trainable: {name}")
    with torch.no_grad():
        for name, value in mapped.items():
            target = parameters[name]
            target.copy_(value.to(device=target.device, dtype=target.dtype))
