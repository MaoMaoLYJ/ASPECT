# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

from verl.trainer.ppo.gradient_diagnostics import (
    align_lorasb_route_gradient_shards,
    summarize_aligned_route_gradient_gram,
    summarize_gradient_gram,
    summarize_update_accumulation,
)

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # I6 (gradient-direction coherence regularizer): per-role EMA of accumulated grad shards,
        # keyed as {role: {param_name: ema_tensor}}. Lazy init in _apply_coherence_regularizer.
        # Persisted per-rank via save/load_coherence_ema_state (called from the FSDP worker).
        self._coherence_ema: dict[str, dict[str, torch.Tensor]] = {}

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    raise NotImplementedError("This minimal runtime supports text-only models")

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    def _apply_coherence_regularizer(
        self,
        role: str,
        weight: float,
        ema_window: int,
        eps: float,
        metrics: dict,
    ):
        """I6 — Gradient-direction coherence regularizer (step-scaling form).

        After mini-batch backward accumulation completes, scale each parameter's grad
        shard by `(1 - weight * max(0, cos(g, ḡ)))`, where ḡ is an EMA of past grads
        for `role`. Then update the EMA in place using the post-surgery grad. Only the
        currently-active LoRA adapter has `requires_grad=True`, so the parameter loop
        is implicitly role-scoped.

        Args:
            role: agent name; used as the EMA dict key.
            weight: λ in `1 - λ·max(0, cos)`. 0 short-circuits.
            ema_window: τ = 1 - 1/window (e.g., 128 → τ ≈ 0.992).
            eps: numerical floor on the cosine denominator.
            metrics: dict to populate with `actor/coherence_{cos,scale,ema_norm}`.
        """
        if weight <= 0.0:
            return

        ema_role = self._coherence_ema.setdefault(role, {})

        # Pass 1 — collect (param_name, g_local, ema_local) and accumulate dot products.
        # Use float32 accumulators to avoid bf16 underflow.
        device = None
        gg = torch.zeros((), dtype=torch.float32)
        ee = torch.zeros((), dtype=torch.float32)
        ge = torch.zeros((), dtype=torch.float32)
        targets = []
        for name, p in self.actor_module.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            grad = p.grad
            if isinstance(grad, DTensor):
                g_local = grad.to_local()
            else:
                g_local = grad
            if g_local.numel() == 0:
                continue
            if device is None:
                device = g_local.device
                gg = gg.to(device)
                ee = ee.to(device)
                ge = ge.to(device)
            ema_local = ema_role.get(name)
            if ema_local is None:
                ema_local = g_local.detach().clone()
                ema_role[name] = ema_local
            g_f32 = g_local.to(torch.float32)
            e_f32 = ema_local.to(torch.float32)
            gg += g_f32.pow(2).sum()
            ee += e_f32.pow(2).sum()
            ge += (g_f32 * e_f32).sum()
            targets.append((g_local, ema_local))

        if not targets:
            return

        # All-reduce SUM on the 3 scalars across world (trivial cost — 12 bytes per scalar).
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            stacked = torch.stack([gg, ee, ge])
            torch.distributed.all_reduce(stacked, op=torch.distributed.ReduceOp.SUM)
            gg, ee, ge = stacked[0], stacked[1], stacked[2]

        cos = ge / (gg.sqrt() * ee.sqrt() + eps)
        cos_clamped = cos.clamp(0.0, 1.0)
        scale = 1.0 - float(weight) * cos_clamped  # scalar tensor on device

        scale_val = scale.to(torch.float32)
        tau = 1.0 - 1.0 / float(max(ema_window, 1))

        # Pass 2 — apply surgery in-place, then update EMA toward post-surgery grad.
        for g_local, ema_local in targets:
            g_local.mul_(scale_val.to(g_local.dtype))
            ema_local.mul_(tau).add_(g_local.to(ema_local.dtype), alpha=1.0 - tau)

        metrics["actor/coherence_cos"] = float(cos.detach().item())
        metrics["actor/coherence_scale"] = float(scale.detach().item())
        metrics["actor/coherence_ema_norm"] = float(ee.detach().sqrt().item())

    def save_coherence_ema_state(self, path: str) -> None:
        """Serialize the per-rank I6 EMA shards to `path` (one file per rank).

        Layout: torch.save({"world_size": int, "rank": int, "ema": {role: {name: cpu_tensor}}}).
        EMA shards live alongside FSDP-sharded gradients, so reloading is only valid
        when the world size matches; otherwise load_coherence_ema_state cold-starts.
        Skips the write entirely when the EMA dict is empty (e.g. roles with weight=0).
        """
        if not self._coherence_ema:
            return
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        cpu_ema = {
            role: {name: tensor.detach().to("cpu") for name, tensor in role_ema.items()}
            for role, role_ema in self._coherence_ema.items()
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"world_size": world_size, "rank": rank, "ema": cpu_ema}, path)

    def load_coherence_ema_state(self, path: str) -> bool:
        """Reload per-rank EMA shards from `path` if shape/world conventions match.

        Returns True on a successful load. Returns False (and leaves `_coherence_ema`
        as the cold-start empty dict) when the file is absent or the saved world size
        differs from the current world size — at that point the regularizer simply
        re-initializes lazily on the next step, the same as a fresh run.
        """
        if not os.path.exists(path):
            return False
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning(f"[I6] failed to load coherence EMA from {path}: {exc}")
            return False
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        if int(blob.get("world_size", -1)) != world_size:
            logger.warning(
                f"[I6] coherence EMA world_size mismatch (file={blob.get('world_size')}, "
                f"current={world_size}); cold-starting EMA"
            )
            return False
        ema_blob = blob.get("ema", {}) or {}
        device = next(
            (p.device for p in self.actor_module.parameters() if p.requires_grad),
            torch.device("cpu"),
        )
        self._coherence_ema = {
            role: {name: tensor.to(device) for name, tensor in role_ema.items()}
            for role, role_ema in ema_blob.items()
        }
        return True

    def _gradient_diag_probe_loss(self, sample: DataProto, temperature: float, on_policy: bool) -> torch.Tensor:
        """Compute the configured PPO loss for one diagnostic sequence."""

        model_inputs = {**sample.batch, **sample.non_tensor_batch}
        response_mask = model_inputs["response_mask"]
        calculate_entropy = self.config.entropy_coeff != 0
        entropy, log_prob = self._forward_micro_batch(
            model_inputs,
            temperature=temperature,
            calculate_entropy=calculate_entropy,
        )
        old_log_prob = log_prob.detach() if on_policy else model_inputs["old_log_probs"]
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, _ = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=model_inputs["advantages"],
            response_mask=response_mask,
            loss_agg_mode=self.config.loss_agg_mode,
            config=self.config,
            rollout_is_weights=model_inputs.get("rollout_is_weights", None),
        )
        policy_loss = pg_loss
        if calculate_entropy:
            entropy_loss = agg_loss(
                loss_mat=entropy,
                loss_mask=response_mask,
                loss_agg_mode=self.config.loss_agg_mode,
            )
            policy_loss = policy_loss - entropy_loss * self.config.entropy_coeff
        if self.config.use_kl_loss:
            kld = kl_penalty(
                logprob=log_prob,
                ref_logprob=model_inputs["ref_log_prob"],
                kl_penalty=self.config.kl_loss_type,
            )
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=response_mask,
                loss_agg_mode=self.config.loss_agg_mode,
            )
            kl_coef = float(sample.meta_info.get("kl_loss_coef_override", self.config.kl_loss_coef))
            policy_loss = policy_loss + kl_loss * kl_coef
        return policy_loss

    def _clone_trainable_grad_shards(self) -> dict[str, torch.Tensor]:
        """Clone active trainable gradient shards without gathering parameters.

        Figure 4 freezes the base model, so every trainable parameter belongs
        to the active LoRA adapter. FSDP with ``use_orig_params=False`` may
        expose flattened names that no longer contain ``lora_``; filtering by
        name would silently discard the gradients being measured.
        """

        shards: dict[str, torch.Tensor] = {}
        for name, parameter in self.actor_module.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            grad = parameter.grad
            local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
            if local_grad.numel() == 0:
                continue
            shards[name] = local_grad.detach().clone()
        return shards

    def _collect_gradient_diagnostics(
        self,
        data: DataProto,
        *,
        temperature: float,
        on_policy: bool,
    ) -> dict[str, float]:
        """Probe role/slot LoRA gradients and restore training state afterwards.

        Each rank uses the same fixed number of one-sequence forward/backward
        calls per group. Loss scaling converts FSDP's rank mean into a global
        sequence mean, including when a rank has no local sample for a group.
        Only scalar Gram matrices are all-reduced; parameter gradients remain
        sharded and are discarded before the real PPO backward pass.
        """

        if not data.meta_info.get("gradient_diag_enable", False) or len(data) == 0:
            return {}

        mode = str(data.meta_info["gradient_diag_mode"])
        label_key = (
            "gradient_diag_role_labels" if mode == "sp" else "gradient_diag_slot_labels"
        )
        groups = [str(group) for group in data.meta_info.get("gradient_diag_groups", [])]
        if not groups:
            return {}
        labels = [str(label) for label in data.non_tensor_batch[label_key]]
        samples_per_group = max(
            1,
            int(data.meta_info.get("gradient_diag_samples_per_group_per_rank", 1)),
        )
        eps = float(data.meta_info.get("gradient_diag_eps", 1e-12))
        distributed = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        device = get_device_id()

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        group_grads: list[dict[str, torch.Tensor]] = []
        global_probe_counts: list[float] = []

        try:
            self.actor_optimizer.zero_grad()
            fallback_idx = 0
            for group in groups:
                all_local_indices = [
                    index for index, label in enumerate(labels) if label == group
                ]
                if len(all_local_indices) <= samples_per_group:
                    local_indices = all_local_indices
                else:
                    local_indices = [
                        all_local_indices[
                            min(
                                len(all_local_indices) - 1,
                                int(
                                    (probe_index + 0.5)
                                    * len(all_local_indices)
                                    / samples_per_group
                                ),
                            )
                        ]
                        for probe_index in range(samples_per_group)
                    ]
                local_count = len(local_indices)
                probe_count = torch.tensor(
                    [float(local_count)],
                    dtype=torch.float32,
                    device=device,
                )
                if distributed:
                    torch.distributed.all_reduce(
                        probe_count,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                global_count = float(probe_count[0].item())
                global_probe_counts.append(global_count)

                for probe_index in range(samples_per_group):
                    is_real = probe_index < local_count and global_count > 0
                    sample_index = local_indices[probe_index] if is_real else fallback_idx
                    sample = data.select_idxs([sample_index]).to(device)
                    probe_loss = self._gradient_diag_probe_loss(
                        sample,
                        temperature=temperature,
                        on_policy=on_policy,
                    )
                    # FSDP averages gradients over ranks. This factor makes the
                    # accumulated probe equal to a global sequence mean.
                    scale = float(world_size) / global_count if is_real else 0.0
                    (probe_loss * scale).backward()

                group_grads.append(self._clone_trainable_grad_shards())
                self.actor_optimizer.zero_grad()

            first_shard = next(
                (
                    shard
                    for gradients in group_grads
                    for shard in gradients.values()
                ),
                None,
            )
            if first_shard is None:
                raise RuntimeError(
                    "Gradient diagnostics found no trainable gradient shards. "
                    "Figure 4 requires a frozen base model with trainable LoRA adapters."
                )

            gram = torch.zeros(
                (len(groups), len(groups)),
                dtype=torch.float32,
                device=first_shard.device,
            )
            for i in range(len(groups)):
                for j in range(i, len(groups)):
                    inner = torch.zeros((), dtype=torch.float32, device=first_shard.device)
                    common_names = group_grads[i].keys() & group_grads[j].keys()
                    for name in common_names:
                        left = group_grads[i][name].to(torch.float32)
                        right = group_grads[j][name].to(torch.float32)
                        inner += (left * right).sum()
                    gram[i, j] = inner
                    gram[j, i] = inner
            if distributed:
                torch.distributed.all_reduce(gram, op=torch.distributed.ReduceOp.SUM)

            metrics = summarize_gradient_gram(
                gram=gram,
                groups=groups,
                sequence_counts=data.meta_info["gradient_diag_sequence_counts"],
                trajectory_counts=data.meta_info[
                    "gradient_diag_trajectory_counts"
                ],
                response_token_counts=data.meta_info[
                    "gradient_diag_response_token_counts"
                ],
                probe_counts=global_probe_counts,
                mode=mode,
                scope=str(data.meta_info.get("gradient_diag_scope", "")) or None,
                eps=eps,
            )
            if data.meta_info.get("lorasb_route_gradient_enable", False):
                counts = torch.as_tensor(
                    data.meta_info["gradient_diag_sequence_counts"],
                    dtype=torch.float64,
                )
                count_sum = float(counts.sum().item())
                shares = (
                    counts / count_sum if count_sum > 0 else torch.zeros_like(counts)
                )
                route_gradient: dict[str, torch.Tensor] = {}
                for group_index, gradients in enumerate(group_grads):
                    weight = float(shares[group_index].item())
                    for name, shard in gradients.items():
                        contribution = shard.to(torch.float32) * weight
                        if name in route_gradient:
                            route_gradient[name].add_(contribution)
                        else:
                            route_gradient[name] = contribution.clone()
                metrics.update(
                    self._collect_lorasb_route_gradient_diagnostics(
                        route_gradient,
                        route=str(data.meta_info["lorasb_route"]),
                        expected_routes=[
                            str(route)
                            for route in data.meta_info["lorasb_expected_routes"]
                        ],
                        global_step=int(data.meta_info["lorasb_global_step"]),
                        sequence_count=float(
                            data.meta_info["lorasb_route_sequence_count"]
                        ),
                        eps=eps,
                    )
                )
            return metrics
        finally:
            self.actor_optimizer.zero_grad()
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

    def _collect_lorasb_route_gradient_diagnostics(
        self,
        gradient_shards: dict[str, torch.Tensor],
        *,
        route: str,
        expected_routes: list[str],
        global_step: int,
        sequence_count: float,
        eps: float,
    ) -> dict[str, float]:
        """Accumulate one aligned probe per route and emit one exact Gram summary."""

        if not expected_routes or len(set(expected_routes)) != len(expected_routes):
            raise ValueError("LoRA-SB expected routes must be non-empty and unique")
        if route not in expected_routes:
            raise ValueError(
                f"LoRA-SB route {route!r} is absent from expected routes {expected_routes!r}"
            )
        if sequence_count < 0:
            raise ValueError("LoRA-SB route sequence count must be non-negative")
        aligned = align_lorasb_route_gradient_shards(gradient_shards, route=route)
        state = getattr(self, "_lorasb_route_gradient_state", None)
        if (
            not isinstance(state, dict)
            or state.get("global_step") != int(global_step)
            or state.get("expected_routes") != tuple(expected_routes)
        ):
            state = {
                "global_step": int(global_step),
                "expected_routes": tuple(expected_routes),
                "gradients": {},
                "sequence_counts": {},
            }
            self._lorasb_route_gradient_state = state
        state["gradients"][route] = aligned
        state["sequence_counts"][route] = float(sequence_count)
        if any(expected not in state["gradients"] for expected in expected_routes):
            return {}

        reference_keys = set(state["gradients"][expected_routes[0]])
        for expected in expected_routes[1:]:
            if set(state["gradients"][expected]) != reference_keys:
                raise RuntimeError(
                    "LoRA-SB aligned route gradients have inconsistent module coordinates"
                )
        first_shard = next(iter(state["gradients"][expected_routes[0]].values()))
        gram = torch.zeros(
            (len(expected_routes), len(expected_routes)),
            dtype=torch.float32,
            device=first_shard.device,
        )
        for left_index, left_route in enumerate(expected_routes):
            left = state["gradients"][left_route]
            for right_index in range(left_index, len(expected_routes)):
                right_route = expected_routes[right_index]
                right = state["gradients"][right_route]
                inner = torch.zeros((), dtype=torch.float32, device=first_shard.device)
                for key in reference_keys:
                    inner += (left[key].to(torch.float32) * right[key].to(torch.float32)).sum()
                gram[left_index, right_index] = inner
                gram[right_index, left_index] = inner
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(gram, op=torch.distributed.ReduceOp.SUM)
        metrics = summarize_aligned_route_gradient_gram(
            gram=gram,
            routes=expected_routes,
            sequence_counts=[
                state["sequence_counts"][expected] for expected in expected_routes
            ],
            eps=eps,
        )
        self._lorasb_route_gradient_state = None
        return metrics

    def _accumulate_optimizer_gradient(
        self,
        gradient_sum: dict[str, torch.Tensor],
    ) -> float | None:
        """Add the current real pre-clip gradient to an iteration accumulator."""

        current = self._clone_trainable_grad_shards()
        if not current:
            return None
        first = next(iter(current.values()))
        norm_sq = torch.zeros((), dtype=torch.float32, device=first.device)
        for shard in current.values():
            shard_f32 = shard.to(torch.float32)
            norm_sq += shard_f32.square().sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(norm_sq, op=torch.distributed.ReduceOp.SUM)
        if not torch.isfinite(norm_sq):
            return None
        for name, shard in current.items():
            shard_f32 = shard.detach().to(torch.float32)
            if name not in gradient_sum:
                gradient_sum[name] = shard_f32.clone()
            else:
                gradient_sum[name].add_(shard_f32)
        return float(norm_sq.clamp(min=0).sqrt().item())

    def _summarize_optimizer_gradient_accumulation(
        self,
        gradient_sum: dict[str, torch.Tensor],
        individual_norm_sum: float,
        update_count: int,
        data: DataProto,
    ) -> dict[str, float]:
        if not gradient_sum or update_count == 0:
            return {}
        first = next(iter(gradient_sum.values()))
        summed_norm_sq = torch.zeros((), dtype=torch.float32, device=first.device)
        for shard in gradient_sum.values():
            summed_norm_sq += shard.to(torch.float32).square().sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                summed_norm_sq,
                op=torch.distributed.ReduceOp.SUM,
            )
        return summarize_update_accumulation(
            summed_gradient_norm=float(summed_norm_sq.clamp(min=0).sqrt().item()),
            individual_gradient_norm_sum=individual_norm_sum,
            update_count=update_count,
            mode=str(data.meta_info["gradient_diag_mode"]),
            scope=str(data.meta_info.get("gradient_diag_scope", "")) or None,
            eps=float(data.meta_info.get("gradient_diag_eps", 1e-12)),
        )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if data.meta_info.get("gradient_diag_enable", False):
            non_tensor_select_keys.extend(
                ["gradient_diag_role_labels", "gradient_diag_slot_labels"]
            )

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        gradient_diag_enabled = bool(data.meta_info.get("gradient_diag_enable", False))
        optimizer_gradient_sum: dict[str, torch.Tensor] = {}
        optimizer_individual_norm_sum = 0.0
        optimizer_update_count = 0
        if gradient_diag_enabled:
            diagnostic_metrics = self._collect_gradient_diagnostics(
                data,
                temperature=temperature,
                on_policy=on_policy,
            )
            append_to_dict(metrics, diagnostic_metrics)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # Per-role KL coefficient override (I1 intervention).
                        # Trainer sets `kl_loss_coef_override` on sub_batch.meta_info per agent for IP runs;
                        # falls through to global `kl_loss_coef` otherwise.
                        kl_coef = float(data.meta_info.get("kl_loss_coef_override", self.config.kl_loss_coef))
                        policy_loss = policy_loss + kl_loss * kl_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = kl_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                # I6 — gradient-direction coherence regularizer. Surgery on the active
                # adapter's accumulated grads, before clip + step. Trainer sets
                # `coherence_reg_*` on data.meta_info per agent under IP runs.
                if data.meta_info.get("coherence_reg_weight", 0.0) > 0:
                    coh_metrics: dict = {}
                    self._apply_coherence_regularizer(
                        role=str(data.meta_info["coherence_reg_role"]),
                        weight=float(data.meta_info["coherence_reg_weight"]),
                        ema_window=int(data.meta_info.get("coherence_reg_ema_window", 128)),
                        eps=float(data.meta_info.get("coherence_reg_eps", 1e-8)),
                        metrics=coh_metrics,
                    )
                    append_to_dict(metrics, coh_metrics)

                if gradient_diag_enabled:
                    update_norm = self._accumulate_optimizer_gradient(
                        optimizer_gradient_sum
                    )
                    if update_norm is not None:
                        optimizer_individual_norm_sum += update_norm
                        optimizer_update_count += 1

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        if gradient_diag_enabled:
            update_metrics = self._summarize_optimizer_gradient_accumulation(
                optimizer_gradient_sum,
                optimizer_individual_norm_sum,
                optimizer_update_count,
                data,
            )
            append_to_dict(metrics, update_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
