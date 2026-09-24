"""Fail closed when persisted trajectory diagnostics prove policy collapse."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_PRIMARY_CODE_ROLES = frozenset({"generator", "worker", "synthesizer"})


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class IntegrityObservation:
    metrics: dict[str, float]
    triggered: bool
    reason: str | None = None


class CodeTrajectoryIntegrityGate:
    """Detect a sustained, objectively invalid Code or Math trajectory regime.

    The gate observes already-computed diagnostics. It never changes rewards,
    samples, optimizer state, or model parameters. A triggered gate only stops
    an invalid run before it can waste more compute or publish checkpoints.
    """

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        config = config or {}
        self.enabled = bool(config.get("enable", False))
        self.task_kind = str(config.get("task_kind", "code"))
        if self.task_kind not in {"code", "math"}:
            raise ValueError("trajectory integrity task_kind must be code or math")
        self.min_step = int(config.get("min_step", 6))
        self.consecutive_steps = int(config.get("consecutive_steps", 3))
        self.max_success = float(config.get("max_success", 0.0))
        self.max_python_parseable = float(
            config.get("max_python_parseable", 0.01)
        )
        self.max_code_fence_retention = float(
            config.get("max_code_fence_retention", 0.01)
        )
        self.min_truncation_rate = float(
            config.get("min_truncation_rate", 0.95)
        )
        self.max_response_length_mean = float(
            config.get("max_response_length_mean", 16.0)
        )
        self.max_role_grad_norm = float(config.get("max_role_grad_norm", 1e-8))
        self.min_evaluator_unknown_rate = float(
            config.get("min_evaluator_unknown_rate", 0.99)
        )
        if self.min_step < 1:
            raise ValueError("trajectory integrity min_step must be positive")
        if self.consecutive_steps < 1:
            raise ValueError(
                "trajectory integrity consecutive_steps must be positive"
            )
        self._consecutive_bad_steps = 0

    def observe(
        self,
        *,
        global_step: int,
        metrics: Mapping[str, object],
    ) -> IntegrityObservation:
        if not self.enabled:
            return IntegrityObservation(metrics={}, triggered=False)

        if self.task_kind == "math":
            return self._observe_math(global_step=global_step, metrics=metrics)

        return self._observe_code(global_step=global_step, metrics=metrics)

    def _observe_code(
        self,
        *,
        global_step: int,
        metrics: Mapping[str, object],
    ) -> IntegrityObservation:
        success = _as_float(metrics.get("batch/success"))
        role_health: dict[str, tuple[float, float, float]] = {}
        prefix = "batch/paper_diag/"
        suffix = "/python_parseable"
        for key, raw_parseable in metrics.items():
            if not key.startswith(prefix) or not key.endswith(suffix):
                continue
            role = key[len(prefix) : -len(suffix)]
            if "/" in role or role not in _PRIMARY_CODE_ROLES:
                continue
            parseable = _as_float(raw_parseable)
            code_fence = _as_float(
                metrics.get(f"{prefix}{role}/code_fence_retention")
            )
            truncation = _as_float(
                metrics.get(f"{prefix}{role}/truncation_rate")
            )
            if None not in (parseable, code_fence, truncation):
                role_health[role] = (
                    float(parseable),
                    float(code_fence),
                    float(truncation),
                )

        unhealthy_roles = {
            role
            for role, (parseable, code_fence, truncation) in role_health.items()
            if parseable <= self.max_python_parseable
            and code_fence <= self.max_code_fence_retention
            and truncation >= self.min_truncation_rate
        }
        all_code_roles_unhealthy = bool(role_health) and (
            unhealthy_roles == set(role_health)
        )
        bad_step = (
            global_step >= self.min_step
            and success is not None
            and success <= self.max_success
            and all_code_roles_unhealthy
        )
        if bad_step:
            self._consecutive_bad_steps += 1
        else:
            self._consecutive_bad_steps = 0

        triggered = self._consecutive_bad_steps >= self.consecutive_steps
        diagnostic_metrics = {
            "integrity/code_collapse/bad_step": float(bad_step),
            "integrity/code_collapse/consecutive_bad_steps": float(
                self._consecutive_bad_steps
            ),
            "integrity/code_collapse/observed_primary_roles": float(
                len(role_health)
            ),
            "integrity/code_collapse/unhealthy_primary_roles": float(
                len(unhealthy_roles)
            ),
            "integrity/code_collapse/triggered": float(triggered),
        }
        if not triggered:
            return IntegrityObservation(
                metrics=diagnostic_metrics,
                triggered=False,
            )

        reason = (
            "Code trajectory integrity gate triggered: "
            f"step={global_step} success={success} "
            f"roles={sorted(role_health)} unhealthy={sorted(unhealthy_roles)} "
            f"consecutive_bad_steps={self._consecutive_bad_steps}. "
            "The run is invalid and must restart from step 0 in a new output root."
        )
        return IntegrityObservation(
            metrics=diagnostic_metrics,
            triggered=True,
            reason=reason,
        )

    def _observe_math(
        self,
        *,
        global_step: int,
        metrics: Mapping[str, object],
    ) -> IntegrityObservation:
        success = _as_float(metrics.get("batch/success"))
        response_length = _as_float(metrics.get("response_length/mean"))
        evaluator_unknown = _as_float(
            metrics.get("batch/paper_diag/evaluator/verdict_unknown_rate")
        )
        role_grad_norms = {
            key.removeprefix("actor/").removesuffix("/grad_norm"): float(value)
            for key, raw_value in metrics.items()
            if key.startswith("actor/")
            and key.endswith("/grad_norm")
            and key.count("/") == 2
            and (value := _as_float(raw_value)) is not None
        }
        all_role_gradients_zero = bool(role_grad_norms) and all(
            value <= self.max_role_grad_norm for value in role_grad_norms.values()
        )
        evaluator_is_degenerate = (
            evaluator_unknown is None
            or evaluator_unknown >= self.min_evaluator_unknown_rate
        )
        bad_step = (
            global_step >= self.min_step
            and success is not None
            and success <= self.max_success
            and response_length is not None
            and response_length <= self.max_response_length_mean
            and all_role_gradients_zero
            and evaluator_is_degenerate
        )
        if bad_step:
            self._consecutive_bad_steps += 1
        else:
            self._consecutive_bad_steps = 0

        triggered = self._consecutive_bad_steps >= self.consecutive_steps
        diagnostic_metrics = {
            "integrity/math_collapse/bad_step": float(bad_step),
            "integrity/math_collapse/consecutive_bad_steps": float(
                self._consecutive_bad_steps
            ),
            "integrity/math_collapse/observed_role_grad_norms": float(
                len(role_grad_norms)
            ),
            "integrity/math_collapse/all_role_gradients_zero": float(
                all_role_gradients_zero
            ),
            "integrity/math_collapse/triggered": float(triggered),
        }
        if not triggered:
            return IntegrityObservation(
                metrics=diagnostic_metrics,
                triggered=False,
            )

        reason = (
            "Math trajectory integrity gate triggered: "
            f"step={global_step} success={success} "
            f"response_length_mean={response_length} "
            f"role_grad_norms={role_grad_norms} "
            f"evaluator_unknown_rate={evaluator_unknown} "
            f"consecutive_bad_steps={self._consecutive_bad_steps}. "
            "The run is invalid and must restart from step 0 in a new output root."
        )
        return IntegrityObservation(
            metrics=diagnostic_metrics,
            triggered=True,
            reason=reason,
        )
