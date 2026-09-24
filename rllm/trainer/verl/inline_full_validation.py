"""Auditable online evaluation for checkpoint-free full-parameter runs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class TailAuditComplete(Exception):
    """The companion finished while a noncanonical verification was active."""


def all_primary_complete(directory):
    return all((Path(directory) / f"{w}.json").is_file() for w in ("eval_opt", "orch_workers"))


def tail_verification(trainer):
    """Use the finished live policy for a clearly separate repeatability audit."""
    from omegaconf import open_dict
    cfg = trainer.config.trainer.get("inline_full_validation", {})
    control = cfg.get("tail_control_dir")
    if not cfg.get("enable", False) or not control:
        return
    output = Path(cfg.output)
    for step in range(10, 201, 10):
        row = json.loads((output / f"step_{step:03d}.json").read_text())
        if not row["canonical"] or row["step"] != step or row["num_total"] != 1412:
            raise ValueError("Cannot enter tail audit without complete primary validation")
    write_record(Path(control) / f"{cfg.workflow}.json", {"completed_step": 200, "online_validations": 20})
    original_step = trainer.global_steps
    trainer.global_steps = 200
    audit_number = 0
    try:
        while not all_primary_complete(control):
            audit_number += 1
            audit_root = output.parent / "AUDIT_ONLY_NOT_CANONICAL" / f"repeatability_{audit_number:04d}"
            with open_dict(cfg):
                cfg.canonical = False
                cfg.output = str(audit_root)
            write_record(audit_root / "protocol.json", {
                "canonical": False, "purpose": "final live-policy stochastic validation repeatability",
                "training_updates": False, "temperature": 0.7, "n": 1,
                "sample_stream": "continued independent draws, not reset to repeat previous answers",
                "exclude_from_paper_and_training_curves": True,
            })
            try:
                trainer._validate_agent()
            except TailAuditComplete:
                write_record(audit_root / "AUDIT_ABORTED_AFTER_PRIMARY_COMPLETION.json",
                             {"canonical": False, "primary_complete": True})
                break
    finally:
        trainer.global_steps = original_step
        with open_dict(cfg):
            cfg.canonical = True
            cfg.output = str(output)


def mark_activity(config, step, count):
    if config.get("tail_control_dir"):
        write_record(Path(config.tail_control_dir) / f"activity_{config.workflow}.json",
                     {"step": step, "completed_problems": count, "canonical": config.canonical,
                      "time": time.time()})


def validation_record(*, step, ids, correct, expected_rows, dataset_sha256,
                      workflow, canonical=True):
    if step <= 0 or (canonical and step not in range(10, 201, 10)):
        raise ValueError("Online evaluation must follow an updated step 10..200")
    if canonical and expected_rows != 1412:
        raise ValueError("Formal DAPO evaluation requires 1412 problems")
    if (len(ids) != expected_rows or len(correct) != expected_rows
            or set(ids) != {f"eval_{i}" for i in range(expected_rows)}):
        raise ValueError("Missing or duplicate online validation problems")
    if len(dataset_sha256) != 64 or any(c not in "0123456789abcdef" for c in dataset_sha256):
        raise ValueError("Dataset SHA256 is required")
    if any(x not in (True, False, 0, 1) for x in correct):
        raise ValueError("Invalid correctness outcome")
    outcomes = dict(zip(ids, map(bool, correct), strict=True))
    values = [int(outcomes[f"eval_{i}"]) for i in range(expected_rows)]
    return {"step": step, "workflow": workflow, "dataset": "dapo_math",
            "split": "test", "dataset_sha256": dataset_sha256,
            "num_total": expected_rows, "num_correct": sum(values),
            "accuracy": sum(values) / expected_rows, "per_problem_n_correct": values,
            "n_rollouts": 1, "canonical": canonical, "checkpoint_saved": False,
            "evaluation_source": "live_updated_full_parameter_policy"}


def write_record(path, record):
    from scripts.repro.shared_volume_io import retry_estale

    @retry_estale
    def publish():
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    publish()
