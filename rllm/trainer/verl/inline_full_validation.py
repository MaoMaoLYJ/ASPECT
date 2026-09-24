"""Atomic online evaluation records for an independent Full-FT run."""
from __future__ import annotations

import json
import os
from pathlib import Path


def validation_record(*, step, ids, correct, expected_rows, dataset_sha256,
                      workflow, canonical=True):
    if step <= 0:
        raise ValueError("Online evaluation must follow a completed update")
    if canonical and expected_rows != 1412:
        raise ValueError("Official DAPO evaluation requires 1412 problems")
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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
