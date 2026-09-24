#!/usr/bin/env python3
"""Prepare only the DAPO Math split needed by full checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rllm.data.dataset import DatasetRegistry

from aspect.data.contracts import OFFICIAL_DATASET_CONTRACT
from aspect.data.sources import _load_mount, _prepare_dapo, _sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    raw_train, train_source = _load_mount(
        args.train_root,
        "train",
        config_hint=OFFICIAL_DATASET_CONTRACT["dapo"]["config"],
    )
    if train_source is None:
        raise ValueError(
            "Formal DAPO full validation input must expose the exact source file "
            "so its SHA256 value can be verified; load_from_disk mounts are not accepted."
        )

    train_source_sha256 = _sha256_file(train_source)
    expected_train_sha256 = OFFICIAL_DATASET_CONTRACT["dapo"]["source_sha256"]
    if train_source_sha256 != expected_train_sha256:
        raise ValueError(
            "DAPO source SHA256 mismatch: "
            f"expected {expected_train_sha256}, got {train_source_sha256} "
            f"for {train_source}"
        )

    expected_raw_rows = OFFICIAL_DATASET_CONTRACT["dapo"]["raw_rows"]
    if len(raw_train) != expected_raw_rows:
        raise ValueError(
            f"DAPO en row-count mismatch: expected {expected_raw_rows}, got {len(raw_train)}"
        )

    train, held_out = _prepare_dapo(raw_train)
    expected_train_rows = OFFICIAL_DATASET_CONTRACT["dapo"]["train_rows"]
    expected_test_rows = OFFICIAL_DATASET_CONTRACT["dapo"]["test_rows"]
    if len(train) != expected_train_rows or len(held_out) != expected_test_rows:
        raise ValueError(
            "DAPO deterministic split mismatch: expected "
            f"{expected_train_rows}/{expected_test_rows}, got {len(train)}/{len(held_out)}"
        )

    registered_train = DatasetRegistry.register_dataset("dapo_math", train, "train")
    registered_test = DatasetRegistry.register_dataset("dapo_math", held_out, "test")

    summary = {
        "train_mount": str(args.train_root.resolve()),
        "dapo_repository": OFFICIAL_DATASET_CONTRACT["dapo"]["repository"],
        "dapo_revision": OFFICIAL_DATASET_CONTRACT["dapo"]["revision"],
        "dapo_config": OFFICIAL_DATASET_CONTRACT["dapo"]["config"],
        "dapo_source_file": str(train_source.resolve()),
        "dapo_source_sha256": train_source_sha256,
        "raw_train_rows": len(raw_train),
        "dapo_train_rows": len(registered_train),
        "dapo_held_out_rows": len(registered_test),
        "dapo_train_path": registered_train.get_data_path(),
        "dapo_train_verl_path": registered_train.get_verl_data_path(),
        "dapo_test_path": registered_test.get_data_path(),
        "dapo_test_verl_path": registered_test.get_verl_data_path(),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
