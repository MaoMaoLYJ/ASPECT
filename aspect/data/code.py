#!/usr/bin/env python3
"""Reconstruct and register the exact DeepCoder Code split used by the paper.

The upstream repository omits ``coding_dataset_filtering/runtime_results.json``.
The pinned public bridge dataset contains the post-runtime-filter test lists.
This module verifies every bridge row against the pinned official source row,
reconstructs the filtered source in original order, applies the repository's
seed-42 1000/rest split, and emits an auditable reproducibility package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from rllm.data.dataset import DatasetRegistry
from rllm.data.utils import fetch_live_code_bench_system_prompt

RAW_REPOSITORY = "agentica-org/DeepCoder-Preview-Dataset"
RAW_REVISION = "780aeec7b7716e34f0b9e02bb624fc6ad8384725"
RAW_CONFIG = "primeintellect"
RAW_ROWS = 16_252
RAW_SHARD_SHA256 = (
    "a1efc9d5e9eb6a9977576b08e5349758c8b139131ea5d13ef5c390f2a21db538",
    "2a645ca9e77976f49699ad9c25d79a697203e8811b0887ae256e7301d3e72681",
    "2bd0cc7207fa0aad722cc21a275afd777ab1debb318c4b9a9eccb18f7fee32f0",
    "4503d28636d63dab88e4a07a54b4a8257c9ad2e1663e8cdcb1216e2306550427",
    "b1148bc2f0939843d500e43668e643c619a67f17797358a1ddf8e33c9782a545",
)

BRIDGE_REPOSITORY = "mnoukhov/deepcoder_primeintellect"
BRIDGE_REVISION = "90f49967b15f25ce5a8513fb3606d13dc41e6a94"
BRIDGE_SHA256 = "34651aa15b21e8a5ff6b9b14db4b8d474892e67189c77ecc9b41693a51c90212"
BRIDGE_ROWS = 14_995
BRIDGE_PROMPT_SUFFIX = (
    "\n\nWrite Python code to solve this problem. Your program should read the input "
    "from stdin and write the output to stdout. Enclose your complete solution "
    "in a single ```python code block."
)

SPLIT_SEED = 42
TEST_ROWS = 1_000
TRAIN_ROWS = 13_995
CALIBRATION_ROWS = 50
DATASET_NAME = "deepcoder_primeintellect"
PACKAGE_FORMAT = "deepcoder_primeintellect_runtime_filtered_v1"
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "448a457ea53a188ca20a15f4635256f13dbb0f1a5a3b4a3debd84b081dbd8da7"
)
EXPECTED_PACKAGE_SUCCESS_SHA256 = (
    "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"
)
EXPECTED_PACKAGE_FILE_SHA256 = {
    "train.parquet": "dfd1affcaf6f0a90ddb5677e6b862136c4f1044c3975ab799c5254224cb7f30e",
    "test.parquet": "1c91105cfd202001a6e7e46d20ab48caa824f313cd36f6ff29be49651c812982",
    "calibration.parquet": "ddd00fcd87e190af8662e84005b60d7dfd213a86c13b9f4d6fd6150914f40c8a",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_package_root(mount_root: Path) -> Path:
    """Resolve and verify the one exact DeepCoder package under a local dataset directory."""

    mount_root = mount_root.expanduser().resolve()
    if not mount_root.is_dir():
        raise ValueError(f"DeepCoder mount root is not a directory: {mount_root}")

    required = {
        "manifest.json",
        "_SUCCESS",
        *EXPECTED_PACKAGE_FILE_SHA256,
    }
    candidates = {
        manifest.parent.resolve()
        for manifest in mount_root.rglob("manifest.json")
        if all((manifest.parent / name).is_file() for name in required)
    }
    if len(candidates) != 1:
        rendered = [str(path) for path in sorted(candidates)]
        raise ValueError(
            "Expected exactly one complete pinned DeepCoder package under "
            f"{mount_root}, found {len(candidates)}: {rendered}"
        )

    package_root = candidates.pop()
    manifest_path = package_root / "manifest.json"
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != EXPECTED_PACKAGE_MANIFEST_SHA256:
        raise ValueError(
            "DeepCoder package manifest SHA mismatch: expected "
            f"{EXPECTED_PACKAGE_MANIFEST_SHA256}, got {actual_manifest_sha}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "format": PACKAGE_FORMAT,
        "raw_repository": RAW_REPOSITORY,
        "raw_revision": RAW_REVISION,
        "runtime_filter_bridge_repository": BRIDGE_REPOSITORY,
        "runtime_filter_bridge_revision": BRIDGE_REVISION,
        "split_seed": SPLIT_SEED,
        "train_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "calibration_rows": CALIBRATION_ROWS,
    }
    mismatches = {
        key: (manifest.get(key), expected)
        for key, expected in expected_metadata.items()
        if manifest.get(key) != expected
    }
    if manifest.get("files") != EXPECTED_PACKAGE_FILE_SHA256:
        mismatches["files"] = (
            manifest.get("files"),
            EXPECTED_PACKAGE_FILE_SHA256,
        )
    if mismatches:
        raise ValueError(f"DeepCoder package metadata mismatch: {mismatches}")

    success_sha = sha256_file(package_root / "_SUCCESS")
    if success_sha != EXPECTED_PACKAGE_SUCCESS_SHA256:
        raise ValueError(
            "DeepCoder _SUCCESS SHA mismatch: expected "
            f"{EXPECTED_PACKAGE_SUCCESS_SHA256}, got {success_sha}"
        )
    for name, expected_sha in EXPECTED_PACKAGE_FILE_SHA256.items():
        actual_sha = sha256_file(package_root / name)
        if actual_sha != expected_sha:
            raise ValueError(
                f"DeepCoder {name} SHA mismatch: expected {expected_sha}, got {actual_sha}"
            )
    return package_root


def _json_sha256(rows: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


def _decode_tests(value: Any) -> list[dict[str, Any]]:
    tests = json.loads(value) if isinstance(value, str) else value
    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        tests = [
            {
                "input": input_value,
                "output": output_value,
                "testtype": "stdin_stdout",
            }
            for input_value, output_value in zip(
                tests["inputs"], tests["outputs"], strict=False
            )
        ]
    if not isinstance(tests, list):
        tests = [tests] if tests else []
    if not all(isinstance(test, dict) for test in tests):
        raise ValueError("DeepCoder tests must be dictionaries")
    return tests


def _test_key(test: dict[str, Any]) -> str:
    return json.dumps(test, ensure_ascii=False, sort_keys=True)


def _is_test_multiset_subset(
    source: Sequence[dict], kept: Sequence[dict]
) -> bool:
    source_counts = Counter(_test_key(test) for test in source)
    kept_counts = Counter(_test_key(test) for test in kept)
    return all(source_counts[key] >= count for key, count in kept_counts.items())


def _restore_source_test_order(
    source: Sequence[dict], kept: Sequence[dict]
) -> list[dict]:
    remaining = Counter(_test_key(test) for test in kept)
    restored = []
    for test in source:
        key = _test_key(test)
        if remaining[key] > 0:
            restored.append(test)
            remaining[key] -= 1
    if any(remaining.values()):
        raise ValueError("Filtered tests are not a multiset subset of source tests")
    return restored


def reconstruct_runtime_filtered_rows(
    raw_rows: Sequence[dict[str, Any]],
    bridge_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover post-runtime-filter rows and prove their source correspondence."""

    problem_to_indices: dict[str, list[int]] = {}
    for source_index, row in enumerate(raw_rows):
        problem_to_indices.setdefault(str(row["problem"]), []).append(source_index)

    by_source_index: dict[int, dict[str, Any]] = {}
    bridge_source_indices: list[int] = []
    ambiguous_compatible_rows = 0
    last_source_index = -1
    for bridge_index, bridge in enumerate(bridge_rows):
        messages = bridge.get("messages")
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or messages[0].get("role") != "user"
            or not isinstance(messages[0].get("content"), str)
        ):
            raise ValueError(f"Bridge row {bridge_index} has an unexpected prompt schema")
        content = messages[0]["content"]
        if not content.endswith(BRIDGE_PROMPT_SUFFIX):
            raise ValueError(f"Bridge row {bridge_index} has an unknown prompt suffix")
        problem = content[: -len(BRIDGE_PROMPT_SUFFIX)]
        if problem not in problem_to_indices:
            raise ValueError(f"Bridge row {bridge_index} has no official source match")
        kept_tests = _decode_tests(bridge["ground_truth"])
        if not kept_tests:
            raise ValueError(f"Bridge row {bridge_index} retained no tests")
        compatible_candidates = [
            source_index
            for source_index in problem_to_indices[problem]
            if source_index not in by_source_index
            and _is_test_multiset_subset(
                _decode_tests(raw_rows[source_index]["tests"]), kept_tests
            )
        ]
        candidates = [
            source_index
            for source_index in compatible_candidates
            if source_index > last_source_index
        ]
        if not candidates:
            raise ValueError(
                f"Bridge row {bridge_index} cannot be mapped in official source "
                f"order; compatible_candidates={compatible_candidates}, "
                f"last_source_index={last_source_index}"
            )
        if len(candidates) > 1:
            ambiguous_compatible_rows += 1
        source_index = min(candidates)
        last_source_index = source_index
        source_row = dict(raw_rows[source_index])
        source_row["tests"] = json.dumps(
            _restore_source_test_order(
                _decode_tests(source_row["tests"]), kept_tests
            ),
            ensure_ascii=False,
        )
        source_row["source_index"] = source_index
        by_source_index[source_index] = source_row
        bridge_source_indices.append(source_index)

    filtered = [by_source_index[index] for index in sorted(by_source_index)]
    audit = {
        "raw_rows": len(raw_rows),
        "filtered_rows": len(filtered),
        "excluded_rows": len(raw_rows) - len(filtered),
        "unique_source_rows": len(by_source_index),
        "bridge_rows_in_source_order": bridge_source_indices
        == sorted(bridge_source_indices),
        "rows_with_later_compatible_duplicate": ambiguous_compatible_rows,
    }
    return filtered, audit


def _source_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("train-*-of-00005.parquet")) if root.is_dir() else [root]
    if len(files) != 5:
        raise ValueError(f"Expected five PrimeIntellect shards under {root}, found {len(files)}")
    actual = tuple(sha256_file(path) for path in files)
    if actual != RAW_SHARD_SHA256:
        raise ValueError(f"Official PrimeIntellect shard SHA mismatch: {actual}")
    return files


def _bridge_file(root: Path) -> Path:
    candidates = sorted(root.rglob("*.parquet")) if root.is_dir() else [root]
    matches = [path for path in candidates if sha256_file(path) == BRIDGE_SHA256]
    if len(matches) != 1:
        raise ValueError(f"Expected one pinned runtime-filter bridge under {root}")
    return matches[0]


def _preprocess_row(example: dict[str, Any], index: int) -> dict[str, Any]:
    starter_code = str(example.get("starter_code", "") or "")
    metadata = example.get("metadata", {}) or {}
    tests = _decode_tests(example["tests"])
    for test in tests:
        if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
            test["metadata"] = {"func_name": str(metadata["func_name"])}
        else:
            test["metadata"] = {"func_name": None}
    return {
        "question": fetch_live_code_bench_system_prompt(
            str(example["problem"]), starter_code or None
        ),
        "ground_truth": json.dumps(tests, ensure_ascii=False),
        "data_source": "livecodebench",
        "uid": f"deepcoder_{index}",
        "index": index,
        "starter_code": starter_code,
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "source_index": int(example["source_index"]),
    }


def build_package(raw_root: Path, bridge_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite DeepCoder package {output_dir}")
    raw_files = _source_files(raw_root)
    bridge_file = _bridge_file(bridge_root)
    raw = load_dataset("parquet", data_files=[str(path) for path in raw_files], split="train")
    bridge = load_dataset("parquet", data_files=[str(bridge_file)], split="train")
    if len(raw) != RAW_ROWS or len(bridge) != BRIDGE_ROWS:
        raise ValueError(
            f"DeepCoder row-count mismatch: raw={len(raw)}, bridge={len(bridge)}"
        )

    filtered_rows, reconstruction_audit = reconstruct_runtime_filtered_rows(
        raw.to_list(), bridge.to_list()
    )
    if len(filtered_rows) != BRIDGE_ROWS:
        raise ValueError(f"Filtered row-count mismatch: {len(filtered_rows)}")

    shuffled = Dataset.from_list(filtered_rows).shuffle(seed=SPLIT_SEED)
    test_source = shuffled.select(range(TEST_ROWS))
    train_source = shuffled.select(range(TEST_ROWS, len(shuffled)))
    if len(train_source) != TRAIN_ROWS:
        raise ValueError(f"Train row-count mismatch: {len(train_source)}")

    train = Dataset.from_list(
        [_preprocess_row(row, index) for index, row in enumerate(train_source)]
    )
    test = Dataset.from_list(
        [_preprocess_row(row, index) for index, row in enumerate(test_source)]
    )
    calibration = Dataset.from_list(
        [
            {
                "question": train[index]["question"],
                "solution": str(train_source[index]["solutions"][0]),
                "source_index": int(train_source[index]["source_index"]),
            }
            for index in range(CALIBRATION_ROWS)
        ]
    )

    staging = output_dir.with_name(f"{output_dir.name}.incomplete-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        train_path = staging / "train.parquet"
        test_path = staging / "test.parquet"
        calibration_path = staging / "calibration.parquet"
        train.to_parquet(str(train_path))
        test.to_parquet(str(test_path))
        calibration.to_parquet(str(calibration_path))
        manifest = {
            "format": PACKAGE_FORMAT,
            "raw_repository": RAW_REPOSITORY,
            "raw_revision": RAW_REVISION,
            "raw_config": RAW_CONFIG,
            "raw_rows": len(raw),
            "raw_shard_sha256": list(RAW_SHARD_SHA256),
            "runtime_filter_bridge_repository": BRIDGE_REPOSITORY,
            "runtime_filter_bridge_revision": BRIDGE_REVISION,
            "runtime_filter_bridge_sha256": BRIDGE_SHA256,
            "filtered_rows": len(filtered_rows),
            "split_seed": SPLIT_SEED,
            "split_rule": "shuffle(seed=42); test=first 1000; train=rest",
            "train_rows": len(train),
            "test_rows": len(test),
            "calibration_rows": len(calibration),
            "train_source_indices_sha256": _json_sha256(train["source_index"]),
            "test_source_indices_sha256": _json_sha256(test["source_index"]),
            "calibration_source_indices_sha256": _json_sha256(
                calibration["source_index"]
            ),
            "files": {
                "train.parquet": sha256_file(train_path),
                "test.parquet": sha256_file(test_path),
                "calibration.parquet": sha256_file(calibration_path),
            },
            "reconstruction_audit": reconstruction_audit,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / "_SUCCESS").write_text("\n", encoding="utf-8")
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def register_package(package_root: Path, summary_path: Path) -> dict[str, Any]:
    package_root = resolve_package_root(package_root)
    manifest_path = package_root / "manifest.json"
    success_path = package_root / "_SUCCESS"
    if not manifest_path.is_file() or not success_path.is_file():
        raise ValueError(f"Incomplete DeepCoder package at {package_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != PACKAGE_FORMAT:
        raise ValueError(f"Unexpected DeepCoder package format: {manifest.get('format')}")

    datasets: dict[str, Dataset] = {}
    for split in ("train", "test"):
        path = package_root / f"{split}.parquet"
        expected = manifest["files"][path.name]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{path.name} SHA mismatch: expected {expected}, got {actual}")
        datasets[split] = load_dataset("parquet", data_files=[str(path)], split="train")
    if len(datasets["train"]) != TRAIN_ROWS or len(datasets["test"]) != TEST_ROWS:
        raise ValueError("Prepared DeepCoder split row count changed")

    registered_train = DatasetRegistry.register_dataset(
        DATASET_NAME, datasets["train"], "train"
    )
    registered_test = DatasetRegistry.register_dataset(
        DATASET_NAME, datasets["test"], "test"
    )
    summary = {
        "package_root": str(package_root.resolve()),
        "package_manifest_sha256": sha256_file(manifest_path),
        "train_rows": len(registered_train),
        "test_rows": len(registered_test),
        "train_path": registered_train.get_data_path(),
        "train_verl_path": registered_train.get_verl_data_path(),
        "test_path": registered_test.get_data_path(),
        "test_verl_path": registered_test.get_verl_data_path(),
        "source_contract": manifest,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--raw-root", required=True, type=Path)
    build.add_argument("--bridge-root", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    register = subparsers.add_parser("register")
    register.add_argument("--package-root", required=True, type=Path)
    register.add_argument("--summary", required=True, type=Path)
    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--mount-root", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "build":
        payload = build_package(args.raw_root, args.bridge_root, args.output_dir)
    elif args.command == "register":
        payload = register_package(args.package_root, args.summary)
    else:
        print(resolve_package_root(args.mount_root))
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
