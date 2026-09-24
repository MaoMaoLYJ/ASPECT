import json

from pathlib import Path

import pytest

import aspect.data.code as code_data

from aspect.data.code import (
    BRIDGE_PROMPT_SUFFIX,
    reconstruct_runtime_filtered_rows,
    resolve_package_root,
)

def _raw(problem: str, tests: list[dict], solution: str = "print(1)") -> dict:
    return {
        "problem": problem,
        "solutions": [solution],
        "tests": json.dumps(tests),
    }

def _bridge(problem: str, tests: list[dict]) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": problem + BRIDGE_PROMPT_SUFFIX,
            }
        ],
        "ground_truth": json.dumps(tests),
        "dataset": "code_stdio",
    }

def test_reconstructs_filtered_rows_in_original_source_order():
    first_tests = [
        {"type": "stdin_stdout", "input": "1\n", "output": "1\n"},
        {"type": "stdin_stdout", "input": "2\n", "output": "2\n"},
    ]
    third_tests = [
        {"type": "stdin_stdout", "input": "3\n", "output": "3\n"},
    ]
    raw_rows = [
        _raw("problem zero", first_tests),
        _raw("problem removed", first_tests),
        _raw("problem two", third_tests),
    ]
    bridge_rows = [
        _bridge("problem zero", [first_tests[1], first_tests[0]]),
        _bridge("problem two", third_tests),
    ]

    filtered, audit = reconstruct_runtime_filtered_rows(raw_rows, bridge_rows)

    assert [row["source_index"] for row in filtered] == [0, 2]
    assert json.loads(filtered[0]["tests"]) == first_tests
    assert filtered[0]["solutions"] == ["print(1)"]
    assert audit == {
        "raw_rows": 3,
        "filtered_rows": 2,
        "excluded_rows": 1,
        "unique_source_rows": 2,
        "bridge_rows_in_source_order": True,
        "rows_with_later_compatible_duplicate": 0,
    }

def test_rejects_filtered_tests_that_are_not_a_source_multiset_subset():
    source_test = {"type": "stdin_stdout", "input": "1\n", "output": "1\n"}
    foreign_test = {"type": "stdin_stdout", "input": "9\n", "output": "9\n"}

    with pytest.raises(ValueError, match="cannot be mapped"):
        reconstruct_runtime_filtered_rows(
            [_raw("problem", [source_test])],
            [_bridge("problem", [foreign_test])],
        )

def test_rejects_bridge_rows_that_regress_source_order():
    test = {"type": "stdin_stdout", "input": "1\n", "output": "1\n"}

    with pytest.raises(ValueError, match="official source order"):
        reconstruct_runtime_filtered_rows(
            [_raw("first", [test]), _raw("second", [test])],
            [_bridge("second", [test]), _bridge("first", [test])],
        )

def _write_test_package(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root.mkdir(parents=True)
    files = {
        "train.parquet": b"train",
        "test.parquet": b"test",
        "calibration.parquet": b"calibration",
    }
    expected_files = {}
    for name, content in files.items():
        path = root / name
        path.write_bytes(content)
        expected_files[name] = code_data.sha256_file(path)
    manifest = {
        "format": code_data.PACKAGE_FORMAT,
        "raw_repository": code_data.RAW_REPOSITORY,
        "raw_revision": code_data.RAW_REVISION,
        "runtime_filter_bridge_repository": code_data.BRIDGE_REPOSITORY,
        "runtime_filter_bridge_revision": code_data.BRIDGE_REVISION,
        "split_seed": code_data.SPLIT_SEED,
        "train_rows": code_data.TRAIN_ROWS,
        "test_rows": code_data.TEST_ROWS,
        "calibration_rows": code_data.CALIBRATION_ROWS,
        "files": expected_files,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    success_path = root / "_SUCCESS"
    success_path.write_text("\n")
    monkeypatch.setattr(code_data, "EXPECTED_PACKAGE_FILE_SHA256", expected_files)
    monkeypatch.setattr(
        code_data,
        "EXPECTED_PACKAGE_MANIFEST_SHA256",
        code_data.sha256_file(manifest_path),
    )
    monkeypatch.setattr(
        code_data,
        "EXPECTED_PACKAGE_SUCCESS_SHA256",
        code_data.sha256_file(success_path),
    )
    return root

def test_resolves_exact_package_from_nested_dataset_directory(tmp_path, monkeypatch):
    package = _write_test_package(tmp_path / "uploaded-folder", monkeypatch)
    (tmp_path / "README.md").write_text("Dataset metadata\n")

    assert resolve_package_root(tmp_path) == package.resolve()

def test_rejects_ambiguous_packages_under_mount(tmp_path, monkeypatch):
    first = _write_test_package(tmp_path / "first", monkeypatch)
    second = tmp_path / "second"
    second.mkdir()
    for source in first.iterdir():
        (second / source.name).write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="exactly one"):
        resolve_package_root(tmp_path)
