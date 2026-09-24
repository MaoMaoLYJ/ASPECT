"""Local dataset loading and the original deterministic Math preprocessing."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from datasets import Dataset as HFDataset, DatasetDict, load_dataset, load_from_disk
SUPPORTED_SUFFIXES = {".parquet", ".json", ".jsonl", ".csv"}

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _select_source_file(
    path: Path,
    preferred_split: str,
    *,
    config_hint: str | None = None,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported dataset file: {path}")
        return path

    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No supported dataset files under {path}")

    candidates = files
    if config_hint:
        config_candidates = [
            candidate
            for candidate in candidates
            if config_hint.lower()
            in {part.lower() for part in candidate.relative_to(path).parts[:-1]}
        ]
        if config_candidates:
            candidates = config_candidates

    split_candidates = [
        candidate
        for candidate in candidates
        if preferred_split.lower() in candidate.name.lower()
    ]
    if split_candidates:
        candidates = split_candidates

    if len(candidates) != 1:
        rendered = "\n  - ".join(str(candidate) for candidate in candidates)
        raise ValueError(
            f"Ambiguous dataset mount {path}; expected one {config_hint or ''} "
            f"{preferred_split} source file, found {len(candidates)}:\n  - {rendered}"
        )
    return candidates[0]

def _load_mount(
    path: Path,
    preferred_split: str,
    *,
    config_hint: str | None = None,
) -> tuple[HFDataset, Path | None]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {path}")

    if path.is_dir() and (
        (path / "dataset_dict.json").exists() or (path / "state.json").exists()
    ):
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            if preferred_split in loaded:
                return loaded[preferred_split], None
            if len(loaded) == 1:
                return next(iter(loaded.values())), None
            raise ValueError(f"No '{preferred_split}' split in {path}; found {list(loaded)}")
        return loaded, None

    source_file = _select_source_file(
        path,
        preferred_split,
        config_hint=config_hint,
    )
    builder = {
        ".parquet": "parquet",
        ".json": "json",
        ".jsonl": "json",
        ".csv": "csv",
    }[source_file.suffix.lower()]
    return (
        load_dataset(builder, data_files=[str(source_file)], split="train"),
        source_file,
    )

def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    raise ValueError(f"Cannot convert question value to text: {type(value).__name__}")

def _prepare_dapo(dataset: HFDataset) -> tuple[HFDataset, HFDataset]:
    required = {"prompt", "solution"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(f"DAPO source is missing columns {sorted(missing)}; found {dataset.column_names}")

    def convert(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "question": _text(example["prompt"]),
            "final_answer": example["solution"],
            "data_source": example.get("data_source", "dapo_math"),
            "ability": example.get("ability", ""),
            "reward_model": example.get("reward_model", {}),
            "extra_info": example.get("extra_info", {}),
        }

    mapped = dataset.map(convert)
    split = mapped.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]
