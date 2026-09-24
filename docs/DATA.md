# Public Data Provenance

## Model revisions

Public reference revisions recorded by the experiment input contract:

| Model | Revision |
| --- | --- |
| Qwen/Qwen3-0.6B | `025f7769d0cc53135b8051d32416c7bb42669467` |
| Qwen/Qwen3-1.7B | `0060bc56d46589041c1048efd1a397421b1142b5` |
| Qwen/Qwen3-4B | `531c80e289d6cff3a7cd8c0db8110231d23a6f7a` |

Download with `hf download REPOSITORY --revision REVISION --local-dir DIRECTORY`.
Freeze the revision and retain the downloaded file hashes with the experiment;
do not rely on a moving `main` branch. These references do not certify that all
historical runs, including earlier diagnostic runs, used identical model bytes.

## Task sources

Math: `open-r1/DAPO-Math-17k-Processed`, revision
`31dd309567e3da778038cc87d868b6097a3ccf68`, configuration `en`.
The parquet source SHA256 is
`40f672fe8b6dbeee953ae5acc75a38e99144e682450e07c625c87c734ec76285`.
The 14,116 rows are mapped with the original prompt/answer adapter and split
using Hugging Face Datasets `train_test_split(test_size=0.1, seed=42)`.
Calibration selects the first 50 training rows, never the test split.

Code: `agentica-org/DeepCoder-Preview-Dataset`, revision
`780aeec7b7716e34f0b9e02bb624fc6ad8384725`, configuration `primeintellect`.
The public filtered bridge is `mnoukhov/deepcoder_primeintellect`, revision
`90f49967b15f25ce5a8513fb3606d13dc41e6a94`.
The preparation script verifies every bridge problem and retained test against
the original source, restores source/test order, and performs the seed-42
shuffle, first-1,000 test split. It does not fabricate the upstream missing
runtime-filter file or rerun a potentially different filtering policy.

Exact file hashes, row counts, source indices, and split identities are checked
by `aspect/data/code.py`. Parquet binary hashes can depend on writer versions;
use the pinned Datasets/PyArrow environment rather than disabling the checks.
All preparation outputs are local artifacts and excluded from source control.

Dataset references identify public sources; licenses and use restrictions remain
those of the original data providers. This repository distributes preparation
code, not the underlying dataset contents.
