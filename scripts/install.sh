#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --upgrade pip wheel==0.48.0 packaging ninja==1.13.0 setuptools==78.1.1
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/runtime.txt vllm==0.10.0
python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
python -m pip install --no-deps .
python -m aspect.environment
