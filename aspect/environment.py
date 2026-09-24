"""Report package provenance and reject a shadowed training backend."""
import importlib
import importlib.metadata
import json
from pathlib import Path


def main():
    import torch
    import rllm
    import verl
    root = Path(__file__).resolve().parents[1]
    for module in (rllm, verl):
        if root not in Path(module.__file__).resolve().parents:
            raise RuntimeError(f'{module.__name__} is shadowed by another installation')
    expected = {'torch': '2.7.1', 'vllm': '0.10.0', 'transformers': '4.57.0',
                'peft': '0.19.1', 'ray': '2.55.1'}
    actual = {name: importlib.metadata.version(name) for name in expected}
    mismatches = {name: value for name, value in actual.items() if value.split('+')[0] != expected[name]}
    if mismatches:
        raise RuntimeError(f'Research runtime version mismatch: {mismatches}')
    print(json.dumps({'packages': actual, 'cuda': torch.version.cuda,
                      'cuda_available': torch.cuda.is_available(), 'bundled_backend': True}, indent=2))


if __name__ == '__main__':
    main()
