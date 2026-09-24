"""Select a small held-out subset only inside a non-canonical run registry."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(run_root):
    root = Path(run_root).resolve()
    run = json.loads((root / 'run.json').read_text())
    if run['canonical'] or not run.get('quick_smoke') or not run['method'].endswith('_Full_FT'):
        raise ValueError('A held-out subset is allowed only for explicit quick smoke')
    from rllm.data.dataset import DatasetRegistry
    if Path(DatasetRegistry._DATASET_DIR).resolve() != root / 'datasets':
        raise ValueError('Refusing to modify a dataset registry outside this smoke run')
    test = DatasetRegistry.load_dataset('dapo_math', 'test')
    if test is None or len(test) != 1412:
        raise ValueError('Prepare the complete official split before selecting a smoke subset')
    path = Path(test.get_data_path())
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    count = run['overrides']['+trainer.inline_full_validation']['expected_rows']
    if count != 8:
        raise ValueError('The quick profile selects exactly eight held-out problems')
    subset = DatasetRegistry.register_dataset('dapo_math', test.select(range(count)).get_data(), split='test')
    from aspect.audit import write_json
    write_json(root / 'smoke_subset.json', {
        'canonical': False, 'selection': 'first eight official held-out rows',
        'original_rows': 1412, 'original_split_sha256': original_sha,
        'subset_rows': count, 'indices': list(range(count)),
        'subset_sha256': hashlib.sha256(Path(subset.get_data_path()).read_bytes()).hexdigest(),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    prepare(parser.parse_args().run)
