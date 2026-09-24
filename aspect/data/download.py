"""Download pinned public inputs without any provider-specific storage API."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('math', 'code'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.task == 'math':
        from aspect.data.contracts import OFFICIAL_DATASET_CONTRACT
        from aspect.data.sources import _sha256_file
        contract = OFFICIAL_DATASET_CONTRACT['dapo']
        snapshot_download(contract['repository'], repo_type='dataset', revision=contract['revision'],
                          allow_patterns=[contract['source_file']], local_dir=root)
        if _sha256_file(root / contract['source_file']) != contract['source_sha256']:
            raise ValueError('Public Math source hash mismatch')
    else:
        from aspect.data.code import RAW_REPOSITORY, RAW_REVISION, BRIDGE_REPOSITORY, BRIDGE_REVISION, build_package, resolve_package_root
        raw = snapshot_download(RAW_REPOSITORY, repo_type='dataset', revision=RAW_REVISION,
                                allow_patterns=['primeintellect/*.parquet'], local_dir=root / 'source')
        bridge = snapshot_download(BRIDGE_REPOSITORY, repo_type='dataset', revision=BRIDGE_REVISION,
                                   allow_patterns=['*.parquet', '**/*.parquet'], local_dir=root / 'bridge')
        build_package(Path(raw), Path(bridge), root / 'prepared')
        resolve_package_root(root / 'prepared')
    print(json.dumps({'task': args.task, 'data_directory': str(root)}))


if __name__ == '__main__':
    main()
