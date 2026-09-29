"""Verify the analysis allowlist; optionally stage only its files (never commit/push)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'abstract','paper','figures','tables','ProjectionDataset','.git'}


def validate():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    paths=[]
    for item in manifest['files']:
        relative=Path(item['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert not EXCLUDED.intersection(relative.parts),relative
        assert not str(relative).startswith(('data/raw/','data/processed/','results_sd10/','results_zscore/'))
        assert relative.name!='ABSTRACT.md' and not relative.name.startswith('tentative_full_paper')
        path=ROOT/relative
        assert path.is_file() and not path.is_symlink(),relative
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],relative
        assert path.stat().st_size==item['bytes'],relative
        if path.suffix=='.csv':
            assert relative.parts[0]=='reference_results',relative
            with path.open() as f:columns=next(csv.reader(f))
            assert not {'player_id','PlayerId','Name','y_true','prediction','focal_player_id','comparison_player_id'}.intersection(columns),relative
        paths.append(str(relative))
    assert len(paths)==len(set(paths))
    return paths+['RELEASE_MANIFEST.json']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',action='store_true')
    args=parser.parse_args()
    paths=validate()
    if args.stage:
        existing=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0')
        unexpected=set(filter(None,existing))-set(paths)
        if unexpected:raise SystemExit(f'Unrelated staged paths; index left unchanged: {sorted(unexpected)}')
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0')
        assert set(filter(None,staged))<=set(paths)
    print(f'Validated {len(paths)} analysis files. '+('Staged allowlisted changes only.' if args.stage else 'No staging performed.'))


if __name__=='__main__':main()
