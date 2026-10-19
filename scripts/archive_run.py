"""Archive one trusted run locally for scp/object storage; never upload automatically."""
import argparse
from pathlib import Path
import tarfile
from dataset import sha256
from run import ROOT, load_config

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/train.json')
    a=p.parse_args(); out=Path(load_config(a.config)['output'])
    if not (out/'manifest.json').is_file(): raise SystemExit('Run manifest missing')
    directory=ROOT/'archives'; directory.mkdir(exist_ok=True)
    target=directory/(out.name+'.tar.gz')
    if target.exists(): raise SystemExit('Archive exists; move/rename it before creating another')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(out.resolve()) for p in files):
        raise SystemExit('Refusing archive with symlink/external files')
    with tarfile.open(target,'w:gz') as tar:
        for path in files: tar.add(path,arcname=str(Path(out.name)/path.relative_to(out)),recursive=False)
    digest=sha256(target)
    target.with_suffix(target.suffix+'.sha256').write_text(digest+'  '+target.name+'\n')
    print(target);print('SHA256:',digest)
