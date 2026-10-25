"""Archive one trusted run locally for file transfer; never upload automatically."""
import argparse
import gzip
from pathlib import Path
import tarfile
from artifact_io import staged_file
from dataset import sha256
from run import ROOT, load_config


def archive_run(out, directory):
    out, directory = Path(out), Path(directory)
    if not (out / 'manifest.json').is_file():
        raise ValueError('Run manifest missing')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (out.name + '.tar.gz')
    if target.exists():
        raise FileExistsError('Archive exists; use a different destination')
    files = sorted(p for p in out.rglob('*') if p.is_file())
    if any(p.is_symlink() or not p.resolve().is_relative_to(out.resolve()) for p in files):
        raise ValueError('Refusing archive with symlink/external files')
    # Publish checksum first, archive last. An orphan checksum is not an archive;
    # retry can regenerate it while an existing final archive is never overwritten.
    with staged_file(target) as temporary:
        with tarfile.open(temporary, 'w:gz') as tar:
            for path in files:
                tar.add(path, arcname=str(Path(out.name) / path.relative_to(out)), recursive=False)
        with gzip.open(temporary, 'rb') as stream:
            while stream.read(1024 * 1024):
                pass
        with tarfile.open(temporary, 'r:gz') as tar:
            if len(tar.getmembers()) != len(files):
                raise ValueError('Incomplete archive')
        digest = sha256(temporary)
        with staged_file(target.with_suffix(target.suffix + '.sha256')) as checksum:
            checksum.write_text(digest + '  ' + target.name + '\n', encoding='ascii')
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.json')
    args = parser.parse_args()
    print(archive_run(Path(load_config(args.config)['output']), ROOT / 'archives'))


if __name__ == '__main__':
    main()
