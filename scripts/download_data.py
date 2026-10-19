"""Cloud opt-in download of the authors' preprocessed T&T / Deep Blending archive."""
import argparse
import json
from pathlib import Path
import urllib.request
import zipfile
from dataset import sha256

URL = 'https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip'
ROOT = Path(__file__).resolve().parents[1]

def safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination): raise ValueError('Unsafe ZIP path')
            if (info.external_attr >> 16) & 0o170000 == 0o120000: raise ValueError('ZIP symlinks are not supported')
            if target.exists() and not info.is_dir(): raise ValueError('Refusing overwrite: '+str(target))
        z.extractall(destination)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--accept-dataset-terms',action='store_true')
    a=p.parse_args()
    if not a.accept_dataset_terms:
        raise SystemExit('Read dataset/provider terms linked in docs/CLOUD.md, then pass --accept-dataset-terms.')
    root=ROOT/'data'; root.mkdir(exist_ok=True)
    archive=root/'tandt_db.zip'
    if not archive.exists():
        part=archive.with_suffix('.zip.part')
        print('Downloading about 650 MB:', URL, flush=True)
        with urllib.request.urlopen(URL,timeout=120) as response, part.open('wb') as f:
            while True:
                chunk=response.read(1024*1024)
                if not chunk: break
                f.write(chunk)
        part.replace(archive)
    print('Extracting without overwriting existing files',flush=True)
    safe_extract(archive,root)
    (root/'download.json').write_text(json.dumps({'url':URL,'sha256':sha256(archive),
        'note':'Locally recorded checksum, not a publisher-provided checksum'},indent=2))
    candidates=list(root.glob('**/train/sparse/0'))
    print('Detected train scene roots:',*[str(x.parents[1]) for x in candidates],sep='\n')
    print('If the archive has another outer folder, update source in configs/*.json.')

if __name__=='__main__': main()
