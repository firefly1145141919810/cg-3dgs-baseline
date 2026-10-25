"""Cloud opt-in download of the authors' preprocessed T&T / Deep Blending archive."""
import argparse
import json
import re
from pathlib import Path
import urllib.error
import urllib.request
import zipfile
import zlib
import shutil
from artifact_io import staged_file
from dataset import sha256

URL = 'https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip'
ROOT = Path(__file__).resolve().parents[1]

def copy_download(response, output, progress, total=None):
    """Copy small chunks so even slow connections produce visible updates."""
    downloaded = 0
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        if total is not None and downloaded + len(chunk) > total:
            raise OSError('Response exceeded its declared length')
        output.write(chunk)
        downloaded += len(chunk)
        progress.update(len(chunk))
    if total is not None and downloaded != total:
        raise OSError(f'Incomplete download: received {downloaded} of {total} bytes')

def verify_zip(path):
    print('Checking ZIP integrity before extraction...', flush=True)
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad is not None:
            raise OSError('ZIP CRC check failed: ' + bad)

def download_archive(archive, progress_factory, opener=urllib.request.urlopen):
    """Resume HTTP ranges; validate responses and CRC before publishing the ZIP."""
    part = archive.with_suffix('.zip.part')
    meta_path = part.with_suffix('.part.json')
    offset = part.stat().st_size if part.exists() else 0
    meta = {}
    if offset and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('url') != URL:
            raise OSError('Partial download belongs to another URL; preserve/move it before restarting.')
    headers = {'Accept-Encoding': 'identity'}
    validator = meta.get('validator')
    if offset:
        headers['Range'] = f'bytes={offset}-'
        if validator:
            headers['If-Range'] = validator
        else:
            print('Legacy partial file has no version metadata; resuming with final ZIP CRC validation.', flush=True)
        print(f'Requesting resume at {offset:,} bytes.', flush=True)
    print('Connecting to download server (socket timeout: 120 s):', URL, flush=True)
    try:
        response = opener(urllib.request.Request(URL, headers=headers), timeout=120)
    except urllib.error.HTTPError as exc:
        content_range = exc.headers.get('Content-Range', '')
        exc.close()
        match = re.fullmatch(r'bytes \*/(\d+)', content_range)
        if exc.code != 416:
            raise
        if offset and match and offset == int(match[1]):
            # A previous run may have downloaded everything and stopped before rename.
            verify_zip(part)
            part.replace(archive)
            return
        raise OSError('Server rejected the resume offset (HTTP 416). Partial file preserved; do not append.') from exc
    with response:
        status = response.status
        length = response.headers.get('Content-Length')
        length = int(length) if length and length.isdigit() else None
        etag = response.headers.get('ETag')
        new_validator = (etag if etag and not etag.startswith('W/') else
                         response.headers.get('Last-Modified'))
        if status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match:
                raise OSError('Missing or invalid Content-Range; partial file preserved')
            start, end, total = map(int, match.groups())
            if start != offset or end != total - 1 or end < start:
                raise OSError('Unexpected response range; partial file preserved')
            expected = total - offset
            if length is not None and length != expected:
                raise OSError('Content-Length disagrees with Content-Range')
            if validator and new_validator and validator != new_validator:
                raise OSError('Remote file version changed during resume; partial file preserved')
            if meta.get('total') is not None and meta['total'] != total:
                raise OSError('Remote file size changed during resume; partial file preserved')
            mode = 'ab' if offset else 'wb'
        elif status == 200:
            if offset:
                print('Server returned a full file (range unsupported or file changed); restarting from zero.', flush=True)
            offset = 0
            total = expected = length
            mode = 'wb'
        else:
            raise OSError(f'Unexpected HTTP status {status}; partial file preserved')
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise OSError('Encoded responses cannot safely be resumed')
        if total is None:
            print('Unknown total size: showing downloaded bytes and speed only.', flush=True)
        with part.open(mode) as f:
            meta_path.write_text(json.dumps({'url': URL, 'validator': new_validator or
                (validator if status == 206 else None), 'total': total}), encoding='utf-8')
            with progress_factory(total=total, initial=offset, desc='Downloading', unit='B',
                    unit_scale=True, unit_divisor=1024, mininterval=0.5, dynamic_ncols=True) as progress:
                copy_download(response, f, progress, expected)
    verify_zip(part)
    part.replace(archive)

def safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination): raise ValueError('Unsafe ZIP path')
            if (info.external_attr >> 16) & 0o170000 == 0o120000: raise ValueError('ZIP symlinks are not supported')
            if target.exists() and not info.is_dir():
                if not target.is_file() or target.stat().st_size != info.file_size:
                    raise ValueError('Refusing overwrite: ' + str(target))
                crc = 0
                with target.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        crc = zlib.crc32(block, crc)
                if crc != info.CRC:
                    raise ValueError('Refusing overwrite: ' + str(target))
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with staged_file(target) as temporary:
                    with z.open(info) as source, temporary.open('wb') as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--accept-dataset-terms',action='store_true')
    a=p.parse_args()
    if not a.accept_dataset_terms:
        raise SystemExit('Read dataset/provider terms linked in docs/CLOUD.md, then pass --accept-dataset-terms.')
    root=ROOT/'data'; root.mkdir(exist_ok=True)
    archive=root/'tandt_db.zip'
    if not archive.exists():
        from tqdm import tqdm
        try:
            download_archive(archive, tqdm)
        except (OSError, zipfile.BadZipFile, KeyboardInterrupt) as exc:
            print('\nDownload stopped. Partial data is kept. Network interruptions can resume on the next run; integrity/version errors need inspection. Do not run two downloads concurrently.', flush=True)
            raise SystemExit(str(exc) or 'Interrupted by user') from exc
    print('Extracting without overwriting existing files',flush=True)
    safe_extract(archive,root)
    (root/'download.json').write_text(json.dumps({'url':URL,'sha256':sha256(archive),
        'note':'Locally recorded checksum, not a publisher-provided checksum'},indent=2))
    candidates=list(root.glob('**/train/sparse/0'))
    print('Detected train scene roots:',*[str(x.parents[1]) for x in candidates],sep='\n')
    print('If the archive has another outer folder, update source in configs/*.json.')

if __name__=='__main__': main()
