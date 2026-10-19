"""Validate COLMAP data and reproduce the pinned upstream split (no torch needed)."""
import hashlib
import json
import struct
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def unpack(f, fmt):
    size = struct.calcsize('<' + fmt)
    data = f.read(size)
    if len(data) != size: raise ValueError('Truncated COLMAP binary')
    return struct.unpack('<' + fmt, data)

def binary_names(path):
    records = []
    with path.open('rb') as f:
        for _ in range(unpack(f, 'Q')[0]):
            fields = unpack(f, 'idddddddi')
            name = bytearray()
            while True:
                byte = f.read(1)
                if not byte: raise ValueError('Truncated image name')
                if byte == b'\0': break
                name.extend(byte)
            records.append((name.decode('utf-8'), fields[-1]))
            count = unpack(f, 'Q')[0]
            f.seek(24 * count, 1)
            if f.tell() > path.stat().st_size: raise ValueError('Truncated image points')
    return records

def text_names(path):
    records = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            values = line.split()
            if len(values) < 10: raise ValueError('Invalid COLMAP images.txt pose line')
            records.append((values[9], int(values[8])))
            if next(f, None) is None: raise ValueError('Missing COLMAP observations line')
    return records

def validate(source, images='images'):
    source = Path(source).resolve()
    sparse = source / 'sparse/0'
    if not sparse.is_dir():
        raise ValueError('Expected COLMAP sparse/0. First baseline uses undistorted COLMAP data, not raw photos.')
    camera_ids = set()
    files = []
    if (sparse / 'images.bin').is_file() and (sparse / 'cameras.bin').is_file():
        files = [sparse / 'images.bin', sparse / 'cameras.bin']
        names = binary_names(files[0])
        with files[1].open('rb') as f:
            for _ in range(unpack(f, 'Q')[0]):
                camera_id, model, width, height = unpack(f, 'iiQQ')
                if model not in (0, 1):
                    raise ValueError('Only undistorted SIMPLE_PINHOLE/PINHOLE cameras supported; run convert.py first.')
                if min(width, height) < 1: raise ValueError('Invalid camera size')
                unpack(f, 'd' * (3 if model == 0 else 4))
                camera_ids.add(camera_id)
    else:
        files = [sparse / 'images.txt', sparse / 'cameras.txt']
        names = text_names(files[0])
        for line in files[1].read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'): continue
            values = line.split()
            if values[1] not in ('PINHOLE', 'SIMPLE_PINHOLE'):
                raise ValueError('Camera images must be undistorted first')
            camera_ids.add(int(values[0]))
    if len(names) < 8: raise ValueError('Need at least 8 registered images for this evaluation protocol')
    if any(camera not in camera_ids for _, camera in names): raise ValueError('Missing camera intrinsics')
    points = next((sparse / n for n in ['points3D.ply','points3D.bin','points3D.txt'] if (sparse / n).is_file()), None)
    if points is None: raise ValueError('Sparse initialization points missing')
    # Match upstream: basename, then split at FIRST dot, not Path.stem.
    names = sorted([Path(n).name for n, _ in names], key=lambda n: n.split('.')[0])
    if len({n.split('.')[0] for n in names}) != len(names):
        raise ValueError('Upstream image_name collision (before first dot); rename/reprocess dataset')
    missing = [n for n in names if not (source / images / n).is_file()]
    if missing: raise ValueError('Missing registered images: ' + repr(missing[:5]))
    return {'source': str(source), 'images': images,
            'train': [n for i, n in enumerate(names) if i % 8 != 0],
            'test': [n for i, n in enumerate(names) if i % 8 == 0],
            'split': 'sorted upstream image_name; index % 8 == 0 held out',
            'camera_hashes': {p.name: sha256(p) for p in files},
            'initialization_file': points.name, 'initialization_sha256': sha256(points),
            'image_sha256': {n: sha256(source / images / n) for n in names}}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('source'); p.add_argument('--images', default='images')
    a = p.parse_args()
    report = validate(a.source, a.images)
    print(json.dumps(report, indent=2, ensure_ascii=False))
