"""CPU checks connecting summary fields to model, images and timing evidence."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from PIL import Image
from dataset import sha256


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def ply_count(path):
    with Path(path).open('rb') as stream:
        for _ in range(1000):
            line = stream.readline().decode('ascii').strip()
            if line.startswith('element vertex '):
                return int(line.split()[-1])
            if line == 'end_header':
                break
    raise ValueError('PLY vertex count not found')


def image_pairs(model, iteration, count):
    expected = {f'{i:05d}.png' for i in range(count)}
    if not expected:
        raise ValueError('Empty test set')
    base = Path(model) / f'test/ours_{iteration}'
    for kind in ('renders', 'gt'):
        if {p.name for p in (base / kind).glob('*.png')} != expected:
            raise ValueError('Incomplete/unexpected image set: ' + str(base / kind))
    sizes = set()
    for name in sorted(expected):
        with Image.open(base / 'renders' / name) as a, Image.open(base / 'gt' / name) as b:
            a.load(); b.load()
            if a.mode != 'RGB' or b.mode != 'RGB' or a.size != b.size:
                raise ValueError('Image size/mode mismatch: ' + name)
            sizes.add(a.size)
    return sorted(map(list, sizes))


def checked_benchmark(b, count, views, resolution, sizes, ply, config):
    samples = b['samples_ms']
    if (b['gaussian_count'] != count or b['test_views'] != views or
        not isinstance(b['repeats'], int) or b['repeats'] < 1 or
        len(samples) != views * b['repeats'] or
        not all(math.isfinite(x) and x > 0 for x in samples) or
        not math.isclose(statistics.mean(samples), b['mean_ms'], rel_tol=1e-7) or
        not math.isclose(1000 / b['mean_ms'], b['fps_from_mean_ms'], rel_tol=1e-7) or
        b['config_resolution'] != resolution or sorted(b['image_sizes']) != sizes or
        not isinstance(b['peak_torch_allocated_bytes'], int) or b['peak_torch_allocated_bytes'] < 0):
        raise ValueError('Benchmark disagrees with model/images/statistics')
    if 'median_ms' in b and not math.isclose(statistics.median(samples), b['median_ms'], rel_tol=1e-7):
        raise ValueError('Benchmark median differs from samples')
    provenance = b.get('source')
    if provenance is None:
        return 'legacy: structural/statistical checks only; model freshness unproven'
    if provenance != {'ply_sha256': sha256(ply), 'config_sha256': config_digest(config)}:
        raise ValueError('Benchmark source model/config changed')
    return 'model and configuration hashes verified'


def checked_row(row, model, config, views, baseline, recovery_steps):
    model = Path(model)
    ply = model / f'point_cloud/iteration_{config["iterations"]}/point_cloud.ply'
    count = ply_count(ply)
    sizes = image_pairs(model, config['iterations'], views)
    b = json.loads((model / 'benchmark.json').read_text(encoding='utf-8'))
    checked_benchmark(b, count, views, config['resolution'], sizes, ply, config)
    expected = {'gaussian_count': count, 'ply_bytes': ply.stat().st_size,
                'recovery_steps': recovery_steps,
                **{k: b[k] for k in ('mean_ms', 'fps_from_mean_ms', 'peak_torch_allocated_bytes')},
                **{'delta_' + k: row[k] - baseline[k] for k in ('PSNR', 'SSIM', 'LPIPS')}}
    if any(row.get(k) != value for k, value in expected.items()):
        raise ValueError('Summary resource/performance fields disagree with artifacts')
