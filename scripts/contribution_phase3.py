"""Compare opacity vs visible alpha-compositing weight at a fixed 50% budget.

No CUDA rebuild, optimizer updates, test-view scoring or recovery in this phase.
"""
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from plyfile import PlyData, PlyElement
from run import ROOT, UPSTREAM, resolve, write_json, command, execute
from dataset import sha256
from compression_phase1 import read_json, render_config, compare_images, checked_metrics, assert_image_sets
from recovery_phase2 import verify_parent, ply_path
import recovery_phase2
from prune_ply import read_vertices, select_indices

VARIANTS = ('identity100', 'opacity50', 'contribution50')


def code_hashes():
    return {**recovery_phase2.code_hashes(), 'contribution_phase3.py': sha256(Path(__file__))}


def selected_views(count, limit):
    if type(limit) is not int or limit < 0 or count < 1:
        raise ValueError('Use --views 0 for all train cameras, positive for smoke')
    return list(range(count if limit == 0 else min(count, limit)))


def checked_scores(scores, count):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (count,) or not np.isfinite(scores).all() or (scores < 0).any() or scores.sum() <= 0:
        raise ValueError('Invalid/nonpositive contribution scores')
    return scores


def write_selection(source, target, scores):
    vertices = read_vertices(source)
    scores = checked_scores(scores, len(vertices))
    indices = select_indices(scores, .5)
    if target.exists():
        raise FileExistsError('Refusing to overwrite derived model')
    target.parent.mkdir(parents=True)
    selected = vertices[indices].copy()
    PlyData([PlyElement.describe(selected, 'vertex')], text=False, byte_order='<').write(target)
    loaded = read_vertices(target)
    if len(loaded) != len(selected) or any(loaded[k].tobytes() != selected[k].tobytes() for k in selected.dtype.names):
        raise ValueError('Selected PLY fields changed')
    return indices


def context(experiment):
    info = read_json(experiment / 'experiment.json')
    if info['schema'] != 3 or info['script_sha256'] != code_hashes():
        raise ValueError('Code/schema changed; use a new experiment directory')
    parent = Path(info['parent'])
    previous = verify_parent(parent)
    if (sha256(parent / 'experiment.json') != info['parent_manifest_sha256'] or
        sha256(parent / 'evaluation.json') != info['parent_evaluation_sha256']):
        raise ValueError('Parent changed')
    return info, previous


def score(parent, experiment, views):
    previous = verify_parent(parent)
    train_count = len(read_json(parent / 'dataset.json')['train'])
    chosen = selected_views(train_count, views)
    if not experiment.exists():
        for p in (parent, Path(previous['baseline_config']['source']), Path(previous['baseline_config']['output'])):
            if experiment == p or experiment.is_relative_to(p) or p.is_relative_to(experiment):
                raise ValueError('Output overlaps a source directory')
        experiment.mkdir(parents=True)
        write_json(experiment / 'experiment.json', {
            'schema': 3, 'parent': str(parent), 'parent_manifest_sha256': sha256(parent / 'experiment.json'),
            'parent_evaluation_sha256': sha256(parent / 'evaluation.json'),
            'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'script_sha256': code_hashes(), 'view_limit': views, 'train_indices': chosen,
            'formal': len(chosen) == train_count, 'keep_fraction': .5,
            'method': 'mean train-view pixel alpha*T weight via precomputed-color gradient',
            'recovery_steps': 0})
    info, previous = context(experiment)
    if info['parent'] != str(parent) or info['view_limit'] != views:
        raise ValueError('Resume must use the same parent and --views')
    if (experiment / 'scoring.json').exists():
        scoring(experiment, previous)
        print('Scoring already complete and verified')
        return
    from doctor import diagnose
    env = diagnose()
    if env['errors']:
        raise ValueError(str(env))
    write_json(experiment / 'scoring_environment.json', env)
    sys.path.insert(0, str(UPSTREAM))
    import torch
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from tqdm import tqdm
    c = dict(previous['baseline_config'], output=str(parent / 'identity100'))
    dataset = render_config(c); dataset.data_device = 'cpu'
    model = GaussianModel(3)
    scene = Scene(dataset, model, load_iteration=c['iterations'], shuffle=False)
    for name in ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation'):
        getattr(model, name).requires_grad_(False)
    cameras = scene.getTrainCameras()
    if len(cameras) != train_count:
        raise ValueError('Unexpected train camera count')
    count = previous['variants']['identity100']['kept_count']
    if model.get_xyz.shape[0] != count:
        raise ValueError('Unexpected model count')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, device='cuda')  # Diagnostic colors only, never saved as model parameters.
    colors = torch.full((count, 3), .5, device='cuda', requires_grad=True)
    total, records, seconds = np.zeros(count, dtype=np.float64), [], 0.
    checkpoint = experiment / 'score_checkpoint.npz'
    manifest_hash = sha256(experiment / 'experiment.json')
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(saved['manifest']) != manifest_hash:
                raise ValueError('Score checkpoint belongs to another experiment')
            total = saved['total'].copy()
            records = json.loads(str(saved['records']))
            seconds = float(saved['seconds'])
        if total.shape != (count,) or not np.isfinite(total).all() or (total < 0).any() or len(records) > len(chosen):
            raise ValueError('Invalid scoring checkpoint')
        if [r['train_index'] for r in records] != chosen[:len(records)]:
            raise ValueError('Invalid checkpoint view sequence')
    started = time.perf_counter()
    for index in tqdm(chosen[len(records):], desc='Train-view contribution'):
        cam = cameras[index]
        image = render(cam, model, pipe, background, override_color=colors)['render']
        # d mean(RGB image) / d per-Gaussian RGB; sum RGB gradients = mean_p(T*alpha).
        gradient, = torch.autograd.grad(image.mean(), colors)
        per_view = gradient.sum(dim=1)
        if not torch.isfinite(per_view).all().item() or (per_view < 0).any().item():
            raise ValueError('Nonfinite/negative color-gradient weight')
        mass = float(per_view.sum().item())
        coverage = float(image.detach().mean().item()) * 2  # colors=.5, bg=0.
        if not 0 < mass <= 1.0001 or not math.isclose(mass, coverage, rel_tol=2e-4, abs_tol=2e-5):
            raise ValueError('Weight-sum identity failed; do not trust these scores')
        record = {'train_index': index, 'camera': cam.image_name, 'weight_sum': mass,
                  'coverage': coverage, 'width': cam.image_width, 'height': cam.image_height}
        if not records:
            # Independent finite difference of a fixed, non-test-selected subset.
            with torch.no_grad():
                perturbed = colors.detach().clone()
                perturbed[::17] += .01
                shifted = render(cam, model, pipe, background, override_color=perturbed)['render']
                measured = float((shifted.mean() - image.detach().mean()).item()) / .01
            predicted = float(per_view[::17].sum().item())
            if not math.isclose(measured, predicted, rel_tol=.005, abs_tol=2e-5):
                raise ValueError('Color-gradient finite-difference check failed')
            record.update(finite_difference=measured, predicted_derivative=predicted)
        total += per_view.detach().cpu().numpy().astype(np.float64)
        records.append(record)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - started
        # Atomic, pickle-free checkpoint after every view; excludes checkpoint I/O from timing.
        temp = experiment / 'score_checkpoint.tmp'
        with temp.open('wb') as f:
            np.savez(f, total=total, records=json.dumps(records), seconds=seconds, manifest=manifest_hash)
        temp.replace(checkpoint)
        started = time.perf_counter()
    scores = checked_scores(total / len(chosen), count)
    np.save(experiment / 'scores.npy', scores, allow_pickle=False)
    write_json(experiment / 'scoring.json', {
        'complete': True, 'manifest_sha256': manifest_hash, 'scores_sha256': sha256(experiment / 'scores.npy'),
        'source_ply_sha256': previous['variants']['identity100']['ply_sha256'],
        'views': records, 'gaussian_count': count, 'seconds_excluding_checkpoint_io': seconds,
        'min': float(scores.min()), 'max': float(scores.max()), 'sum': float(scores.sum()),
        'nonzero_count': int(np.count_nonzero(scores)),
        'note': 'No GT pixels, no optimizer, no removal-error claim; float32 GPU atomics, float64 accumulation'})
    print('Scoring complete. Formal:', info['formal'])


def scoring(experiment, previous):
    report = read_json(experiment / 'scoring.json')
    info = read_json(experiment / 'experiment.json')
    if (not report['complete'] or report['manifest_sha256'] != sha256(experiment / 'experiment.json')
        or report['scores_sha256'] != sha256(experiment / 'scores.npy')
        or [r['train_index'] for r in report['views']] != info['train_indices']
        or report['source_ply_sha256'] != previous['variants']['identity100']['ply_sha256']):
        raise ValueError('Invalid/changed scoring result')
    return checked_scores(np.load(experiment / 'scores.npy', allow_pickle=False),
                          previous['variants']['identity100']['kept_count'])


def prepare(experiment):
    info, previous = context(experiment)
    scores = scoring(experiment, previous)
    if (experiment / 'prepared.json').exists() or any((experiment / v).exists() for v in VARIANTS):
        raise FileExistsError('Preparation exists/was interrupted; use a new experiment')
    parent = Path(info['parent'])
    c = previous['baseline_config']
    result = {}
    for v in VARIANTS:
        model = experiment / v
        target = ply_path(model, c['iterations'])
        if v == 'contribution50':
            indices = write_selection(ply_path(parent / 'identity100', c['iterations']), target, scores)
            np.save(model / 'kept_indices.npy', indices, allow_pickle=False)
            opacity_indices = np.load(parent / 'opacity50/kept_indices.npy', allow_pickle=False)
            overlap = len(np.intersect1d(indices, opacity_indices))
            result[v] = {'count': len(indices), 'indices_sha256': sha256(model / 'kept_indices.npy'),
                         'overlap_with_opacity50': overlap, 'overlap_fraction': overlap / len(indices)}
        else:
            target.parent.mkdir(parents=True)
            shutil.copy2(ply_path(parent / v, c['iterations']), target)
            result[v] = {'count': previous['variants'][v]['kept_count']}
        result[v]['ply_sha256'] = sha256(target)
        vc = dict(c, output=str(model))
        write_json(model / 'config.json', vc)
        write_json(model / 'manifest.json', {'config': vc})
        (model / 'cfg_args').write_text(repr(render_config(vc)), encoding='utf-8')
    write_json(experiment / 'prepared.json', {'variants': result, 'scoring_sha256': sha256(experiment / 'scoring.json')})
    print('Prepared three arms, no recovery:', experiment)


def prepared(experiment):
    info, previous = context(experiment)
    scoring(experiment, previous)
    ready = read_json(experiment / 'prepared.json')
    if set(ready['variants']) != set(VARIANTS) or ready['scoring_sha256'] != sha256(experiment / 'scoring.json'):
        raise ValueError('Prepared experiment changed')
    c = previous['baseline_config']
    for v, report in ready['variants'].items():
        model = experiment / v
        vc = dict(c, output=str(model))
        if (sha256(ply_path(model, c['iterations'])) != report['ply_sha256']
            or read_json(model / 'config.json') != vc
            or read_json(model / 'manifest.json')['config'] != vc
            or (model / 'cfg_args').read_text(encoding='utf-8') != repr(render_config(vc))):
            raise ValueError('Derived model/config changed')
        if v == 'contribution50' and sha256(model / 'kept_indices.npy') != report['indices_sha256']:
            raise ValueError('Selection indices changed')
    return info, previous, ready


def evaluate(experiment):
    info, previous, ready = prepared(experiment)
    from doctor import diagnose
    env = diagnose()
    if env['errors']:
        raise ValueError(str(env))
    write_json(experiment / 'evaluation_environment.json', env)
    write_json(experiment / 'evaluation.json', {'complete': False})
    c = previous['baseline_config']; count = previous['test_views']
    reference = Path(c['output']) / f'test/ours_{c["iterations"]}'
    rows, hashes = [], {}
    for v in VARIANTS:
        model = experiment / v; vc = dict(c, output=str(model))
        images = model / f'test/ours_{c["iterations"]}'
        execute(command('render', vc), model, 'render')
        if not compare_images(reference / 'gt', images / 'gt', count)['passed']:
            raise ValueError('GT mismatch')
        assert_image_sets(images / 'renders', count)
        if v != 'contribution50':
            prior = Path(info['parent']) / v / f'test/ours_{c["iterations"]}/renders'
            if not compare_images(prior, images / 'renders', count)['passed']:
                raise ValueError('Control image mismatch')
        for name in ('results.json', 'per_view.json', 'benchmark.json'):
            file = model / name
            if file.exists():
                file.rename(model / (name + '.' + dt.datetime.now().strftime('%Y%m%dT%H%M%S%f') + '.bak'))
        execute(command('metrics', vc), model, 'metrics')
        metrics = checked_metrics(model, c['iterations'], count)
        execute([sys.executable, str(ROOT / 'scripts/benchmark.py'), '--config', str(model / 'config.json')],
                model, 'benchmark')
        b = read_json(model / 'benchmark.json'); times = b['samples_ms']
        if (b['gaussian_count'] != ready['variants'][v]['count'] or b['test_views'] != count
            or b['repeats'] != 3 or len(times) != 3 * count
            or not all(math.isfinite(t) and t > 0 for t in times)
            or not math.isclose(sum(times) / len(times), b['mean_ms'], rel_tol=1e-6)):
            raise ValueError('Benchmark count/timing mismatch')
        rows.append({'variant': v, 'recovery_steps': 0, **metrics,
                     'gaussian_count': b['gaussian_count'], 'ply_bytes': ply_path(model, c['iterations']).stat().st_size,
                     'mean_ms': b['mean_ms'], 'fps_from_mean_ms': b['fps_from_mean_ms'],
                     'peak_torch_allocated_bytes': b['peak_torch_allocated_bytes'],
                     **{'delta_' + k: metrics[k] - previous['baseline_metrics'][k] for k in ('PSNR', 'SSIM', 'LPIPS')}})
        for file in [model / 'results.json', model / 'per_view.json', model / 'benchmark.json',
                     *images.glob('gt/*.png'), *images.glob('renders/*.png')]:
            hashes[file.relative_to(experiment).as_posix()] = sha256(file)
    write_json(experiment / 'evaluation.json', {'complete': True, 'rows': rows, 'artifact_sha256': hashes,
               'prepared_sha256': sha256(experiment / 'prepared.json')})
    print('Three-arm evaluation complete')


def export(experiment, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Unsafe export label')
    info, previous, ready = prepared(experiment)
    e = read_json(experiment / 'evaluation.json')
    if not e.get('complete') or e['prepared_sha256'] != sha256(experiment / 'prepared.json'):
        raise ValueError('Complete evaluation required')
    if any(sha256(experiment / f) != h for f, h in e['artifact_sha256'].items()):
        raise ValueError('Evaluated artifacts changed')
    c = previous['baseline_config']
    if [r['variant'] for r in e['rows']] != list(VARIANTS):
        raise ValueError('Invalid summary rows')
    for row in e['rows']:
        actual = checked_metrics(experiment / row['variant'], c['iterations'], previous['test_views'])
        if any(row[k] != actual[k] for k in actual):
            raise ValueError('Summary metrics differ from artifacts')
    dest = ROOT / 'results' / name
    if dest.exists():
        raise FileExistsError('Use a new export label')
    dest.mkdir(parents=True)
    write_json(dest / 'summary.json', e)
    with (dest / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(e['rows'][0])); writer.writeheader(); writer.writerows(e['rows'])
    public = {k: v for k, v in info.items() if k != 'parent'}
    public['baseline_config'] = {k: v for k, v in c.items() if k not in ('source', 'output')}
    public['variants'] = ready['variants']
    public['upstream'] = previous['upstream']
    names = [f'{i:05d}.png' for i in (0, 14, 24, 31, 33, 35) if i < previous['test_views']]
    public['selected_images'] = names
    public['selection_note'] = 'Same diagnostic views, not an unbiased gallery'
    env = read_json(experiment / 'evaluation_environment.json')
    public['environment'] = {k: env.get(k) for k in ('python', 'torch', 'torch_cuda', 'gpu', 'platform')}
    write_json(dest / 'protocol.json', public)
    shutil.copy2(experiment / 'scoring.json', dest / 'scoring.json')
    for v in VARIANTS:
        (dest / v / 'renders').mkdir(parents=True)
        shutil.copy2(experiment / v / 'benchmark.json', dest / v / 'benchmark.json')
        shutil.copy2(experiment / v / 'per_view.json', dest / v / 'metrics_per_view.json')
        for n in names:
            shutil.copy2(experiment / v / f'test/ours_{c["iterations"]}/renders' / n, dest / v / 'renders' / n)
    (dest / 'gt').mkdir()
    for n in names:
        shutil.copy2(experiment / 'identity100' / f'test/ours_{c["iterations"]}/gt' / n, dest / 'gt' / n)
    print('Exported:', dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['score', 'prepare', 'evaluate', 'export'])
    p.add_argument('--parent', default='outputs/compression/train_phase1_v2')
    p.add_argument('--experiment', default='outputs/compression/train_contribution50')
    p.add_argument('--views', type=int, default=0, help='score only: 0=all train views; 2=smoke')
    p.add_argument('--name')
    a = p.parse_args(); experiment = resolve(a.experiment)
    if a.stage == 'score':
        score(resolve(a.parent), experiment, a.views)
    elif a.stage == 'prepare':
        prepare(experiment)
    elif a.stage == 'evaluate':
        evaluate(experiment)
    else:
        if not a.name:
            p.error('export requires --name')
        export(experiment, a.name)


if __name__ == '__main__':
    main()
