"""Phase 1 only: exact SH3 PLY roundtrip and opacity top-50% pruning.

prepare: CPU; evaluate: cloud CUDA, no training; export: CPU.
Uses the original render/metrics code without modifying upstream.
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
import numpy as np
from PIL import Image
from dataset import sha256, validate
from prune_ply import write_pruned
from run import ROOT, UPSTREAM, command, execute, load_config, resolve, verify_upstream, write_json

VARIANTS = [('identity100', 1.0), ('opacity50', 0.5)]


def code_hashes():
    return {name: sha256(ROOT / 'scripts' / name) for name in
            ('compression_phase1.py', 'prune_ply.py', 'run.py', 'benchmark.py', 'dataset.py', 'doctor.py')}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assert_image_sets(folder, count):
    expected = {f'{i:05d}.png' for i in range(count)}
    actual = {p.name for p in Path(folder).glob('*.png')}
    if actual != expected:
        raise ValueError(f'Incomplete/unexpected image set: {folder}')


def compare_images(reference, candidate, count):
    for folder in (reference, candidate):
        assert_image_sets(folder, count)
    rows = []
    total_error = total_values = differing = 0
    for i in range(count):
        name = f'{i:05d}.png'
        with Image.open(Path(reference) / name) as a, Image.open(Path(candidate) / name) as b:
            if a.mode != 'RGB' or b.mode != 'RGB' or a.size != b.size:
                raise ValueError('Image size/mode mismatch: ' + name)
            delta = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16))
        changed = int(np.count_nonzero(delta))
        rows.append({'image': name, 'max_abs_8bit': int(delta.max()),
                     'mean_abs_8bit': float(delta.mean()), 'changed_channels': changed})
        total_error += int(delta.sum())
        total_values += delta.size
        differing += changed
    return {'passed': differing == 0, 'comparison': 'exact decoded RGB PNG pixels, not float render tensors',
            'max_abs_8bit': max(r['max_abs_8bit'] for r in rows),
            'mean_abs_8bit': total_error / total_values,
            'changed_channels': differing, 'per_view': rows}


def checked_metrics(model, iteration, count):
    key = f'ours_{iteration}'
    values = read_json(model / 'results.json')[key]
    per_view = read_json(model / 'per_view.json')[key]
    expected = {f'{i:05d}.png' for i in range(count)}
    for metric in ('PSNR', 'SSIM', 'LPIPS'):
        v = values[metric]
        views = per_view[metric]
        if not math.isfinite(v) or set(views) != expected or not all(math.isfinite(x) for x in views.values()):
            raise ValueError('Invalid metrics: ' + metric)
        if not math.isclose(sum(views.values()) / count, v, rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError('Per-view mean does not match summary: ' + metric)
    return values


def prepare(config, experiment):
    c = load_config(config)
    parent = Path(c['output'])
    if experiment == parent or experiment.is_relative_to(parent) or parent.is_relative_to(experiment):
        raise ValueError('Experiment and baseline directories must be disjoint')
    if experiment.exists():
        raise FileExistsError('Experiment already exists. Reuse evaluate/export or choose a new --experiment.')
    upstream = verify_upstream()
    manifest = read_json(parent / 'manifest.json')
    if manifest['config'] != c or manifest['upstream']['commit'] != upstream['commit']:
        raise ValueError('Baseline configuration/version mismatch')
    data = validate(c['source'], c['images'])
    recorded = read_json(parent / 'dataset.json')
    for key in ('train', 'test', 'camera_hashes', 'image_sha256'):
        if recorded[key] != data[key]:
            raise ValueError('Baseline dataset changed: ' + key)
    count = len(data['test'])
    checked_metrics(parent, c['iterations'], count)
    source = parent / f'point_cloud/iteration_{c["iterations"]}/point_cloud.ply'
    source_hash = sha256(source)
    image_root = parent / f'test/ours_{c["iterations"]}'
    image_hashes = {}
    for kind in ('renders', 'gt'):
        assert_image_sets(image_root / kind, count)
        image_hashes[kind] = {p.name: sha256(p) for p in sorted((image_root / kind).glob('*.png'))}
    experiment.mkdir(parents=True)
    info = {'schema': 1, 'phase': 'opacity pruning without recovery',
            'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'baseline_config': c, 'baseline_ply_sha256': source_hash,
            'baseline_metrics': checked_metrics(parent, c['iterations'], count),
            'baseline_image_sha256': image_hashes, 'test_views': count,
            'upstream': upstream, 'baseline_wrapper_commit': manifest['wrapper_commit'],
            'experiment_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'script_sha256': code_hashes(),
            'variants': {}}
    write_json(experiment / 'dataset.json', data)
    for name, fraction in VARIANTS:
        model = experiment / name
        target = model / f'point_cloud/iteration_{c["iterations"]}/point_cloud.ply'
        indices, report = write_pruned(source, target, fraction)
        np.save(model / 'kept_indices.npy', indices, allow_pickle=False)
        vc = dict(c, output=str(model))
        write_json(model / 'config.json', vc)
        write_json(model / 'manifest.json', {'config': vc, 'derived_experiment': str(experiment),
                   'note': 'iteration directory denotes source baseline, not new training'})
        # Original renderer requires this file and evals it: generate trusted empty Namespace;
        # every dataset setting is supplied explicitly on its command line.
        (model / 'cfg_args').write_text('Namespace()', encoding='utf-8')
        report['ply_sha256'] = sha256(target)
        report['indices_sha256'] = sha256(model / 'kept_indices.npy')
        info['variants'][name] = report
    if sha256(source) != source_hash:
        raise ValueError('Baseline model changed during preparation')
    write_json(experiment / 'experiment.json', info)
    print('Prepared two variants. No model training performed:', experiment)


def verify_inputs(experiment):
    info = read_json(experiment / 'experiment.json')
    if info.get('schema') != 1 or set(info['variants']) != {n for n, _ in VARIANTS}:
        raise ValueError('Unexpected experiment schema/variants')
    if info['script_sha256'] != code_hashes():
        raise ValueError('Experiment scripts changed; use the recorded code version or a new experiment')
    c = info['baseline_config']
    verify_upstream()
    parent = Path(c['output'])
    source = parent / f'point_cloud/iteration_{c["iterations"]}/point_cloud.ply'
    if sha256(source) != info['baseline_ply_sha256']:
        raise ValueError('Baseline PLY changed')
    data = validate(c['source'], c['images'])
    recorded = read_json(experiment / 'dataset.json')
    for key in ('train', 'test', 'camera_hashes', 'image_sha256'):
        if data[key] != recorded[key]:
            raise ValueError('Dataset changed: ' + key)
    for kind, hashes in info['baseline_image_sha256'].items():
        folder = parent / f'test/ours_{c["iterations"]}' / kind
        assert_image_sets(folder, info['test_views'])
        if any(sha256(folder / name) != digest for name, digest in hashes.items()):
            raise ValueError('Baseline reference images changed')
    if checked_metrics(parent, c['iterations'], info['test_views']) != info['baseline_metrics']:
        raise ValueError('Baseline reference metrics changed')
    for name, _ in VARIANTS:
        model = experiment / name
        target = model / f'point_cloud/iteration_{c["iterations"]}/point_cloud.ply'
        r = info['variants'][name]
        if sha256(target) != r['ply_sha256'] or sha256(model / 'kept_indices.npy') != r['indices_sha256']:
            raise ValueError('Derived model or selection indices changed')
        if read_json(model / 'config.json') != dict(c, output=str(model)):
            raise ValueError('Variant configuration changed')
    return info


def evaluate(experiment):
    info = verify_inputs(experiment)
    from doctor import diagnose
    diagnostic = diagnose()
    if diagnostic['errors']:
        raise ValueError(json.dumps(diagnostic, indent=2))
    write_json(experiment / 'evaluation_environment.json', diagnostic)
    c = info['baseline_config']
    count = info['test_views']
    reference = Path(c['output']) / f'test/ours_{c["iterations"]}'
    # Invalidate a previous success marker before rerunning any evaluation.
    write_json(experiment / 'evaluation.json', {'complete': False})
    rows = []
    artifact_hashes = {}
    for name, _ in VARIANTS:
        model = experiment / name
        vc = dict(c, output=str(model))
        execute(command('render', vc), model, 'render')
        images = model / f'test/ours_{c["iterations"]}'
        gt_check = compare_images(reference / 'gt', images / 'gt', count)
        if not gt_check['passed']:
            raise ValueError('Ground truth differs from baseline; do not compare metrics')
        if name == 'identity100':
            check = compare_images(reference / 'renders', images / 'renders', count)
            write_json(experiment / 'identity_check.json', check)
            if not check['passed']:
                raise ValueError('100% roundtrip PNG check failed; see identity_check.json. Opacity50 evaluation blocked.')
        assert_image_sets(images / 'renders', count)
        # Never allow an old result to hide upstream's broad exception handler.
        for filename in ('results.json', 'per_view.json'):
            prior = model / filename
            if prior.exists():
                stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                prior.rename(model / (filename + '.' + stamp + '.bak'))
        execute(command('metrics', vc), model, 'metrics')
        metrics = checked_metrics(model, c['iterations'], count)
        execute([sys.executable, str(ROOT / 'scripts/benchmark.py'),
                 '--config', str(model / 'config.json')], model, 'benchmark')
        benchmark = read_json(model / 'benchmark.json')
        report = info['variants'][name]
        if benchmark['gaussian_count'] != report['kept_count']:
            raise ValueError('Renderer loaded the wrong Gaussian count')
        times = benchmark['samples_ms']
        if (benchmark['test_views'] != count or len(times) != count * benchmark['repeats'] or
            not all(math.isfinite(t) and t > 0 for t in times) or
            not math.isclose(sum(times) / len(times), benchmark['mean_ms'], rel_tol=1e-6)):
            raise ValueError('Invalid benchmark samples')
        for file in [model / 'results.json', model / 'per_view.json', model / 'benchmark.json',
                     *sorted((images / 'renders').glob('*.png')), *sorted((images / 'gt').glob('*.png'))]:
            artifact_hashes[file.relative_to(experiment).as_posix()] = sha256(file)
        rows.append({'variant': name, 'recovery_steps': 0, **metrics,
                     'gaussian_count': report['kept_count'], 'ply_bytes': report['output_ply_bytes'],
                     'mean_ms': benchmark['mean_ms'], 'fps_from_mean_ms': benchmark['fps_from_mean_ms'],
                     'peak_torch_allocated_bytes': benchmark['peak_torch_allocated_bytes'],
                     'delta_PSNR': metrics['PSNR'] - info['baseline_metrics']['PSNR'],
                     'delta_SSIM': metrics['SSIM'] - info['baseline_metrics']['SSIM'],
                     'delta_LPIPS': metrics['LPIPS'] - info['baseline_metrics']['LPIPS']})
    write_json(experiment / 'evaluation.json', {'complete': True, 'rows': rows,
               'artifact_sha256': artifact_hashes,
               'experiment_manifest_sha256': sha256(experiment / 'experiment.json'),
               'note': 'identity100 is a fresh same-session unpruned timing control; no training in either variant'})
    print('Phase 1 complete. Inspect and export results. This is NOT importance pruning or recovery training.')


def export(experiment, label):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', label):
        raise ValueError('Unsafe export name')
    info = verify_inputs(experiment)
    evaluation = read_json(experiment / 'evaluation.json')
    if (not evaluation.get('complete') or not read_json(experiment / 'identity_check.json')['passed'] or
        evaluation['experiment_manifest_sha256'] != sha256(experiment / 'experiment.json')):
        raise ValueError('Only a fully evaluated experiment with passed identity gate can be exported')
    if any(sha256(experiment / name) != digest for name, digest in evaluation['artifact_sha256'].items()):
        raise ValueError('Evaluation images, metrics or benchmark changed after evaluation')
    dest = ROOT / 'results' / label
    if dest.exists():
        raise FileExistsError('Export exists; use a new label')
    c = info['baseline_config']
    # Fixed diagnostic sample, not cherry-picked anew for each method.
    selected_names = [f'{i:05d}.png' for i in [0, 14, 24, 31, 33, 35] if i < info['test_views']]
    for name, _ in VARIANTS:
        model = experiment / name
        images = model / f'test/ours_{c["iterations"]}'
        metrics = checked_metrics(model, c['iterations'], info['test_views'])
        row = next(r for r in evaluation['rows'] if r['variant'] == name)
        if any(metrics[k] != row[k] for k in ('PSNR', 'SSIM', 'LPIPS')):
            raise ValueError('Metrics changed after evaluation')
        assert_image_sets(images / 'renders', info['test_views'])
    dest.mkdir(parents=True)
    write_json(dest / 'summary.json', evaluation)
    with (dest / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(evaluation['rows'][0]))
        writer.writeheader()
        writer.writerows(evaluation['rows'])
    public = dict(info)
    public['baseline_config'] = {k: v for k, v in c.items() if k not in ('source', 'output')}
    public['selected_images'] = selected_names
    public['selection_note'] = 'Fixed diagnostic views, including previously observed failures; not an unbiased gallery'
    env = read_json(experiment / 'evaluation_environment.json')
    public['evaluation_environment'] = {k: env.get(k) for k in
        ('python', 'platform', 'torch', 'torch_cuda', 'gpu', 'compute_capability', 'nvidia-smi', 'nvcc')}
    write_json(dest / 'protocol.json', public)
    shutil.copy2(experiment / 'identity_check.json', dest / 'identity_check.json')
    for name, _ in VARIANTS:
        model = experiment / name
        (dest / name / 'renders').mkdir(parents=True)
        shutil.copy2(model / 'benchmark.json', dest / name / 'benchmark.json')
        shutil.copy2(model / 'per_view.json', dest / name / 'metrics_per_view.json')
        for image in selected_names:
            shutil.copy2(model / f'test/ours_{c["iterations"]}/renders' / image, dest / name / 'renders' / image)
    (dest / 'gt').mkdir()
    for image in selected_names:
        shutil.copy2(Path(c['output']) / f'test/ours_{c["iterations"]}/gt' / image, dest / 'gt' / image)
    print('Exported metrics AND paired images (no model/checkpoint):', dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'evaluate', 'export'])
    parser.add_argument('--baseline-config', default='configs/train.json')
    parser.add_argument('--experiment', default='outputs/compression/train_phase1')
    parser.add_argument('--name', help='Unique result export directory name')
    args = parser.parse_args()
    experiment = resolve(args.experiment)
    if args.stage == 'prepare':
        prepare(args.baseline_config, experiment)
    elif args.stage == 'evaluate':
        evaluate(experiment)
    else:
        if not args.name:
            parser.error('export requires --name')
        export(experiment, args.name)


if __name__ == '__main__':
    main()
