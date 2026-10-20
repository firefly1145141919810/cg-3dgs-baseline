"""Fixed-count recovery: prepare, train, evaluate four arms, export small results."""
import argparse
import csv
import datetime as dt
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from run import ROOT, command, execute, resolve, write_json
from dataset import sha256
import compression_phase1 as phase1
from compression_phase1 import read_json, render_config, checked_metrics, compare_images, assert_image_sets
from recovery_train import optimization, check_plan

VARIANTS = ('identity100', 'opacity50', 'identity100_recovery', 'opacity50_recovery')
TRAIN_VARIANTS = VARIANTS[2:]


def code_hashes():
    return {**phase1.code_hashes(), **{n: sha256(ROOT / 'scripts' / n)
            for n in ('recovery_phase2.py', 'recovery_train.py')}}


def ply_path(model, iteration):
    return model / f'point_cloud/iteration_{iteration}/point_cloud.ply'


def model_config(info, experiment, variant):
    c = dict(info['baseline_config'], output=str(experiment / variant))
    c['iterations'] = info['source_iteration'] + (info['steps'] if variant in TRAIN_VARIANTS else 0)
    c['eval_iterations'] = c['checkpoint_iterations'] = [c['iterations']]
    return c


def verify_parent(parent):
    info = phase1.verify_inputs(parent)
    evaluation = read_json(parent / 'evaluation.json')
    check = read_json(parent / 'identity_check.json')
    if not evaluation.get('complete') or not check['passed']:
        raise ValueError('Phase 1 must have passed evaluation and identity gate')
    if evaluation['experiment_manifest_sha256'] != sha256(parent / 'experiment.json'):
        raise ValueError('Parent manifest changed')
    if any(sha256(parent / f) != h for f, h in evaluation['artifact_sha256'].items()):
        raise ValueError('Parent evaluation artifacts changed')
    return info


def prepare(parent, experiment, steps, seed):
    check_plan(steps, seed)
    info = verify_parent(parent)
    for protected in (parent, Path(info['baseline_config']['output']), Path(info['baseline_config']['source'])):
        if experiment == protected or experiment.is_relative_to(protected) or protected.is_relative_to(experiment):
            raise ValueError('Experiment must be disjoint from source directories')
    if experiment.exists():
        raise FileExistsError('Use a new experiment directory; never overwrite old results')
    result = {'schema': 2, 'phase': 'fixed-count recovery', 'steps': steps, 'seed': seed,
              'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'formal': steps == 3000, 'parent': str(parent),
              'parent_manifest_sha256': sha256(parent / 'experiment.json'),
              'parent_evaluation_sha256': sha256(parent / 'evaluation.json'),
              'baseline_config': info['baseline_config'], 'source_iteration': info['baseline_config']['iterations'],
              'test_views': info['test_views'], 'train_views': len(read_json(parent / 'dataset.json')['train']),
              'optimization': vars(optimization(steps)), 'script_sha256': code_hashes(),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'variants': {}}
    experiment.mkdir(parents=True)
    for variant in VARIANTS:
        source_name = variant.removesuffix('_recovery')
        source = ply_path(parent / source_name, result['source_iteration'])
        target = ply_path(experiment / variant, result['source_iteration'])
        target.parent.mkdir(parents=True)
        shutil.copy2(source, target)
        digest = sha256(source)
        if sha256(target) != digest:
            raise ValueError('Model copy differs from source')
        result['variants'][variant] = {'source_sha256': digest,
            'count': info['variants'][source_name]['kept_count']}
        c = model_config(result, experiment, variant)
        write_json(experiment / variant / 'config.json', c)
        write_json(experiment / variant / 'manifest.json', {'config': c})
        (experiment / variant / 'cfg_args').write_text(repr(render_config(c)), encoding='utf-8')
    write_json(experiment / 'experiment.json', result)
    print('Prepared four arms; formal:', result['formal'], experiment)


def verify(experiment):
    info = read_json(experiment / 'experiment.json')
    check_plan(info['steps'], info['seed'])
    if info['schema'] != 2 or set(info['variants']) != set(VARIANTS) or info['script_sha256'] != code_hashes():
        raise ValueError('Experiment schema/code changed; use a new experiment')
    parent = Path(info['parent'])
    verify_parent(parent)
    if (sha256(parent / 'experiment.json') != info['parent_manifest_sha256'] or
        sha256(parent / 'evaluation.json') != info['parent_evaluation_sha256']):
        raise ValueError('Parent experiment changed')
    for v in VARIANTS:
        model = experiment / v
        if sha256(ply_path(model, info['source_iteration'])) != info['variants'][v]['source_sha256']:
            raise ValueError('Initial PLY changed')
        c = model_config(info, experiment, v)
        if read_json(model / 'config.json') != c or read_json(model / 'manifest.json')['config'] != c:
            raise ValueError('Configuration changed')
        if (model / 'cfg_args').read_text(encoding='utf-8') != repr(render_config(c)):
            raise ValueError('Render configuration changed')
    return info


def check_finished(info, experiment, v):
    r = read_json(experiment / v / 'training.json')
    c = model_config(info, experiment, v)
    if (not r['complete'] or r['steps'] != info['steps'] or r['count'] != info['variants'][v]['count']
        or r['manifest_sha256'] != sha256(experiment / 'experiment.json')
        or r['source_ply_sha256'] != info['variants'][v]['source_sha256']
        or r['final_ply_sha256'] != sha256(ply_path(experiment / v, c['iterations']))
        or len(r['trace']) != info['steps']
        or any(t['step'] != i + 1 or not math.isfinite(t['loss']) for i, t in enumerate(r['trace']))):
        raise ValueError('Training incomplete or modified: ' + v)
    return r


def train(experiment):
    verify(experiment)
    for v in TRAIN_VARIANTS:
        execute([sys.executable, str(ROOT / 'scripts/recovery_train.py'),
                 '--experiment', str(experiment), '--variant', v], experiment / v, 'recovery')


def evaluate(experiment):
    info = verify(experiment)
    reports = [check_finished(info, experiment, v) for v in TRAIN_VARIANTS]
    if reports[0]['camera_sequence_sha256'] != reports[1]['camera_sequence_sha256']:
        raise ValueError('Training arms used different camera sequences')
    from doctor import diagnose
    env = diagnose()
    if env['errors']:
        raise ValueError(str(env))
    write_json(experiment / 'evaluation_environment.json', env)
    write_json(experiment / 'evaluation.json', {'complete': False})
    rows, hashes = [], {}
    c0 = info['baseline_config']
    reference = Path(c0['output']) / f'test/ours_{info["source_iteration"]}'
    for v in VARIANTS:
        model = experiment / v
        c = model_config(info, experiment, v)
        images = model / f'test/ours_{c["iterations"]}'
        execute(command('render', c), model, 'render')
        if not compare_images(reference / 'gt', images / 'gt', info['test_views'])['passed']:
            raise ValueError('GT mismatch')
        assert_image_sets(images / 'renders', info['test_views'])
        if v not in TRAIN_VARIANTS:
            previous = Path(info['parent']) / v / f'test/ours_{info["source_iteration"]}/renders'
            if not compare_images(previous, images / 'renders', info['test_views'])['passed']:
                raise ValueError('Untrained control changed: ' + v)
        for name in ('results.json', 'per_view.json', 'benchmark.json'):
            old = model / name
            if old.exists():
                old.rename(model / (name + '.' + dt.datetime.now().strftime('%Y%m%dT%H%M%S%f') + '.bak'))
        execute(command('metrics', c), model, 'metrics')
        metrics = checked_metrics(model, c['iterations'], info['test_views'])
        execute([sys.executable, str(ROOT / 'scripts/benchmark.py'), '--config', str(model / 'config.json')],
                model, 'benchmark')
        b = read_json(model / 'benchmark.json')
        times = b['samples_ms']
        if (b['gaussian_count'] != info['variants'][v]['count'] or b['test_views'] != info['test_views']
            or len(times) != info['test_views'] * 3 or b['repeats'] != 3
            or not all(math.isfinite(t) and t > 0 for t in times)
            or not math.isclose(sum(times) / len(times), b['mean_ms'], rel_tol=1e-6)):
            raise ValueError('Invalid timing/count')
        final_ply = ply_path(model, c['iterations'])
        rows.append({'variant': v, 'recovery_steps': info['steps'] if v in TRAIN_VARIANTS else 0,
                     **metrics, 'gaussian_count': b['gaussian_count'], 'ply_bytes': final_ply.stat().st_size,
                     'mean_ms': b['mean_ms'], 'fps_from_mean_ms': b['fps_from_mean_ms'],
                     'peak_torch_allocated_bytes': b['peak_torch_allocated_bytes'],
                     **{'delta_' + k: metrics[k] - read_json(Path(info['parent']) / 'experiment.json')['baseline_metrics'][k]
                        for k in ('PSNR', 'SSIM', 'LPIPS')}})
        files = [final_ply, model / 'results.json', model / 'per_view.json', model / 'benchmark.json',
                 *images.glob('renders/*.png'), *images.glob('gt/*.png')]
        if v in TRAIN_VARIANTS:
            files += [model / 'training.json', model / 'training_environment.json']
        for file in files:
            hashes[file.relative_to(experiment).as_posix()] = sha256(file)
    write_json(experiment / 'evaluation.json', {'complete': True, 'rows': rows,
        'artifact_sha256': hashes, 'manifest_sha256': sha256(experiment / 'experiment.json'),
        'note': 'All four arms freshly timed; PNG/disk I/O excluded; no claim of deterministic CUDA training'})
    print('Four-arm evaluation complete; formal:', info['formal'])


def export(experiment, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Unsafe export name')
    info = verify(experiment)
    e = read_json(experiment / 'evaluation.json')
    if not e.get('complete') or e['manifest_sha256'] != sha256(experiment / 'experiment.json'):
        raise ValueError('Complete evaluation required')
    if any(sha256(experiment / f) != h for f, h in e['artifact_sha256'].items()):
        raise ValueError('Evaluated artifact changed')
    if [r['variant'] for r in e['rows']] != list(VARIANTS):
        raise ValueError('Missing/duplicate evaluation rows')
    for row in e['rows']:
        v = row['variant']
        c = model_config(info, experiment, v)
        metrics = checked_metrics(experiment / v, c['iterations'], info['test_views'])
        if any(row[k] != metrics[k] for k in ('PSNR', 'SSIM', 'LPIPS')):
            raise ValueError('Summary disagrees with evaluated metrics')
        if v in TRAIN_VARIANTS:
            check_finished(info, experiment, v)
    dest = ROOT / 'results' / name
    if dest.exists():
        raise FileExistsError('Use a new result label')
    dest.mkdir(parents=True)
    write_json(dest / 'summary.json', e)
    with (dest / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(e['rows'][0]))
        writer.writeheader(); writer.writerows(e['rows'])
    public = {k: v for k, v in info.items() if k not in ('parent', 'baseline_config')}
    public['baseline_config'] = {k: v for k, v in info['baseline_config'].items() if k not in ('source', 'output')}
    selected = [f'{i:05d}.png' for i in (0, 14, 24, 31, 33, 35) if i < info['test_views']]
    public['selected_images'] = selected
    public['selection_note'] = 'Same fixed diagnostic views as Phase 1; not an unbiased gallery'
    env = read_json(experiment / 'evaluation_environment.json')
    public['environment'] = {k: env.get(k) for k in ('python', 'torch', 'torch_cuda', 'gpu', 'platform')}
    write_json(dest / 'protocol.json', public)
    for v in VARIANTS:
        model = experiment / v
        (dest / v / 'renders').mkdir(parents=True)
        for src, target in [('benchmark.json', 'benchmark.json'), ('per_view.json', 'metrics_per_view.json')]:
            shutil.copy2(model / src, dest / v / target)
        if v in TRAIN_VARIANTS:
            shutil.copy2(model / 'training.json', dest / v / 'training.json')
        iteration = model_config(info, experiment, v)['iterations']
        for n in selected:
            shutil.copy2(model / f'test/ours_{iteration}/renders' / n, dest / v / 'renders' / n)
    (dest / 'gt').mkdir()
    for n in selected:
        shutil.copy2(experiment / 'identity100' / f'test/ours_{info["source_iteration"]}/gt' / n, dest / 'gt' / n)
    print('Exported small results, no PLY/checkpoint:', dest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'train', 'evaluate', 'export'])
    p.add_argument('--parent', default='outputs/compression/train_phase1_v2')
    p.add_argument('--experiment', default='outputs/compression/train_recovery_3k')
    p.add_argument('--steps', type=int, default=3000, help='prepare only; use 20 for smoke test')
    p.add_argument('--seed', type=int, default=0, help='prepare only')
    p.add_argument('--name')
    a = p.parse_args()
    experiment = resolve(a.experiment)
    if a.stage == 'prepare':
        prepare(resolve(a.parent), experiment, a.steps, a.seed)
    elif a.stage == 'train':
        train(experiment)
    elif a.stage == 'evaluate':
        evaluate(experiment)
    else:
        if not a.name:
            p.error('export requires --name')
        export(experiment, a.name)


if __name__ == '__main__':
    main()
