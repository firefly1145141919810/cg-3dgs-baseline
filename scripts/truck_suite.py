"""Frozen cross-scene replication: truck baseline -> pruning -> six-arm recovery.

All GPU work runs in subprocesses using the existing, unchanged experiment code.
plan is read-only; all excludes export. Never automatically select a checkpoint.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from run import ROOT, resolve, load_config, write_json, verify_upstream
from dataset import validate, sha256

STAGES = ('baseline', 'opacity', 'contribution', 'recovery')


def paths(config):
    c = load_config(config)
    # Allow moving the dataset, not silently changing the frozen experiment.
    expected = {'scene': 'truck', 'iterations': 30000, 'resolution': -1,
                'images': 'images', 'white_background': False, 'data_device': 'cuda',
                'eval_iterations': [7000, 30000], 'checkpoint_iterations': [7000, 15000, 30000]}
    if any(c[k] != v for k, v in expected.items()):
        raise ValueError('Truck protocol is frozen; do not change training/resolution settings')
    p = {'baseline': Path(c['output']),
         'opacity': ROOT / 'outputs/compression/truck_phase1',
         'contribution': ROOT / 'outputs/compression/truck_contribution50',
         'recovery': ROOT / 'outputs/compression/truck_scoring_recovery_3k'}
    if p['baseline'] != ROOT / 'outputs/truck_baseline_30k':
        raise ValueError('Use the dedicated truck baseline output, not train outputs')
    for output in p.values():
        source = Path(c['source'])
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Dataset/output overlap')
    return c, p


def commands(config, stage, checkpoint=None):
    _, p = paths(config)
    cfg = str(resolve(config))
    def cmd(script, *args):
        return [sys.executable, str(ROOT / 'scripts' / script), *map(str, args)]
    if stage == 'baseline':
        train = cmd('run.py', 'train', '--config', cfg)
        if checkpoint:
            train += ['--resume', str(resolve(checkpoint))]
        return [train, cmd('run.py', 'render', '--config', cfg),
                cmd('run.py', 'metrics', '--config', cfg),
                cmd('benchmark.py', '--config', cfg)]
    if stage == 'opacity':
        return [cmd('compression_phase1.py', 'prepare', '--baseline-config', cfg, '--experiment', p[stage]),
                cmd('compression_phase1.py', 'evaluate', '--experiment', p[stage])]
    if stage == 'contribution':
        return [cmd('contribution_phase3.py', 'score', '--parent', p['opacity'], '--experiment', p[stage], '--views', 0),
                cmd('contribution_phase3.py', 'prepare', '--experiment', p[stage]),
                cmd('contribution_phase3.py', 'evaluate', '--experiment', p[stage])]
    if stage == 'recovery':
        return [cmd('recovery_phase4.py', 'prepare', '--parent', p['contribution'], '--experiment', p[stage],
                    '--steps', 3000, '--seed', 0),
                cmd('recovery_phase4.py', 'train', '--experiment', p[stage]),
                cmd('recovery_phase4.py', 'evaluate', '--experiment', p[stage])]
    raise ValueError('Unknown stage')


def run_command(cmd):
    print('RUN:', json.dumps(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def evaluation_marker(folder):
    path = folder / 'evaluation.json'
    return path.exists() and read(path).get('complete') is True


def baseline_check(config, require_results=True):
    c, p = paths(config)
    b = p['baseline']
    m = read(b / 'manifest.json')
    upstream = verify_upstream()
    if m['config'] != c or m['upstream']['commit'] != upstream['commit'] or m['seed'] != 0:
        raise ValueError('Existing baseline configuration/version/seed differs')
    data = validate(c['source'], c['images'])
    recorded = read(b / 'dataset.json')
    for key in ('train', 'test', 'camera_hashes', 'image_sha256'):
        if data[key] != recorded[key]:
            raise ValueError('Baseline data changed: ' + key)
    if not require_results:
        return
    from compression_phase1 import checked_metrics, assert_image_sets
    from export_results import ply_count
    checked_metrics(b, 30000, len(data['test']))
    for kind in ('gt', 'renders'):
        assert_image_sets(b / 'test/ours_30000' / kind, len(data['test']))
    bench = read(b / 'benchmark.json'); times = bench['samples_ms']
    if (bench['test_views'] != len(data['test']) or bench['repeats'] != 3
        or len(times) != len(data['test']) * 3
        or not all(math.isfinite(t) and t > 0 for t in times)
        or not math.isclose(sum(times) / len(times), bench['mean_ms'], rel_tol=1e-6)
        or bench['gaussian_count'] != ply_count(b / 'point_cloud/iteration_30000/point_cloud.ply')):
        raise ValueError('Invalid baseline timing/count')


def ready(config, stage):
    """Completed means verified artifacts, not merely an output directory."""
    c, p = paths(config)
    folder = p[stage]
    if stage == 'baseline':
        needed = ('manifest.json', 'dataset.json', 'results.json', 'per_view.json', 'benchmark.json',
                  'point_cloud/iteration_30000/point_cloud.ply')
        if not all((folder / f).exists() for f in needed):
            return False
        baseline_check(config)
        return True
    if not evaluation_marker(folder):
        return False
    if stage == 'opacity':
        from recovery_phase2 import verify_parent
        info = verify_parent(folder)
        if info['baseline_config'] != c:
            raise ValueError('Wrong scene/config in opacity experiment')
    elif stage == 'contribution':
        from recovery_phase4 import verify_parent
        info = verify_parent(folder)
        if info['baseline_config'] != c:
            raise ValueError('Wrong scene/config in contribution experiment')
    else:
        from recovery_phase4 import verify, check_finished, TRAIN_VARIANTS, VARIANTS, model_config
        from compression_phase1 import checked_metrics
        info = verify(folder)
        if info['baseline_config'] != c or Path(info['parent']) != p['contribution']:
            raise ValueError('Wrong scene/parent in recovery experiment')
        if info['steps'] != 3000 or info['seed'] != 0 or not info['formal']:
            raise ValueError('Wrong recovery budget/seed')
        reports = [check_finished(info, folder, v) for v in TRAIN_VARIANTS]
        if len({r['camera_sequence_sha256'] for r in reports}) != 1:
            raise ValueError('Recovery camera sequences differ')
        e = read(folder / 'evaluation.json')
        if (e['manifest_sha256'] != sha256(folder / 'experiment.json') or
            any(sha256(folder / f) != h for f, h in e['artifact_sha256'].items())):
            raise ValueError('Recovery artifacts changed')
        if [r['variant'] for r in e['rows']] != list(VARIANTS):
            raise ValueError('Wrong recovery summary rows')
        for row in e['rows']:
            c = model_config(info, folder, row['variant'])
            m = checked_metrics(folder / row['variant'], c['iterations'], info['test_views'])
            if any(row[k] != m[k] for k in m):
                raise ValueError('Recovery summary differs from metrics')
    return True


def run_stage(config, stage, checkpoint=None):
    c, p = paths(config)
    if ready(config, stage):
        print('Verified complete; skipping:', stage)
        return
    if any((p[s] / 'experiment.json').exists() for s in STAGES[STAGES.index(stage)+1:]):
        raise ValueError('Incomplete upstream stage has derived experiments; preserve them and diagnose before rerunning')
    if stage != 'baseline' and not ready(config, STAGES[STAGES.index(stage)-1]):
        raise ValueError('Finish the preceding stage first')
    cmds = commands(config, stage, checkpoint)
    folder = p[stage]
    if stage == 'baseline':
        manifest = folder / 'manifest.json'
        final = folder / 'point_cloud/iteration_30000/point_cloud.ply'
        if manifest.exists():
            baseline_check(config, require_results=False)
        if final.exists():
            if not manifest.exists():
                raise ValueError('Untracked final PLY; refusing to assume training provenance')
            cmds = cmds[1:]
        elif folder.exists() and any(folder.iterdir()) and not checkpoint:
            raise ValueError('Partial baseline: explicitly pass --checkpoint to a trusted local .pth; no auto-selection')
        for cmd in cmds:
            run_command(cmd)
    elif stage == 'opacity':
        if not (folder / 'experiment.json').exists():
            run_command(cmds[0])
        # evaluate validates the recorded inputs/code even when preparation is skipped.
        run_command(cmds[1])
    elif stage == 'contribution':
        run_command(cmds[0])  # scoring validates and resumes or skips itself.
        if not (folder / 'prepared.json').exists():
            run_command(cmds[1])
        run_command(cmds[2])
    else:
        if not (folder / 'experiment.json').exists():
            run_command(cmds[0])
        run_command(cmds[1])  # per-arm recovery resume/complete verification.
        run_command(cmds[2])
    if not ready(config, stage):
        raise ValueError('Stage returned without valid complete artifacts')
    print('Verified:', stage)


def export(config, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', name):
        raise ValueError('Unsafe export label')
    if not ready(config, 'recovery'):
        raise ValueError('Finish six-arm recovery before export')
    c, p = paths(config)
    import recovery_phase4
    def extra(dest):
        shutil.copy2(p['contribution'] / 'scoring.json', dest / 'scoring.json')
        data = read(p['baseline'] / 'dataset.json')
        write_json(dest / 'cross_scene.json', {
            'scene': 'truck', 'protocol': 'Frozen replication of train: 50% selection, 3000-step recovery, seed 0',
            'config': {k: v for k, v in c.items() if k not in ('source', 'output')},
            'train_views': len(data['train']), 'test_views': len(data['test']),
            'test_names': data['test'], 'dataset_camera_hashes': data['camera_hashes'],
            'upstream': read(p['baseline'] / 'manifest.json')['upstream'],
            'baseline_manifest_sha256': sha256(p['baseline'] / 'manifest.json'),
            'baseline_benchmark': read(p['baseline'] / 'benchmark.json'),
            'contribution_selection': read(p['contribution'] / 'prepared.json')['variants'],
            'scoring_sha256': sha256(dest / 'scoring.json'),
            'suite_script_sha256': sha256(Path(__file__)),
            'warning': 'Within-scene comparisons only; resolutions/view counts can differ from train'})
    recovery_phase4.export(p['recovery'], name, extra=extra)
    dest = ROOT / 'results' / name
    print('Export complete. Upload this results directory; no PLY/checkpoint included:', dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'check', *STAGES, 'all', 'export'])
    parser.add_argument('--config', default='configs/truck.json')
    parser.add_argument('--checkpoint', help='baseline/all only; trusted local baseline checkpoint, not a recovery PLY')
    parser.add_argument('--name', default='truck-scoring-recovery3k-first')
    a = parser.parse_args()
    if a.checkpoint and a.stage not in ('baseline', 'all', 'plan'):
        parser.error('--checkpoint is only for baseline/all/plan')
    c, p = paths(a.config)
    if a.stage == 'plan':
        print(json.dumps({s: commands(a.config, s, a.checkpoint) for s in STAGES}, indent=2))
    elif a.stage == 'check':
        report = validate(c['source'], c['images'])
        print(json.dumps({'source': c['source'], 'train_views': len(report['train']),
                          'test_views': len(report['test']), 'split': report['split'],
                          'outputs': {k: str(v) for k, v in p.items()}}, indent=2))
    elif a.stage == 'export':
        export(a.config, a.name)
    else:
        for stage in STAGES if a.stage == 'all' else (a.stage,):
            run_stage(a.config, stage, a.checkpoint if stage == 'baseline' else None)


if __name__ == '__main__':
    main()
