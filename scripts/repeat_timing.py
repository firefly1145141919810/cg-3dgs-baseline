"""Phase 6: independent-process timing of frozen recovered models; no training."""
import argparse
import math
from pathlib import Path
import re
import statistics
import sys
from artifact_io import recover_json, export_directory
from dataset import sha256
from run import ROOT, execute, write_json
from compression_phase1 import read_json
import recovery_phase4 as phase4
import budget_phase5 as phase5

METHODS = ('identity100_recovery', 'opacity_recovery', 'contribution_recovery')


def order(round_index):
    offset = round_index % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def check_sample(b, count, views):
    times = b['samples_ms']
    if (b['gaussian_count'] != count or b['test_views'] != views or
        b['warmup'] != 20 or b['repeats'] != 3 or len(times) != views * 3 or
        not all(math.isfinite(t) and t > 0 for t in times) or
        not math.isclose(statistics.mean(times), b['mean_ms'], rel_tol=1e-7) or
        not math.isclose(1000 / b['mean_ms'], b['fps_from_mean_ms'], rel_tol=1e-7)):
        raise ValueError('Invalid timing/count/protocol')


def describe(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def collect_sources():
    sources = []
    for scene in ('train', 'truck'):
        for budget in (70, 50, 30):
            module = phase4 if budget == 50 else phase5
            suffix = f'{scene}_scoring_recovery_3k' if budget == 50 else f'{scene}_budget{budget}_recovery3k'
            exp = ROOT / 'outputs/compression' / suffix
            info = module.verify(exp)
            e = read_json(exp / 'evaluation.json')
            if not info['formal'] or info['steps'] != 3000 or not e.get('complete'):
                raise ValueError('Formal completed source required: ' + str(exp))
            if e['manifest_sha256'] != sha256(exp / 'experiment.json'):
                raise ValueError('Source manifest changed')
            for file, digest in e['artifact_sha256'].items():
                if sha256(exp / file) != digest:
                    raise ValueError('Evaluated source changed: ' + file)
            models = {}
            for method in METHODS:
                variant = method.replace('opacity_', 'opacity50_').replace('contribution_', 'contribution50_') if budget == 50 else method
                module.check_finished(info, exp, variant)
                config = exp / variant / 'config.json'
                ply = module.ply_path(exp / variant, info['source_iteration'] + info['steps'])
                models[method] = {'config': str(config), 'config_sha256': sha256(config),
                                  'ply': str(ply), 'ply_sha256': sha256(ply),
                                  'count': info['variants'][variant]['count']}
            sources.append({'scene': scene, 'budget': budget, 'views': info['test_views'],
                            'evaluation_sha256': sha256(exp / 'evaluation.json'), 'models': models})
    return sources


def summarize(manifest, dest):
    rows, hashes, environments = [], {}, set()
    for source in manifest['sources']:
        samples = {m: [] for m in METHODS}
        for r in range(manifest['rounds']):
            for method in METHODS:
                name = f'{source["scene"]}-{source["budget"]}-r{r + 1}-{method}.json'
                b = read_json(dest / name)
                check_sample(b, source['models'][method]['count'], source['views'])
                environments.add((b['gpu'], tuple(map(tuple, b['image_sizes'])), source['scene']))
                samples[method].append(b)
                hashes[name] = sha256(dest / name)
        for method in METHODS:
            times = [b['mean_ms'] for b in samples[method]]
            ratios = [samples[METHODS[0]][r]['mean_ms'] / times[r] for r in range(manifest['rounds'])]
            rows.append({'scene': source['scene'], 'budget': source['budget'], 'method': method,
                         'mean_ms_across_rounds': describe(times),
                         'fps_across_rounds': describe([1000 / t for t in times]),
                         'paired_speedup_across_rounds': describe(ratios),
                         'round_mean_ms': times,
                         'peak_allocated_bytes': describe([b['peak_torch_allocated_bytes'] for b in samples[method]])})
    if len({g for g, _, _ in environments}) != 1:
        raise ValueError('GPU model differs between samples; do not pool them')
    for scene in ('train', 'truck'):
        if len({size for _, size, s in environments if s == scene}) != 1:
            raise ValueError('Image sizes differ within scene')
    return {'complete': True, 'rows': rows, 'sample_sha256': hashes,
            'manifest_sha256': sha256(dest / 'protocol.json'),
            'note': 'Three rotations are process-level repeat measurements, not independent training seeds. '
                    'Report range, not statistical significance; CUDA-event renderer timing excludes I/O.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--rounds', type=int, default=3, choices=[3, 6])
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', a.name):
        p.error('Unsafe result label')
    from doctor import diagnose
    env = diagnose()
    if env['errors']:
        raise ValueError(str(env))
    sources = collect_sources()
    manifest = {'schema': 1, 'rounds': a.rounds, 'sources': sources,
                'script_sha256': {n: sha256(ROOT / 'scripts' / n) for n in
                                  ('repeat_timing.py', 'benchmark_repeat_worker.py', 'artifact_io.py', 'run.py', 'result_validation.py')},
                'order': [list(order(r)) for r in range(a.rounds)], 'warmup': 20, 'repeats': 3,
                'environment': {k: env.get(k) for k in ('python', 'torch', 'torch_cuda', 'gpu', 'platform')}}
    dest = ROOT / 'results' / a.name
    if dest.exists():
        if read_json(dest / 'protocol.json') != manifest:
            raise ValueError('Resume protocol/source/environment differs; use new label')
    else:
        with export_directory(dest) as staging:
            write_json(staging / 'protocol.json', manifest)
    if (dest / 'summary.json').exists():
        previous = recover_json(dest / 'summary.json')
        if previous is not None and previous != summarize(manifest, dest):
            raise ValueError('Completed summary/samples changed')
        if previous is not None:
            print('Verified completed timing; no new measurements:', dest)
            return
    # A resumed run revalidates every existing sample; successful samples are not overwritten.
    for source in sources:
        for r in range(a.rounds):
            for method in order(r):
                model = source['models'][method]
                name = f'{source["scene"]}-{source["budget"]}-r{r + 1}-{method}.json'
                target = dest / name
                existing = recover_json(target)
                if existing is None:
                    if sha256(Path(model['config'])) != model['config_sha256'] or sha256(Path(model['ply'])) != model['ply_sha256']:
                        raise ValueError('Source changed during timing')
                    log_dir = ROOT / 'outputs/repeat_timing' / a.name
                    log_dir.mkdir(parents=True, exist_ok=True)
                    execute([sys.executable, str(ROOT / 'scripts/benchmark_repeat_worker.py'),
                             '--config', model['config'], '--result', str(target),
                             '--warmup', '20', '--repeats', '3'],
                            log_dir, name[:-5])
                check_sample(read_json(target), model['count'], source['views'])
    # Source validation after measurement catches accidental concurrent source writes.
    if collect_sources() != sources:
        raise ValueError('Source changed during experiment')
    write_json(dest / 'summary.json', summarize(manifest, dest))
    print('Complete. Upload:', dest)


if __name__ == '__main__':
    main()
