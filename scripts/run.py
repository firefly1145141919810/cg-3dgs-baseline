"""Explicit, logged wrapper around unmodified official train/render/metrics scripts."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataset import validate

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'vendor/gaussian-splatting'

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def resolve(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()

def load_config(path):
    c = json.loads(resolve(path).read_text(encoding='utf-8'))
    required = {'scene','source','output','images','resolution','iterations','eval_iterations',
                'checkpoint_iterations','data_device','white_background','port'}
    if set(c) != required: raise ValueError('Unexpected or missing config keys: ' + str(set(c) ^ required))
    if not isinstance(c['iterations'], int) or c['iterations'] < 1: raise ValueError('iterations must be positive')
    if c['resolution'] not in [-1,1,2,4,8] and c['resolution'] < 16: raise ValueError('Invalid resolution')
    if c['data_device'] not in ('cuda','cpu'): raise ValueError('Invalid data_device')
    for key in ('eval_iterations','checkpoint_iterations'):
        if not c[key] or any(not isinstance(i,int) or not 1 <= i <= c['iterations'] for i in c[key]):
            raise ValueError('Invalid ' + key)
    if c['iterations'] not in c['eval_iterations']: raise ValueError('Evaluate final iteration')
    if not 0 <= c['port'] <= 65535: raise ValueError('Invalid port')
    if not isinstance(c['white_background'], bool): raise ValueError('white_background must be bool')
    c['source'], c['output'] = str(resolve(c['source'])), str(resolve(c['output']))
    if Path(c['source']) == Path(c['output']): raise ValueError('Output cannot be dataset directory')
    return c

def command(stage, c, checkpoint=None):
    cmd = [sys.executable, str(UPSTREAM / (stage + '.py'))]
    if stage == 'metrics': return cmd + ['-m', c['output']]
    cmd += ['-s', c['source'], '-m', c['output'], '-i', c['images'], '-r', str(c['resolution']),
            '--eval', '--data_device', c['data_device'], '--sh_degree', '3']
    if c['white_background']: cmd += ['--white_background']
    if stage == 'train':
        cmd += ['--iterations', str(c['iterations']), '--ip', '127.0.0.1', '--port', str(c['port']),
                '--test_iterations', *map(str,c['eval_iterations']),
                '--save_iterations', *map(str,c['eval_iterations']),
                '--checkpoint_iterations', *map(str,c['checkpoint_iterations'])]
        if checkpoint: cmd += ['--start_checkpoint', str(resolve(checkpoint))]
    elif stage == 'render':
        cmd += ['--iteration', str(c['iterations']), '--skip_train']
    return cmd

def verify_upstream():
    expected = json.loads((ROOT / 'baseline.lock.json').read_text())['commit']
    current = subprocess.check_output(['git','rev-parse','HEAD'], cwd=UPSTREAM, text=True).strip()
    if current != expected: raise ValueError('Wrong upstream commit; run bootstrap.py')
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=UPSTREAM, text=True)
    if dirty: raise ValueError('Upstream modified. Use a separate experiment, not baseline.')
    modules = subprocess.check_output(['git','submodule','status','--recursive',
                'submodules/diff-gaussian-rasterization','submodules/simple-knn'], cwd=UPSTREAM, text=True)
    if any(line[:1] != ' ' for line in modules.splitlines()): raise ValueError('Missing or mismatched CUDA submodules')
    return {'commit': current, 'submodules': modules.splitlines()}

def execute(cmd, out, stage):
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    start = time.perf_counter()
    env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'
    log = out / (stage + '_' + stamp + '.log')
    with log.open('w', encoding='utf-8') as f:
        f.write(json.dumps(cmd) + '\n'); f.flush()
        process = subprocess.Popen(cmd, cwd=UPSTREAM, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors='replace')
        try:
            for line in process.stdout:
                print(line, end=''); f.write(line); f.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    record = {'stage':stage, 'command':cmd, 'utc':stamp, 'returncode':code,
              'wall_seconds_including_io_eval':time.perf_counter()-start, 'log':log.name}
    with (out / 'stage_history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
    if code: raise SystemExit('Stage failed; see ' + str(log))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['plan','train','render','metrics','all'])
    p.add_argument('--config',default='configs/train.json')
    p.add_argument('--resume', help='Trusted local .pth checkpoint; exact stochastic resume is not guaranteed by upstream')
    args = p.parse_args()
    c = load_config(args.config)
    stages = ['train','render','metrics'] if args.stage in ('all','plan') else [args.stage]
    if args.stage == 'plan':
        print(json.dumps({s:command(s,c,args.resume) for s in stages},indent=2)); return
    upstream = verify_upstream()
    dataset = validate(c['source'],c['images'])
    out = Path(c['output'])
    if args.resume:
        if 'train' not in stages or not resolve(args.resume).is_file(): raise ValueError('Invalid resume checkpoint')
    manifest = out / 'manifest.json'
    if 'train' in stages and not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite existing run. Use new output or --resume.')
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous['config'] != c: raise ValueError('Config differs from recorded run')
        old = json.loads((out/'dataset.json').read_text())
        for key in ('train','test','camera_hashes','image_sha256'):
            if old[key] != dataset[key]: raise ValueError('Dataset changed: '+key)
    elif 'train' not in stages or args.resume:
        raise ValueError('Run manifest is missing; do not evaluate or resume an unrelated model')
    else:
        from doctor import diagnose
        diagnostic = diagnose()
        if diagnostic['errors']: raise ValueError(json.dumps(diagnostic,indent=2))
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'environment.json',diagnostic)
        # pip list avoids copying possible private package URLs from pip freeze.
        packages = subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True)
        (out/'packages.json').write_text(packages,encoding='utf-8')
        write_json(out/'dataset.json',dataset)
        write_json(manifest,{'config':c,'upstream':upstream,'seed':0,
                  'wrapper_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'protocol':'heldout every eighth COLMAP view; official 2023 defaults; no algorithm extensions'})
    for stage in stages:
        if stage == 'render' and not (out/f'point_cloud/iteration_{c["iterations"]}/point_cloud.ply').is_file():
            raise ValueError('Final trained model missing')
        if stage == 'metrics':
            method = out/f'test/ours_{c["iterations"]}'
            for folder in ['renders','gt']:
                if len(list((method/folder).glob('*.png'))) != len(dataset['test']):
                    raise ValueError('Incomplete test renders or ground truth')
            # Official metrics.py catches exceptions and may still exit zero.
            # Move old results aside so they can never masquerade as a successful rerun.
            for filename in ('results.json', 'per_view.json'):
                prior = out / filename
                if prior.exists():
                    prior.rename(out / (filename + '.' + str(time.time_ns()) + '.bak'))
        execute(command(stage,c,args.resume),out,stage)
        if stage == 'metrics':
            result = json.loads((out/'results.json').read_text())
            if f'ours_{c["iterations"]}' not in result: raise ValueError('Upstream metrics failed or skipped final method')
    print('Completed stages:', ', '.join(stages))

if __name__ == '__main__':
    main()
