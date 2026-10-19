"""Export small, sanitized result records suitable for Git; never invent metrics."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
from run import ROOT, load_config, write_json

def ply_count(path):
    with path.open('rb') as f:
        for _ in range(1000):
            line=f.readline().decode('ascii').strip()
            if line.startswith('element vertex '): return int(line.split()[-1])
            if line == 'end_header': break
    raise ValueError('PLY vertex count not found')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/train.json')
    p.add_argument('--name',required=True,help='Unique safe run label, e.g. train-a100-30k-20261020')
    args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.name): raise ValueError('Unsafe run name')
    c=load_config(args.config); out=Path(c['output'])
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['config'] != c: raise ValueError('Config mismatch')
    metrics=json.loads((out/'results.json').read_text())[f'ours_{c["iterations"]}']
    if any(k not in metrics or not math.isfinite(metrics[k]) for k in ('PSNR','SSIM','LPIPS')):
        raise ValueError('Missing or nonfinite metrics')
    ply=out/f'point_cloud/iteration_{c["iterations"]}/point_cloud.ply'
    data=json.loads((out/'dataset.json').read_text())
    env=json.loads((out/'environment.json').read_text())
    row={'run':args.name,'scene':c['scene'],'iterations':c['iterations'],'resolution':c['resolution'],
        **metrics,'gaussian_count':ply_count(ply),'ply_bytes':ply.stat().st_size,
        'gpu':env.get('gpu'),'upstream_commit':manifest['upstream']['commit'],
        'train_views':len(data['train']),'test_views':len(data['test'])}
    benchmark=None
    if (out/'benchmark.json').exists():
        benchmark=json.loads((out/'benchmark.json').read_text())
        row.update({k:benchmark[k] for k in ['mean_ms','fps_from_mean_ms','peak_torch_allocated_bytes']})
    dest=ROOT/'results'/args.name
    dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'summary.json',row)
    with (dest/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    write_json(dest/'metrics_per_view.json',json.loads((out/'per_view.json').read_text()))
    sanitized={k:v for k,v in c.items() if k not in ('source','output')}
    write_json(dest/'protocol.json',{'config':sanitized,'seed':manifest['seed'],
        'upstream':manifest['upstream'],'wrapper_commit':manifest['wrapper_commit'],
        'dataset_camera_hashes':data['camera_hashes'],'test_names':data['test']})
    if benchmark: write_json(dest/'benchmark.json',benchmark)
    print('Exported:',dest)

if __name__=='__main__': main()
