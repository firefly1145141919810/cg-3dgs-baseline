"""Cloud-only render timing. Excludes PNG encoding, training and disk I/O."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import os
import socket
import datetime as dt
from types import SimpleNamespace
from run import ROOT, UPSTREAM, load_config, verify_upstream, write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/train.json')
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--result', required=True, type=Path)
    args=p.parse_args()
    if args.result.exists(): raise FileExistsError('Never overwrite a timing sample')
    if args.repeats < 1 or args.warmup < 0: raise ValueError('Invalid repeat count')
    c=load_config(args.config); verify_upstream()
    out=Path(c['output'])
    if json.loads((out/'manifest.json').read_text())['config'] != c: raise ValueError('Config mismatch')
    sys.path.insert(0,str(UPSTREAM))
    import torch
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    dataset=SimpleNamespace(source_path=c['source'],model_path=c['output'],images=c['images'],
        resolution=c['resolution'],eval=True,data_device='cpu',white_background=c['white_background'],sh_degree=3)
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    gaussians=GaussianModel(3)
    scene=Scene(dataset,gaussians,load_iteration=c['iterations'],shuffle=False)
    cameras=scene.getTestCameras()
    if not cameras: raise ValueError('No test cameras')
    bg=torch.tensor([1,1,1] if c['white_background'] else [0,0,0],device='cuda',dtype=torch.float32)
    with torch.no_grad():
        for i in range(args.warmup): render(cameras[i%len(cameras)],gaussians,pipe,bg)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times=[]
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            for camera in cameras:
                start.record(); image=render(camera,gaussians,pipe,bg); end.record()
                end.synchronize(); times.append(start.elapsed_time(end)); del image
        result={'protocol':'official Python renderer; CUDA events per frame; inference only; excludes PNG and disk I/O',
            'config_resolution':c['resolution'],'gpu':torch.cuda.get_device_name(),
            'warmup':args.warmup,'repeats':args.repeats,'test_views':len(cameras),
            'image_sizes':sorted({(v.image_width,v.image_height) for v in cameras}),
            'mean_ms':statistics.mean(times),'median_ms':statistics.median(times),
            'fps_from_mean_ms':1000/statistics.mean(times),
            'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved(),
            'gaussian_count':int(gaussians.get_xyz.shape[0]),'samples_ms':times,
            'caveat':'Not SIBR viewer FPS, not total nvidia-smi VRAM, not training peak memory.'}
    result['measurement_host'] = socket.gethostname()
    result['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
    result['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    result['completed_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(args.result,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
