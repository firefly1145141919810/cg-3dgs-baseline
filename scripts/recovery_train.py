"""Cloud-only fixed-count optimization; imports CUDA code only inside train()."""
import argparse
import hashlib
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from run import UPSTREAM, write_json
from dataset import sha256
from compression_phase1 import render_config


def optimization(steps):
    """Predeclared conservative recovery schedule, identical in both arms."""
    return SimpleNamespace(
        iterations=steps, position_lr_init=1.6e-6, position_lr_final=1.6e-6,
        position_lr_delay_mult=1.0, position_lr_max_steps=steps,
        feature_lr=0.00025, opacity_lr=0.005, scaling_lr=0.0005,
        rotation_lr=0.0001, percent_dense=0.01, lambda_dssim=0.2)


def camera_sequence(count, steps, seed):
    if count < 1 or steps < 1:
        raise ValueError('Need positive camera count and steps')
    rng = random.Random(seed)
    sequence = []
    while len(sequence) < steps:
        epoch = list(range(count))
        rng.shuffle(epoch)
        sequence.extend(epoch)
    return sequence[:steps]


def check_plan(steps, seed):
    if type(steps) is not int or not 1 <= steps <= 3000:
        raise ValueError('Recovery steps must be 1..3000 (3000 for formal experiment)')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')


def train(experiment, variant):
    from recovery_phase2 import verify, model_config, ply_path, TRAIN_VARIANTS, check_finished
    info = verify(experiment)
    if variant not in TRAIN_VARIANTS:
        raise ValueError('Only recovery variants may be trained')
    model = experiment / variant
    if (model / 'training.json').exists():
        check_finished(info, experiment, variant)
        print('Already complete and verified:', variant)
        return
    from doctor import diagnose
    env = diagnose()
    if env['errors']:
        raise ValueError(str(env))
    write_json(model / 'training_environment.json', env)
    sys.path.insert(0, str(UPSTREAM))
    import numpy as np
    import torch
    from scene import Scene, GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss, ssim
    from tqdm import tqdm
    seed, steps = info['seed'], info['steps']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    c = model_config(info, experiment, variant)
    dataset = render_config(c)
    # Keep source images on CPU; each selected training image moves to CUDA.
    dataset.data_device = 'cpu'
    gaussians = GaussianModel(3)
    scene = Scene(dataset, gaussians, load_iteration=info['source_iteration'], shuffle=False)
    # load_ply does NOT restore these fields: without this scale xyz LR is zero.
    gaussians.spatial_lr_scale = float(scene.cameras_extent)
    if not math.isfinite(gaussians.spatial_lr_scale) or gaussians.spatial_lr_scale <= 0:
        raise ValueError('Invalid scene spatial learning-rate scale')
    count = info['variants'][variant]['count']
    gaussians.max_radii2D = torch.zeros(count, device='cuda')
    opt = optimization(steps)
    gaussians.training_setup(opt)
    cameras = scene.getTrainCameras()
    if len(cameras) != info['train_views']:
        raise ValueError('Unexpected training camera count')
    sequence = camera_sequence(len(cameras), steps, seed)
    sequence_hash = hashlib.sha256(
        '\n'.join(cameras[i].image_name for i in sequence).encode('utf-8')).hexdigest()
    checkpoint = model / 'recovery_checkpoint.pth'
    manifest_hash = sha256(experiment / 'experiment.json')
    first, trace, seconds = 0, [], 0.0
    if checkpoint.exists():
        # Only load our own trusted local checkpoint, never a downloaded pickle.
        saved = torch.load(checkpoint, map_location='cuda')
        if (saved['manifest_sha256'] != manifest_hash or saved['sequence_sha256'] != sequence_hash
            or saved['variant'] != variant):
            raise ValueError('Checkpoint belongs to a different experiment')
        gaussians.restore(saved['model'], opt)
        first, trace, seconds = saved['step'], saved['trace'], saved['training_seconds']
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        if not 0 <= first <= steps or len(trace) != first:
            raise ValueError('Invalid checkpoint step/trace')
    if gaussians.get_xyz.shape[0] != count or gaussians.active_sh_degree != 3:
        raise ValueError('Invalid loaded model count/SH')
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([1, 1, 1] if c['white_background'] else [0, 0, 0],
                              dtype=torch.float32, device='cuda')
    torch.cuda.synchronize()
    started = time.perf_counter()
    for step in tqdm(range(first + 1, steps + 1), desc=variant):
        if gaussians.get_xyz.shape[0] != count or gaussians.active_sh_degree != 3:
            raise ValueError('Gaussian count or SH degree changed')
        gaussians.update_learning_rate(step)
        camera = cameras[sequence[step - 1]]
        image = render(camera, gaussians, pipe, background)['render']
        gt = camera.original_image.to('cuda')
        loss = 0.8 * l1_loss(image, gt) + 0.2 * (1 - ssim(image, gt))
        if not torch.isfinite(loss).item():
            raise ValueError('Non-finite training loss')
        loss.backward()
        # No densification, pruning, opacity reset, SH warm-up or GUI.
        gaussians.optimizer.step()  # Includes the final update, before saving.
        gaussians.optimizer.zero_grad(set_to_none=True)
        trace.append({'step': step, 'camera': camera.image_name, 'loss': float(loss.item())})
        if step % 500 == 0 or step == steps:
            for group in gaussians.optimizer.param_groups:
                if not all(torch.isfinite(p).all().item() for p in group['params']):
                    raise ValueError('Non-finite model parameters')
            torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            temp = model / 'recovery_checkpoint.tmp'
            torch.save({'model': gaussians.capture(), 'step': step, 'trace': trace,
                        'manifest_sha256': manifest_hash, 'sequence_sha256': sequence_hash, 'variant': variant,
                        'training_seconds': seconds, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state()}, temp)
            temp.replace(checkpoint)
            started = time.perf_counter()
    target = ply_path(model, c['iterations'])
    gaussians.save_ply(str(target))
    from prune_ply import read_vertices
    if len(read_vertices(target)) != count:
        raise ValueError('Saved Gaussian count changed')
    write_json(model / 'training.json', {
        'complete': True, 'steps': steps, 'seed': seed, 'count': count,
        'source_ply_sha256': info['variants'][variant]['source_sha256'],
        'final_ply_sha256': sha256(target), 'manifest_sha256': manifest_hash,
        'camera_sequence_sha256': sequence_hash, 'trace': trace,
        'training_seconds_excluding_checkpoint_io': seconds,
        'spatial_lr_scale': gaussians.spatial_lr_scale,
        'optimizer': 'fresh Adam eps=1e-15; restored on interrupted run',
        'optimization': vars(opt),
        'effective_final_lrs': {g['name']: g['lr'] for g in gaussians.optimizer.param_groups},
        'note': 'Train views only; constant Gaussian count; fixed 3000-step plan, no best-test selection'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--variant', required=True)
    args = parser.parse_args()
    train(args.experiment.resolve(), args.variant)
