"""Cloud-only diagnostics. Does not change the installed environment."""
import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys

def diagnose(before_build=False):
    info = {'python': sys.version, 'platform': platform.platform(), 'executable': sys.executable}
    errors = []
    for exe, args in [('nvidia-smi', ['--query-gpu=name,driver_version,memory.total', '--format=csv']),
                      ('nvcc', ['--version']), ('g++', ['--version'])]:
        if shutil.which(exe):
            result = subprocess.run([exe, *args], capture_output=True, text=True)
            info[exe] = result.stdout.strip()
            if result.returncode: errors.append(exe + ' failed')
        else:
            errors.append(exe + ' not found')
    try:
        import torch
        info['torch'] = torch.__version__
        info['torch_cuda'] = torch.version.cuda
        info['cuda_available'] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            errors.append('CUDA is unavailable to PyTorch')
        else:
            cap = torch.cuda.get_device_capability(0)
            info['gpu'] = torch.cuda.get_device_name(0)
            info['compute_capability'] = list(cap)
            if cap[0] >= 10:
                errors.append('This cu118/cu121 profile is not for Blackwell/newer GPUs. Select a modern compatible toolchain first.')
            # Basic cloud GPU operation catches missing binary architectures.
            info['cuda_tensor_test'] = float((torch.ones(4, device='cuda') * 2).sum().item())
        if not before_build:
            for module in ['diff_gaussian_rasterization', 'simple_knn._C', 'torchvision', 'numpy', 'PIL', 'plyfile']:
                try: importlib.import_module(module)
                except Exception as exc: errors.append(module + ': ' + str(exc))
    except Exception as exc:
        errors.append('PyTorch: ' + str(exc))
    info['errors'] = errors
    return info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before-build', action='store_true')
    args = parser.parse_args()
    result = diagnose(args.before_build)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    sys.exit(bool(result['errors']))
