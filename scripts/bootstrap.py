"""Initialize only the optimizer dependencies, excluding the optional SIBR GUI."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/gaussian-splatting'

def run(*args, cwd=ROOT):
    subprocess.run(args, cwd=cwd, check=True)

def main():
    lock = json.loads((ROOT / 'baseline.lock.json').read_text())
    run('git', 'submodule', 'update', '--init', 'vendor/gaussian-splatting')
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip()
    if actual != lock['commit']:
        raise SystemExit('Upstream commit differs from baseline.lock.json; refusing to continue.')
    run('git', 'submodule', 'update', '--init', '--recursive',
        'submodules/diff-gaussian-rasterization', 'submodules/simple-knn', cwd=VENDOR)
    print('Locked upstream:', actual)

if __name__ == '__main__':
    main()
