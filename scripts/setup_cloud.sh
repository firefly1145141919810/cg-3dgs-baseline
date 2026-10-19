#!/usr/bin/env bash
# Run only on a Linux NVIDIA CUDA server in a NEW activated Conda environment.
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-cu121}"
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
  echo 'Activate a dedicated Python 3.10 Conda environment (not base).' >&2; exit 2
fi
command -v nvcc >/dev/null || { echo 'CUDA runtime alone is insufficient: nvcc is required.' >&2; exit 2; }
command -v g++ >/dev/null || { echo 'Install the C++ compiler first.' >&2; exit 2; }
python -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
case "$profile" in
  cu121) torch_version=2.2.2; vision_version=0.17.2; toolkit=12.1 ;;
  cu118) torch_version=2.2.2; vision_version=0.17.2; toolkit=11.8 ;;
  *) echo 'Usage: bash scripts/setup_cloud.sh cu121|cu118' >&2; exit 2 ;;
esac
nvcc --version | grep -q "release ${toolkit}," || {
  echo "Select CUDA toolkit ${toolkit} via CUDA_HOME and PATH before continuing." >&2; exit 2;
}
python scripts/bootstrap.py
python -m pip install 'pip==24.0' 'setuptools==69.5.1' 'wheel==0.43.0'
python -m pip install "torch==${torch_version}" "torchvision==${vision_version}" --index-url "https://download.pytorch.org/whl/${profile}"
python -m pip install -r environment/requirements.txt
python scripts/doctor.py --before-build
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install --no-build-isolation ./vendor/gaussian-splatting/submodules/simple-knn
python -m pip install --no-build-isolation ./vendor/gaussian-splatting/submodules/diff-gaussian-rasterization
python scripts/doctor.py
echo 'Setup finished. Run the smoke experiment on the cloud server next.'
