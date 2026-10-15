#!/bin/bash
# A Python 3.12 environment for the training chain on a new machine: torch on the CUDA 12.8 wheels, the packages of
# train_only/requirements-train.txt, the repository's geokit (editable). The training chain imports only torch, numpy
# and the repository's own packages (measured: train_only/README.zh.md §2); the rest of the thesis env is not needed.
#
#   train_only/setup_env.sh [name]        a conda env (default: aeroviz); conda must be on PATH
#
# The CUDA 12.8 wheels need an NVIDIA driver >= 570 (nvidia-smi shows it). The versions are this machine's (2026-10-15).
set -euo pipefail
NAME=${1:-aeroviz}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$NAME" && { echo "conda env $NAME exists; choose another name" >&2; exit 1; }
conda create -y -n "$NAME" python=3.12
conda activate "$NAME"
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r "$ROOT/train_only/requirements-train.txt"
pip install -e "$ROOT/geokit"
python - <<'EOF'
import torch, numpy, geokit
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
print("numpy", numpy.__version__, "geokit", geokit.__file__)
EOF
echo "done: conda activate $NAME"
