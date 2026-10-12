#!/usr/bin/env bash
# =============================================================================
# setup_env.sh - DNGaussian environment on WSL2 (Ubuntu 24.04) with an NVIDIA GPU
#
# Record of the commands that worked on the laptop, in the order they were run.
#
#   Machine : Windows 11 (build 26200), WSL2 kernel 5.15.167.4, Ubuntu 24.04.4 LTS
#   GPU     : RTX 3050 Ti Laptop, 4 GB, compute capability 8.6
#   Result  : CUDA toolkit 11.3.109 | gcc 9.5.0 | Python 3.7.13 | torch 1.12.1 (cu113)
#
# STATUS: each step was run on the laptop, but this file has NOT been run
#         start to finish as a single script. Steps are guarded, so re-running
#         skips work that is already done.
#
# Usage (inside WSL, repo on the Linux filesystem, NOT under /mnt/c):
#   cd ~/projects/<repo> && bash setup_env.sh
#   You will be asked for your sudo password (apt and dpkg steps).
#
# Not covered: Google Colab (T4, compute capability 7.5) needs its own procedure.
# Deviations from the upstream DNGaussian README are explained at each step.
# =============================================================================
set -eo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="dngaussian"
CUDA_HOME_DIR="/usr/local/cuda-11.3"
GCC_WRAP="$HOME/gcc10bin"                       # historical name; it points at gcc-9
HDRS="$HOME/build-deps/linux-headers-5.4"
ENV_FILE="$HOME/.dngaussian_env.sh"

step() { echo; echo "=== $* ==="; }

case "$REPO_DIR" in
  /mnt/*) echo "ERROR: keep the repo on the Linux filesystem (e.g. ~/projects), not under /mnt/"; exit 1;;
esac
[ -f "$REPO_DIR/environment.yml" ] || { echo "ERROR: environment.yml not found in $REPO_DIR"; exit 1; }

# -----------------------------------------------------------------------------
step "0. GPU visible inside WSL2"
# The NVIDIA driver lives on the Windows side. Never install a Linux driver in WSL.
nvidia-smi > /dev/null || { echo "ERROR: nvidia-smi failed inside WSL2. Fix GPU passthrough first."; exit 1; }

# -----------------------------------------------------------------------------
step "1. System packages"
# gcc-9 / g++-9: nvcc 11.3 rejects Ubuntu 24.04's default gcc 13, and PyTorch 1.12.1's
# own check rejects g++ 10.5.0 (its ceiling is 10.0.0). gcc-9 passes both checks.
# rsync: used by the kernel header install on some versions; included defensively.
sudo apt update
sudo apt install -y build-essential git wget unzip rsync gcc-9 g++-9

# -----------------------------------------------------------------------------
step "2. Miniconda and channel terms of service"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
grep -q ">>> conda initialize" "$HOME/.bashrc" || conda init bash
# conda env create refuses to run until these are accepted
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# -----------------------------------------------------------------------------
step "3. CUDA toolkit 11.3 (gives nvcc)"
# conda's cudatoolkit=11.3 (pinned in environment.yml) is runtime-only and has no nvcc.
# The wsl-ubuntu repo target deliberately excludes the driver. If apt ever proposes
# an nvidia-driver-* or linux-modules-nvidia-* package here, stop and do not confirm.
if [ ! -x "$CUDA_HOME_DIR/bin/nvcc" ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  sudo dpkg -i /tmp/cuda-keyring.deb
  sudo apt-get update
  sudo apt-get -y install cuda-toolkit-11-3
fi

# -----------------------------------------------------------------------------
step "4. gcc-9 wrapper directory (put ahead of the system compilers on PATH)"
mkdir -p "$GCC_WRAP"
ln -sf /usr/bin/gcc-9 "$GCC_WRAP/gcc"
ln -sf /usr/bin/g++-9 "$GCC_WRAP/g++"
ln -sf /usr/bin/gcc-9 "$GCC_WRAP/cc"      # ninja builds call cc / c++
ln -sf /usr/bin/g++-9 "$GCC_WRAP/c++"

# -----------------------------------------------------------------------------
step "5. Linux 5.4 kernel headers"
# nvcc 11.3 cannot parse the __s128 typedef in Ubuntu 24.04's /usr/include/linux/types.h
# ("expected initializer before '__s128'"). Older headers are searched first via CPATH.
# Compile-time only: it has no effect on how anything runs.
if [ ! -f "$HDRS/include/linux/types.h" ]; then
  mkdir -p "$HOME/build-deps"
  (
    cd "$HOME/build-deps"
    [ -f linux-5.4.tar.xz ] || wget -q https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.tar.xz
    [ -d linux-5.4 ] || tar -xf linux-5.4.tar.xz
    cd linux-5.4
    make headers_install INSTALL_HDR_PATH="$HDRS"
  )
fi

# -----------------------------------------------------------------------------
step "6. Build environment variables (persisted for new shells)"
# LD_LIBRARY_PATH needs /usr/lib/wsl/lib, otherwise cuDNN fails with
# "Could not load library libcudnn_cnn_infer.so.8 ... libcuda.so: cannot open shared object file".
# TORCH_CUDA_ARCH_LIST=8.6 is the RTX 3050 Ti. Change it for other GPUs
# (T4 = 7.5, A100 = 8.0).
cat > "$ENV_FILE" <<'EOF'
export PATH=$HOME/gcc10bin:/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-11.3
export TORCH_CUDA_ARCH_LIST="8.6"
export CPATH=$HOME/build-deps/linux-headers-5.4/include${CPATH:+:$CPATH}
EOF
grep -qF ".dngaussian_env.sh" "$HOME/.bashrc" || \
  echo '[ -f ~/.dngaussian_env.sh ] && source ~/.dngaussian_env.sh' >> "$HOME/.bashrc"
# shellcheck disable=SC1090
source "$ENV_FILE"

[[ "$(gcc -dumpversion)" == 9* ]] || { echo "ERROR: gcc on PATH is $(gcc -dumpversion), expected 9.x"; exit 1; }
[[ "$(nvcc --version)" == *"release 11.3"* ]] || { echo "ERROR: nvcc is not CUDA 11.3"; exit 1; }
echo "gcc $(gcc -dumpversion) | $(nvcc --version | tail -2 | head -1)"

# -----------------------------------------------------------------------------
step "7. CUDA extension sources that live in submodules/"
# This team repo vendors both sources under submodules/, so a normal clone skips this step.
# The clones below only run in a bare upstream DNGaussian checkout, which ships submodules/
# empty. The README clones the rasterizer over SSH; HTTPS is used here, so no SSH key is needed.
# diff-gaussian-rasterization is ashawkey's fork, not the original INRIA rasterizer.
# --recursive is required: it pulls the glm headers.
cd "$REPO_DIR"
if [ ! -f submodules/diff-gaussian-rasterization/setup.py ]; then
  git clone https://github.com/ashawkey/diff-gaussian-rasterization.git --recursive submodules/diff-gaussian-rasterization
fi
if [ ! -f submodules/simple-knn/setup.py ]; then
  git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
fi

# -----------------------------------------------------------------------------
step "8. Conda environment (Python 3.7.13, torch 1.12.1, cudatoolkit 11.3, ...)"
# On the laptop `conda env create` installed all conda packages, then exited with
# "CondaEnvException: Pip failed": the unpinned timm pulls a safetensors that has no
# Python 3.7 wheel. The environment remains usable, so the pip packages are installed
# by hand in step 9. The exit code is therefore ignored and the env's existence checked.
if [ ! -d "$CONDA_DIR/envs/$ENV_NAME" ]; then
  conda env create --file environment.yml \
    || echo "(conda env create exited non-zero; checking that the env exists)"
  rm -f condaenv.*.requirements.txt
fi
[ -d "$CONDA_DIR/envs/$ENV_NAME" ] || { echo "ERROR: conda env '$ENV_NAME' was not created"; exit 1; }
conda activate "$ENV_NAME"
python --version

# -----------------------------------------------------------------------------
step "9. pip packages"
# Versions are the ones that resolved on the laptop (see pip_freeze_laptop.txt for the full list).
# timm 0.6.13 predates timm's hard safetensors dependency. Pillow 9.5.0 comes from PyPI because
# conda's Pillow 9.4.0 links against libtiff.so.5, which the env does not contain
# ("ImportError: libtiff.so.5" when the depth script runs).
python -m pip install "timm==0.6.13" "safetensors==0.3.1" "tensorboard==2.11.2" \
  "opencv-python==5.0.0.93" "scikit-image==0.19.3" "imageio==2.31.2" "matplotlib==3.5.3" pillow
python -m pip install --force-reinstall --no-deps "pillow==9.5.0"
python -m pip install "ninja==1.11.1.4"       # parallel builds; MAX_JOBS keeps RAM use down

# -----------------------------------------------------------------------------
step "10. Build and install the three CUDA extensions"
# MAX_JOBS=2 keeps parallel nvcc jobs inside WSL2's default ~7.6 GB RAM.
# Failure hints:
#   "g++ (x.y) is greater than the maximum required version by CUDA 11.3" -> gcc wrapper not first on PATH
#   "expected initializer before '__s128'"                                -> CPATH not set (step 5/6)
export MAX_JOBS=2
build_ext() {   # $1 = path inside the repo, $2 = python snippet that succeeds when installed
  if (cd "$HOME" && python -c "$2" > /dev/null 2>&1); then
    echo "already installed, skipping: $1"
  else
    python -m pip install "$REPO_DIR/$1"
  fi
}
build_ext gridencoder                           "import torch, _gridencoder"
build_ext submodules/diff-gaussian-rasterization "import torch, diff_gaussian_rasterization"
build_ext submodules/simple-knn                  "import torch; from simple_knn._C import distCUDA2"

# -----------------------------------------------------------------------------
step "11. Verify (run from \$HOME; torch must be imported before _gridencoder)"
(cd "$HOME" && python -c "import torch; import _gridencoder; import diff_gaussian_rasterization; from simple_knn._C import distCUDA2; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| all 3 extensions OK')")

cat <<'EOF'

Environment ready. Remaining steps are manual (data is not part of the repo):

  1. Download LLFF (nerf_llff_data.zip) in a browser:
       https://drive.google.com/file/d/11PhkBXZZNYTD2emdG1awALlhCnkq7aN-/view?usp=drive_link
     Copy the zip to the Linux filesystem (reads from /mnt/c are slow), then from the repo root:
       mkdir -p data/llff
       unzip -q nerf_llff_data.zip -d data/llff_tmp
       mv data/llff_tmp/nerf_llff_data/* data/llff/ && rmdir data/llff_tmp/nerf_llff_data data/llff_tmp
     Expected: data/llff/fern/images, data/llff/fern/sparse/0, data/llff/fern/poses_bounds.npy

  2. Monocular depth maps for all 8 scenes (first run downloads MiDaS code + DPT_Hybrid weights):
       conda activate dngaussian
       cd dpt && python get_depth_map_for_llff_dtu.py --root_path ../data/llff --benchmark LLFF
     (do not use dpt/get_depth_map.sh: it points at ../data/distorted/nerf_llff_data)
     Check: data/llff/<scene>/depth_maps/ holds one depth_*.png per image.

  3. Train, render and evaluate one scene (about 7.5 minutes on the RTX 3050 Ti):
       bash scripts/run_llff.sh data/llff/fern output/llff/fern 0
     Metrics are written to output/llff/fern/results_eval.json.
EOF
