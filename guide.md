# Install WeNet from Archive for ONNX Export

This guide explains how to install **WeNet from a downloaded GitHub archive** and prepare the environment for exporting WeNet checkpoints to ONNX.

The install choices are kept together:

- Conda CPU
- Conda GPU
- uv CPU
- uv GPU

The goal is to make this command work:

```bash
python -m wenet.bin.export_onnx_cpu --help
```

For ONNX export, CPU PyTorch is enough. Use the GPU installs when the same environment also needs CUDA PyTorch for training, decoding, or CUDA-side checks.

---

## 1. Get the WeNet Source Archive

Download and extract the WeNet GitHub archive, then enter the source directory:

```bash
cd wenet-main
```

Use Python **3.10** for all install paths below.

Full pinned requirement files used below:

- `requirements-onnx-cpu.txt`
- `requirements-onnx-gpu-cu121.txt`
- `requirements-onnx-gpu-cu118.txt`
- `constraints-build.txt` for uv build isolation

---

## 2. Shared Dependency: sox

Install `sox` before installing WeNet dependencies.

For Conda environments:

```bash
conda install conda-forge::sox -y
```

For uv virtual environments on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y sox libsox-dev
```

If `sox` is already available, this check should print its version:

```bash
sox --version
```

---

## 3. Conda CPU Install

```bash
# Enter WeNet source directory first
cd wenet-main

# Create environment
conda create -n wenet-cpu python=3.10 -y
conda activate wenet-cpu

# Install system/audio dependency
conda install conda-forge::sox -y

# Install WeNet dependencies with pinned CPU PyTorch, NumPy, ONNX, and ONNX Runtime
python -m pip install --upgrade pip
python -m pip install -r requirements-onnx-cpu.txt

# Install WeNet itself
python -m pip install -e .

# Verify
python -c "import torch; import torchaudio; print(torch.__version__, torch.version.cuda); print(torchaudio.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import onnx, onnxruntime; print('onnx ok')"
python -m wenet.bin.export_onnx_cpu --help
```

Expected PyTorch check:

```text
2.2.2+cpu None
2.2.2+cpu
```

---

## 4. Conda GPU Install

This installs the matching CUDA 12.1 PyTorch stack. The NVIDIA driver must support CUDA 12.1 runtime wheels.

```bash
# Enter WeNet source directory first
cd wenet-main

# Create environment
conda create -n wenet-gpu python=3.10 -y
conda activate wenet-gpu

# Install system/audio dependency
conda install conda-forge::sox -y

# Install WeNet dependencies with pinned CUDA PyTorch, NumPy, ONNX, and ONNX Runtime
python -m pip install --upgrade pip
python -m pip install -r requirements-onnx-gpu-cu121.txt

# Optional: use this instead of onnxruntime if you need Python ONNX Runtime CUDA inference
# python -m pip uninstall -y onnxruntime
# python -m pip install onnxruntime-gpu==1.17.3

# Install WeNet itself
python -m pip install -e .

# Verify
nvidia-smi
python -c "import torch; import torchaudio; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torchaudio.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import onnx, onnxruntime; print('onnx ok')"
python -m wenet.bin.export_onnx_cpu --help
```

Expected PyTorch check:

```text
2.2.2+cu121 12.1 True
2.2.2+cu121
```

If your machine requires CUDA 11.8 wheels instead, use:

```bash
python -m pip install -r requirements-onnx-gpu-cu118.txt
```

---

## 5. uv CPU Install

Install `uv` first if it is not already available:

```bash
python -m pip install --user uv
```

Then create and use a Python 3.10 virtual environment:

```bash
# Enter WeNet source directory first
cd wenet-main

# Create environment
uv venv .venv-wenet-cpu --python 3.10
source .venv-wenet-cpu/bin/activate

# Install WeNet dependencies with pinned CPU PyTorch, NumPy, ONNX, and ONNX Runtime
# constraints-build.txt keeps openai-whisper==20231117 build-compatible.
uv pip install -r requirements-onnx-cpu.txt --build-constraint constraints-build.txt

# Install WeNet itself
uv pip install -e .

# Verify
python -c "import torch; import torchaudio; print(torch.__version__, torch.version.cuda); print(torchaudio.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import onnx, onnxruntime; print('onnx ok')"
python -m wenet.bin.export_onnx_cpu --help
```

Expected PyTorch check:

```text
2.2.2+cpu None
2.2.2+cpu
```

---

## 6. uv GPU Install

Install `uv` first if it is not already available:

```bash
python -m pip install --user uv
```

Then create and use a Python 3.10 virtual environment:

```bash
# Enter WeNet source directory first
cd wenet-main

# Create environment
uv venv .venv-wenet-gpu --python 3.10
source .venv-wenet-gpu/bin/activate

# Install WeNet dependencies with pinned CUDA PyTorch, NumPy, ONNX, and ONNX Runtime
# constraints-build.txt keeps openai-whisper==20231117 build-compatible.
uv pip install -r requirements-onnx-gpu-cu121.txt --build-constraint constraints-build.txt

# Optional: use this instead of onnxruntime if you need Python ONNX Runtime CUDA inference
# uv pip uninstall onnxruntime
# uv pip install onnxruntime-gpu==1.17.3

# Install WeNet itself
uv pip install -e .

# Verify
nvidia-smi
python -c "import torch; import torchaudio; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torchaudio.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import onnx, onnxruntime; print('onnx ok')"
python -m wenet.bin.export_onnx_cpu --help
```

Expected PyTorch check:

```text
2.2.2+cu121 12.1 True
2.2.2+cu121
```

If your machine requires CUDA 11.8 wheels instead, use:

```bash
uv pip install -r requirements-onnx-gpu-cu118.txt --build-constraint constraints-build.txt
```

---

# Common Errors and Fixes

## Error: `libcudart.so.13` Not Found

Example:

```text
OSError: libcudart.so.13: cannot open shared object file: No such file or directory
```

Reason:

```text
torch / torchaudio was installed with an incompatible CUDA runtime.
```

CPU fix:

```bash
python -m pip install --force-reinstall -r requirements-onnx-cpu.txt
```

GPU fix:

```bash
python -m pip install --force-reinstall -r requirements-onnx-gpu-cu121.txt
```

For uv:

```bash
uv pip install --reinstall -r requirements-onnx-cpu.txt
```

Or for GPU:

```bash
uv pip install --reinstall -r requirements-onnx-gpu-cu121.txt
```

---

## Error: NumPy 2.x Incompatibility

Example:

```text
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

Fix:

```bash
python -m pip install --force-reinstall "numpy==1.26.4"
```

For uv:

```bash
uv pip install --force-reinstall "numpy==1.26.4"
```

---

## Error: `No module named 'pkg_resources'` When Building `openai-whisper`

Example:

```text
ModuleNotFoundError: No module named 'pkg_resources'
Failed to build `openai-whisper==20231117`
```

Reason:

```text
openai-whisper==20231117 imports pkg_resources during setup, but newer setuptools versions no longer provide it.
```

uv fix:

```bash
uv pip install -r requirements-onnx-cpu.txt --build-constraint constraints-build.txt
```

If that still fails, build Whisper without isolation after installing the build tools into the environment:

```bash
uv pip install "setuptools==80.9.0" "wheel==0.45.1"
uv pip install "openai-whisper==20231117" --no-build-isolation
uv pip install -r requirements-onnx-cpu.txt
```

For pip:

```bash
python -m pip install "setuptools==80.9.0" "wheel==0.45.1"
python -m pip install "openai-whisper==20231117" --no-build-isolation
python -m pip install -r requirements-onnx-cpu.txt
```

---

## Error: `Please install onnx and onnxruntime!`

Fix:

```bash
python -m pip install "onnx==1.16.0" "onnxruntime==1.17.3"
```

For uv:

```bash
uv pip install "onnx==1.16.0" "onnxruntime==1.17.3"
```

---

## Warning: `Module "torch_npu" not found`

Example:

```text
Module "torch_npu" not found.
pip install torch_npu if you are using Ascend NPU, otherwise, ignore it.
```

This warning can be ignored unless you are using Huawei Ascend NPU.

---

# Export a WeNet Checkpoint to ONNX

After the installation works, export a WeNet checkpoint:

```bash
python -m wenet.bin.export_onnx_cpu \
  --config /path/to/train.yaml \
  --checkpoint /path/to/final.pt \
  --chunk_size 16 \
  --num_decoding_left_chunks -1 \
  --output_dir /path/to/onnx_streaming
```

Expected output files:

```text
encoder.onnx
ctc.onnx
decoder.onnx
encoder.quant.onnx
ctc.quant.onnx
decoder.quant.onnx
```

For streaming runtime testing, make sure the `chunk_size` and `num_decoding_left_chunks` used during export are the same values used during runtime decoding.
