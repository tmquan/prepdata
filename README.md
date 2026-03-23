# prepdata
Data preparation using NVIDIA NeMo Curator

## Setup

### 1. Create and activate the conda environment

```bash
conda create -n nemo python=3.13
conda activate nemo
```

### 2. Install uv (fast package installer)

```bash
pip install uv
```

### 3. Install PyTorch with CUDA 13 support

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 4. Install all remaining dependencies

```bash
uv pip install -r requirements.txt
```

### Alternative: use the NVIDIA PyTorch container

```bash
sudo docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:26.02-py3
```

Then run steps 2–4 inside the container.
