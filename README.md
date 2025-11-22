# DINO-based DeepFake Detection Model 

This repository contains an inference pipeline that uses a DINOV2 visual backbone with a single linear classification head to perform binary image classification. The code is lightweight and intended for running inference on a folder of images and producing a simple CSV of binary predictions.

## Project purpose

- Purpose: provide a script to run inference with a DINO backbone and a single linear output (sigmoid) and save per-image binary predictions.
- Typical use-case: evaluate or generate predictions for a dataset of images using a trained checkpoint located in `checkpoints/model.pth`.

## Task

- Binary image classification (single-output sigmoid). The model outputs a scalar per image; predictions are thresholded at 0.5 and written to a CSV (`result/cla_pre.csv`) with format: `image_name_without_extension, prediction` where prediction is 0 or 1.

## Model architecture

- Backbone: DINOV2 visual transformer loaded from `torch.hub` (repository `facebookresearch/dinov2`). The code constructs the backbone via `torch.hub.load(..., model="dinov2_vitl14")` in `src/dino_models.py`.
- Head: a single fully-connected layer (`nn.Linear`) mapping the backbone feature dimension to 1 output (binary). The `load_model` helper in `src/main.py` copies saved `fc.weight` and `fc.bias` from the checkpoint into the model.
- Channel sizes: the `CHANNELS` mapping in `src/dino_models.py` lists expected feature sizes for several architectures; for `ViT-L/14` the code expects 1024.

Notes:
- Although `src/__init__.py` lists many valid names, the current `get_model` function only instantiates DINO models for names starting with `"DINO:"`.
- The backbone loading is currently hard-coded to `dinov2_vitl14` in the `DINOModel` class.

## Files of interest

- `src/main.py` - main inference script. Builds the dataset, loads the model and runs predictions; writes `result/cla_pre.csv`.
- `src/dino_models.py` - defines `DINOModel` wrapper and channel sizes.
- `src/__init__.py` - helper `get_model(name)` that returns a model instance.
- `checkpoints/model.pth` - expected trained checkpoint containing a `model` state dict with `fc.weight` and `fc.bias` keys. This repository includes a `checkpoints/` folder (a file `model.pth` is present in the workspace).
- `doc/readme.docx` - additional documentation (binary docx file present in repo root).

## How it works (quick summary)

1. `TestDataset` scans a data directory (or loads a pickled list) for image files, applies a center crop to 224, converts to tensor, and normalizes using ImageNet mean/std.
2. `load_model(arch, ckpt)` instantiates the model, loads the checkpoint, copies FC weights/bias, moves the model to CUDA and sets eval mode.
3. `predict_and_save_results` runs the model on batches, applies sigmoid, thresholds at 0.5, and writes filename (without extension) + binary label to `result/cla_pre.csv`.

## Requirements

- Python 3.8+ (tested on 3.8/3.9/3.10) 
- PyTorch with CUDA (if you plan to use GPU) or CPU-only PyTorch
- torchvision
- pillow
- numpy
- scikit-learn
- tqdm
- scipy

Example pip install (run in PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision pillow numpy scikit-learn tqdm scipy
```

If you prefer a `requirements.txt`, use the above package list.

## Running inference

1. Prepare your images in a single directory. The script expects `test_path` to point to a directory containing image files (png/jpg/jpeg/bmp) or a pickle file that contains a list of image paths.
2. Ensure the checkpoint `checkpoints/model.pth` exists and contains the expected `model` entry with `fc.weight` and `fc.bias`.
3. Edit `src/main.py` or supply appropriate values in the script for:

- `test_path` (default in file is `/root/testdata` — replace this with your local path on Windows, e.g. `C:\data\test_images`)
- `arch` (default `DINO:ViT-L/14`)
- `ckpt` (default `./checkpoints/model.pth`)

Example: run inference (PowerShell):

```powershell
# Activate virtualenv if not already
.\.venv\Scripts\Activate.ps1

# From repository root
python .\src\main.py
```

The script will create a `result` folder (if not present) and save `cla_pre.csv` there.

### Notes about paths and Windows

- `src/main.py` uses UNIX-like default `test_path = '/root/testdata'` — change it to a Windows path when running locally. Example: `test_path = 'C:/path/to/test_images'` or use raw strings in Python.
- The script moves the model to CUDA with `model.cuda()` when loading; if you only have CPU PyTorch, remove or modify that line (e.g., guard with `if torch.cuda.is_available(): model.cuda()`).

## Output format

- `result/cla_pre.csv` contains two columns per row with no header:
  - column 1: image filename without extension
  - column 2: predicted label (0 or 1)

Example row:

```
000123,1
```

## Assumptions, limitations and suggested improvements

- The model wrapper currently always loads `dinov2_vitl14` from `torch.hub` regardless of the `name` passed into `DINOModel` — update this if you need different backbones.
- `get_model` only supports names starting with `"DINO:"`; other names in the `VALID_NAMES` list are placeholders.
- The script expects the checkpoint to have `fc.weight` and `fc.bias` stored under the `model` key in the checkpoint. If your checkpoint is differently organized, adapt `load_model` in `src/main.py`.
- Consider adding a CLI (argparse) to `src/main.py` to set `test_path`, `ckpt`, `batch_size`, `jpeg_quality`, and `gaussian_sigma` without editing the file.

## Contact / Next steps

If you want, I can:

- Add a `requirements.txt` and a small CLI wrapper to `src/main.py`.
- Add a CPU-only fallback and automatic CUDA availability check.
- Expand the README with examples and format checks.

---

Generated by reading the repository files (`src/main.py`, `src/dino_models.py`, `src/__init__.py`) and the contents of `checkpoints/`.
