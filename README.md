# SpotNet — Deep Learning Spot Detection & Decoding

A PyTorch reimplementation of the DeepCell spot-detection pipeline (`deepcell-spots`), extended with sub-pixel offset regression, multiplex barcode decoding, and a full training / evaluation harness.

---

## Overview

SpotNet detects fluorescent spots (e.g. smFISH, seqFISH, MERFISH) in microscopy images and, optionally, decodes the detected spots into gene identities using a probabilistic codebook model.

The pipeline has two main stages:

1. **Detection** — `SpotNet` (a fully-convolutional PyTorch network) outputs a per-pixel spot probability map together with a sub-pixel offset field, enabling localisation accuracy below one pixel.
2. **Decoding** — `SpotDecoding` fits a mixture model (Relaxed Bernoulli, Bernoulli, or Gaussian) via Pyro/SVI to assign each detected spot to a barcode in a user-supplied codebook.

---

## Repository Structure

```
detection         
    ├── modules.py          # Core building blocks: BNFeatureNet2D, BNFeatureNetSkip2D
    ├── dotnet.py           # Full SpotNet model: feature backbone + classification & regression heads
    ├── loss.py             # Training losses: weighted cross-entropy, focal loss, smooth-L1, regularisation
    ├── loader.py           # PyTorch Dataset / DataLoader; point-list ↔ annotation-map conversion
    ├── augmentation.py     # Affine augmentation (consistent image + coordinate transform)
    ├── utils.py            # Preprocessing, tiling/untiling, normalisation utilities
    ├── inference.py        # SpotDetection application: predict spots from raw images
    ├── postprocessing.py   # Convert raw model output to coordinate lists
    ├── metrics.py          # PointMetrics: Hungarian-matched TP/FP/FN, F1, RMSE, Chamfer distance
    ├── train.py            # Training loop with TensorBoard logging and model checkpointing
    ├── eval.py             # CLI evaluation script; outputs per-image metrics CSV
    └── eval_sweep.py       # Hyperparameter sweep over detection threshold & min-distance

decoding       
    ├── decode.py           # SpotDecoding application: assign spots to barcodes
    └── decoding_utils.py   # Pyro mixture-model implementation and E-step utilities

preprocess.py               # Preprocessing training data from DeepCell into a Zarr format
```

---

## Model Architecture

### SpotNet (`dotnet.py`)

```
Input (B, 1, H, W)
    └── BNFeatureNetSkip2D          ← sequential-refinement backbone
          ├── BNFeatureNet2D × (n_skips + 1)
          │     └── BatchNorm → ReflectPad Conv stack → Dense 1×1 Conv
          └── skip concat: [img, feat_{k-1}] at each iteration
    ├── ClassificationHead          ← 1×1 TensorProduct → Softmax (2 classes)
    └── OffsetRegressionHead        ← 4 × Conv2D(256, 3×3) → Conv2D(2, 3×3)

Output
    ├── detections  (B, 2, H, W)   per-pixel spot probability (softmax)
    └── offsets     (B, 2, H, W)   sub-pixel (Δy, Δx) offset to nearest spot
```

Default hyperparameters: `receptive_field=13`, `n_skips=3`, `n_conv_filters=32`, `n_dense_filters=128`, `regression_feature_size=256`.

---

## Installation

```bash
pip install torch torchvision scikit-image scipy numpy zarr pyro-ppl pandas tqdm tensorboard click
```

> GPU training requires a CUDA-capable device. CPU inference is supported.

---

## Quick Start

### Training

```python
# train.py  —  or run directly:
python -m spotnet.train
```

Data is read from Zarr stores at `~/.deepcell/spotnet/{train,val}.zarr`, each containing:

| Array   | Shape              | Description                          |
|---------|--------------------|--------------------------------------|
| `X`     | `(N, 1, H, W)`     | Grayscale image patches              |
| `y`     | `(N, max_spots, 2)`| Spot coordinates `[y, x]` per image  |
| `y_inds`| `(N,)`             | Number of valid spots in each image  |

Training configuration (epochs, batch size, learning rate, device, paths) is set via the `config` dict at the bottom of `train.py`.

### Inference

```python
from spotnet.inference import SpotDetection

detector = SpotDetection(model_path="path/to/saved_model_best_dict.pth", device="cuda:0")

# image: numpy array, shape (B, 1, H, W)
spots = detector.predict(image, threshold=0.99, min_distance=2)
# spots: numpy array of shape (N, 2) — detected [y, x] coordinates
```

Large images are automatically tiled and reassembled.

### Barcode Decoding

```python
import pandas as pd
from spotnet.decode import SpotDecoding

# df_barcodes: DataFrame with columns ['Gene', 'r0c0', 'r0c1', ..., 'rRcC']
app = SpotDecoding(df_barcodes, rounds=10, channels=2)
result = app.predict(spots_intensities_vec)  # shape (num_spots, rounds * channels)
# result: dict with 'predicted_name', 'probability', 'source' per spot
```

---

## Evaluation

### Single-run evaluation

```bash
python -m spotnet.eval \
  --model-path path/to/saved_model_best_dict.pth \
  --data-path  path/to/test.zarr \
  --device     cuda:0
```

Outputs `eval_results_<timestamp>.csv` with per-image TP, FP, FN, precision, recall, F1, RMSE, and Chamfer distance at τ = 2 px.

### Hyperparameter sweep

```bash
python -m spotnet.eval_sweep \
  --model-path path/to/saved_model_best_dict.pth \
  --data-path  path/to/test.zarr \
  --device     cuda:0
```

Sweeps `threshold ∈ [0.99, 0.999]` (10 steps) × `min_distance ∈ [1, 5]` (5 steps) × `τ ∈ [0, 5]` (10 steps), outputting `eval_sweep.csv`.

---

## Metrics

`PointMetrics` in `metrics.py` evaluates predicted point sets against ground truth using the Hungarian algorithm:

| Metric | Description |
|--------|-------------|
| TP / FP / FN | Matched/unmatched counts at distance threshold τ |
| Precision / Recall / F1 | Standard detection metrics |
| Mean / Median / RMSE dist | Spatial error on matched pairs (px) |
| 95th-percentile dist | Tail error |
| Bias Δx, Δy | Signed systematic offset |
| Chamfer distance | Threshold-free spatial summary |

---

## License

The codebase inherits the modified Apache 2.0 license of the Van Valen Lab at Caltech (non-commercial academic use).

---

## Acknowledgements

Built on [DeepCell](https://github.com/vanvalenlab/deepcell-tf) and [deepcell-spots](https://github.com/vanvalenlab/deepcell-spots) from the Van Valen Lab at the California Institute of Technology, with support from the Paul Allen Family Foundation, Google, and NIH grant U24CA224309-01.