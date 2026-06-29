# torch-spots

A PyTorch pipeline for detecting and decoding fluorescent spots in multiplexed imaging data. The library is organized into two independent branches: **detection** (finding where spots are) and **decoding** (identifying which gene each spot represents).

---

## Detection Branch (`torch_spots/detection/`)

The detection branch locates individual fluorescent spots in microscopy images with sub-pixel precision. It uses a fully-convolutional neural network called **SpotNet** that jointly predicts a spot probability map and a sub-pixel offset map.

### Architecture

**SpotNet** ([dotnet.py](torch_spots/detection/dotnet.py)) has three components:

1. **BNFeatureNetSkip2D backbone** ([modules.py](torch_spots/detection/modules.py)) — a sequential-refinement feature extractor. It runs `n_skips + 1` sub-networks (`BNFeatureNet2D`) in series. The first subnet processes the normalized input image alone; every subsequent subnet receives the concatenation of the original image and the previous subnet's output. This skip-connection design keeps raw pixel evidence visible at every stage of refinement.

   Each `BNFeatureNet2D` is a dilated convolutional stack built from `Conv2D → BatchNorm → ReLU` blocks with a final "dense" conv that captures the remaining receptive field. With `receptive_field=13` (default), the stack produces 2 conv layers plus 1 dilated pass before the dense conv. `ReflectPadConv2d` is used throughout to preserve spatial dimensions `(H, W)` exactly.

   Shape flow through backbone (default config, `n_skips=3`):

   ```text
   Iteration 0: (B, 1, H, W)    → BNFeatureNet2D → (B, 128, H, W)
   Iteration 1: (B, 129, H, W)  → BNFeatureNet2D → (B, 128, H, W)
   Iteration 2: (B, 129, H, W)  → BNFeatureNet2D → (B, 128, H, W)
   Iteration 3: (B, 129, H, W)  → BNFeatureNet2D → (B, 128, H, W)
   ```

2. **ClassificationHead** — two 1×1 convolutions (`Conv → BN → ReLU → Conv`) followed by softmax. Outputs `(B, 2, H, W)`: channel 0 is `p(background)`, channel 1 is `p(spot center)`.

3. **OffsetRegressionHead** — four 3×3 Conv+ReLU layers followed by a final 3×3 conv with no activation. Outputs `(B, 2, H, W)`: the `(Δy, Δx)` sub-pixel offset from each pixel to the nearest spot center. These offsets are only meaningful at pixels where `p(spot)` is high.

### Training

[train.py](torch_spots/detection/train.py) drives training with SGD (Nesterov momentum, weight decay) and an exponential learning rate scheduler.

[loss.py](torch_spots/detection/loss.py) defines `DotNetLosses`, a combined loss with two terms weighted 5:1 (classification:regression):

- **Classification loss**: weighted categorical cross-entropy (or focal loss) over all pixels, with per-class weights inversely proportional to class frequency to compensate for the extreme imbalance between background and spot pixels.
- **Regression loss**: smooth L1 loss computed only on pixels that are near a true spot center (within `d_pixels` pixels). This restricts the regression signal to the pixels where the offset is meaningful.

[loader.py](torch_spots/detection/loader.py) handles data loading. `SpotDataset` converts raw point-list annotations `(y, x)` into dense label arrays using a Euclidean distance transform to assign each pixel its signed offset `(Δy, Δx)` to the nearest spot.

### Inference

[inference.py](torch_spots/detection/inference.py) provides the `SpotDetection` class with two main entry points:

- **`predict_transforms(X)`** — returns the raw `detections` and `offsets` maps for further custom processing.
- **`predict_points(X)`** — runs the full pipeline and returns a list of sub-pixel `[y, x]` coordinates for all detected spots.

The inference pipeline:

1. **Preprocess** ([utils.py](torch_spots/detection/utils.py)): percentile-clip at 99.9% (per channel) then min-max normalize to `[0, 1]`.
2. **Tile**: split large images into 128×128 tiles. A spline window function blends overlapping tile outputs during reassembly to avoid boundary artifacts.
3. **Infer**: run SpotNet on batches of tiles with `torch.inference_mode()`.
4. **Untile**: reassemble tiled predictions back to original image dimensions, tracking per-tile `(x_offset, y_offset)` so that spot coordinates are correctly placed in the full image.
5. **Postprocess**: find local maxima in the spot probability map with `skimage.feature.peak_local_max` (above a `threshold`, minimum `min_distance` apart), then add the predicted `(Δy, Δx)` offset at each peak to yield sub-pixel coordinates.

---

## Decoding Branch (`torch_spots/decoding/`)

The decoding branch assigns gene identities to detected spots in multiplexed FISH imaging. Each spot has been imaged across multiple rounds and channels, producing an intensity vector that is matched against a known binary codebook.

### Input format

`spots_intensities_vec`: a `[num_spots, rounds × channels]` array of intensity values (normalized to `[0, 1]` for Relaxed Bernoulli mode). Each row encodes which (round, channel) combinations lit up for a given spot.

`df_barcodes`: a pandas DataFrame codebook where the first column is the gene name and the remaining `rounds × channels` columns hold the binary `{0, 1}` barcode for each gene.

### Probabilistic model

[decoding_utils.py](torch_spots/decoding/decoding_utils.py) implements a **mixture of Relaxed Bernoulli distributions** using [Pyro](https://pyro.ai/). The model treats each spot's intensity vector as a draw from one of K mixture components — one per gene barcode plus a `Background` class.

The **Relaxed Bernoulli** is a continuous relaxation of the Bernoulli distribution parameterized by a `probs` (location, derived from the binary barcode) and a `temperature` (sharpness). This allows the model to handle soft probability inputs rather than hard 0/1 calls, making it robust to imaging noise.

Model parameters:

- **weights** `w`: K-dimensional simplex, mixture proportions across all gene categories.
- **sigma**: controls the location of each distribution component relative to the ideal barcode (how much probability mass sits at the "on" vs "off" state).
- **temperature**: controls the sharpness of each Relaxed Bernoulli.

The `params_mode` argument controls whether sigma and temperature are shared globally or vary per round/channel:

| Mode      | Parameters            | Description                                              |
|-----------|-----------------------|----------------------------------------------------------|
| `'2'`     | 2 total               | One sigma + temperature for each of {0,1} barcode states |
| `'2*R'`   | 2 × rounds            | Parameters vary per imaging round                        |
| `'2*C'`   | 2 × channels          | Parameters vary per imaging channel                      |
| `'2*R*C'` | 2 × rounds × channels | Per-(round, channel) parameters (default)                |

### Inference procedure

Inference uses **Stochastic Variational Inference (SVI)** with an `AutoDelta` guide (MAP estimation) and Adam optimizer, run for `num_iter` iterations on mini-batches.

After training, the fitted parameters are used in an **E-step** (`rb_e_step`) to compute the posterior probability `P(gene k | spot i)` for every spot and every gene, yielding a `[num_spots, K]` class probability matrix.

### Postprocessing

[decode.py](torch_spots/decoding/decode.py) wraps the model in the `SpotDecoding` class and applies three post-inference steps:

1. **Probability thresholding**: spots whose maximum class probability falls below `pred_prob_thresh` (default 0.95) are relabeled as `Unknown`.

2. **Error rescue** (`rescue_errors=True` by default): spots assigned as `Background` or `Unknown` are re-examined. Their rounded intensity vector is compared against every barcode using Hamming distance. If exactly one barcode is at Hamming distance 1 (a single-bit error), the spot is rescued and assigned to that gene. The source field is set to `'error rescue'`.

3. **Mixed spot rescue** (`rescue_mixed`, off by default): for low-confidence spots, the model tests whether the spot could be the sum of two overlapping barcodes. If the residual intensity (after zeroing the predicted barcode's positions) has Hamming distance 1 to another barcode, a second entry is added to the output for the same spot index with source `'mixed rescue'`.

### Output

`SpotDecoding.predict()` returns a dictionary:

| Key              | Description                                               |
|------------------|-----------------------------------------------------------|
| `spot_index`     | Index of each spot (may have duplicates for mixed rescue) |
| `predicted_id`   | 1-indexed gene ID                                         |
| `predicted_name` | Gene name (`'Background'`, `'Unknown'`, or a gene name)   |
| `probability`    | Maximum posterior probability                             |
| `source`         | `'prediction'`, `'error rescue'`, or `'mixed rescue'`     |

---

## End-to-end usage

The two branches are designed to run in sequence. The detection branch localizes spot coordinates and extracts per-(round, channel) intensity vectors; the decoding branch takes those intensity vectors and returns gene assignments.

```python
from torch_spots.detection.inference import SpotDetection
from torch_spots.decoding.decode import SpotDecoding

# 1. Detect spots — returns sub-pixel [y, x] coordinates
detector = SpotDetection(model_path='path/to/spotnet.pth')
transforms = detector.predict_transforms(X)           # raw output maps
intensities, coords = detector.get_spot_intensities(transforms)

# 2. Decode gene identities from intensity vectors
app = SpotDecoding(df_barcodes=codebook, rounds=10, channels=2)
result = app.predict(intensities)
# result['predicted_name'] → array of gene names per spot
```
