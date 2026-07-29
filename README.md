# torch-spots

torch-spots is a PyTorch pipeline. It detects and decodes fluorescent spots in multiplexed imaging data. The library has two independent branches. The **detection** branch finds where spots are. The **decoding** branch identifies which gene each spot represents.

---

## Detection Branch (`torch_spots/detection/`)

The detection branch finds individual fluorescent spots in microscopy images with sub-pixel precision. It uses a fully-convolutional neural network called **SpotNet**. SpotNet predicts a spot probability map and a sub-pixel offset map together.

### Architecture

**SpotNet** ([dotnet.py](torch_spots/detection/dotnet.py)) has three parts.

1. **BNFeatureNetSkip2D backbone** ([modules.py](torch_spots/detection/modules.py)) is a sequential-refinement feature extractor. It runs `n_skips + 1` sub-networks (`BNFeatureNet2D`) in series. The first subnet processes the normalized input image alone. Each later subnet receives the original image joined with the previous subnet's output. This skip-connection design keeps raw pixel evidence visible at every stage of refinement.

   Each `BNFeatureNet2D` is a dilated convolutional stack. It is built from `Conv2D → BatchNorm → ReLU` blocks, plus a final "dense" conv that captures the remaining receptive field. With `receptive_field=13` (the default), the stack has 2 conv layers plus 1 dilated pass before the dense conv. `ReflectPadConv2d` preserves the spatial dimensions `(H, W)` exactly, throughout the stack.

   The table below shows the shape flow through the backbone, for the default config with `n_skips=3`.

   ```text
   Iteration 0: (B, 1, H, W)    → BNFeatureNet2D → (B, 128, H, W)
   Iteration 1: (B, 129, H, W)  → BNFeatureNet2D → (B, 128, H, W)
   Iteration 2: (B, 129, H, W)  → BNFeatureNet2D → (B, 128, H, W)
   Iteration 3: (B, 129, H, W)  → BNFeatureNet2D → (B, 128, H, W)
   ```

2. **ClassificationHead** uses two 1×1 convolutions (`Conv → BN → ReLU → Conv`), followed by softmax. It outputs `(B, 2, H, W)`. Channel 0 holds `p(background)`. Channel 1 holds `p(spot center)`.

3. **OffsetRegressionHead** uses four 3×3 Conv+ReLU layers, followed by a final 3×3 conv with no activation. It outputs `(B, 2, H, W)`, the `(Δy, Δx)` sub-pixel offset from each pixel to the nearest spot center. These offsets carry meaning only at pixels where `p(spot)` is high.

### Training

[train.py](torch_spots/detection/train.py) runs training with SGD (Nesterov momentum, weight decay) and an exponential learning rate scheduler.

[loss.py](torch_spots/detection/loss.py) defines `DotNetLosses`. This combined loss has two terms, weighted 5:1 (classification:regression).

- **Classification loss**: a weighted categorical cross-entropy (or focal loss) over all pixels. Per-class weights are inversely proportional to class frequency. This compensates for the extreme imbalance between background and spot pixels.
- **Regression loss**: a smooth L1 loss, computed only on pixels near a true spot center (within `d_pixels` pixels). This limits the regression signal to the pixels where the offset carries meaning.

[loader.py](torch_spots/detection/loader.py) loads the data. `SpotDataset` converts raw point-list annotations `(y, x)` into dense label arrays. It uses a Euclidean distance transform to assign each pixel its signed offset `(Δy, Δx)` to the nearest spot.

### Inference

[inference.py](torch_spots/detection/inference.py) provides the `SpotDetection` class, with two main entry points.

- **`predict_transforms(X)`** returns the raw `detections` and `offsets` maps, for further custom processing.
- **`predict_points(X)`** runs the full pipeline and returns a list of sub-pixel `[y, x]` coordinates for all detected spots.

The inference pipeline works in five steps.

1. **Preprocess** ([utils.py](torch_spots/detection/utils.py)): clip each channel at the 99.9th percentile, then apply min-max normalization to `[0, 1]`.
2. **Tile**: split large images into 128×128 tiles. A spline window function blends overlapping tile outputs during reassembly. This avoids boundary artifacts.
3. **Infer**: run SpotNet on batches of tiles, under `torch.inference_mode()`.
4. **Untile**: reassemble the tiled predictions into the original image dimensions. The pipeline tracks each tile's `(x_offset, y_offset)`, so it places spot coordinates correctly in the full image.
5. **Postprocess**: find local maxima in the spot probability map with `skimage.feature.peak_local_max` (above a `threshold`, at least `min_distance` apart). Then add the predicted `(Δy, Δx)` offset at each peak. This yields sub-pixel coordinates. The implementation was deeply inspired by [PoSTcode](https://github.com/gerstung-lab/postcode).

---

## Decoding Branch (`torch_spots/decoding/`)

The decoding branch assigns gene identities to detected spots, in multiplexed FISH imaging. Each spot has an image from multiple rounds and channels. Together these produce an intensity vector. The pipeline matches this vector against a known binary codebook.

### Input format

`spots_intensities_vec` is a `[num_spots, rounds × channels]` array of intensity values, normalized to `[0, 1]` for Relaxed Bernoulli mode. Each row encodes which (round, channel) combinations lit up for a given spot.

`df_barcodes` is a pandas DataFrame codebook. The first column holds the gene name. The remaining `rounds × channels` columns hold the binary `{0, 1}` barcode for each gene.

### Probabilistic model

[decoding_utils.py](torch_spots/decoding/decoding_utils.py) implements a **mixture of Relaxed Bernoulli distributions**, using [Pyro](https://pyro.ai/). The model treats each spot's intensity vector as a draw from one of K mixture components, one per gene barcode plus a `Background` class.

The **Relaxed Bernoulli** is a continuous relaxation of the Bernoulli distribution. It takes a `probs` parameter (the location, derived from the binary barcode) and a `temperature` parameter (the sharpness). This lets the model handle soft probability inputs, rather than hard 0/1 calls, so the model stays robust to imaging noise.

The model has three parameters.

- **weights** `w`: a K-dimensional simplex, the mixture proportions across all gene categories.
- **sigma**: controls the location of each distribution component, relative to the ideal barcode (how much probability mass sits at the "on" state versus the "off" state).
- **temperature**: controls the sharpness of each Relaxed Bernoulli.

The `params_mode` argument controls whether sigma and temperature stay shared globally, or vary per round or channel.

| Mode      | Parameters            | Description                                              |
|-----------|-----------------------|------------------------------------------------------------|
| `'2'`     | 2 total               | One sigma and one temperature, for each of {0,1} barcode states |
| `'2*R'`   | 2 × rounds            | Parameters vary per imaging round                        |
| `'2*C'`   | 2 × channels          | Parameters vary per imaging channel                      |
| `'2*R*C'` | 2 × rounds × channels | Parameters vary per (round, channel) pair (the default)  |

### Inference procedure

Inference uses **Stochastic Variational Inference (SVI)**, with an `AutoDelta` guide (MAP estimation) and the Adam optimizer, on mini-batches, for `num_iter` iterations.

After training, an **E-step** (`rb_e_step`) uses the fitted parameters to compute the posterior probability `P(gene k | spot i)`, for every spot and every gene. This yields a `[num_spots, K]` class probability matrix.

### Postprocessing

[decode.py](torch_spots/decoding/decode.py) wraps the model in the `SpotDecoding` class. This class applies three steps after inference.

1. **Probability thresholding**: if a spot's maximum class probability falls below `pred_prob_thresh` (the default is 0.95), the pipeline relabels the spot as `Unknown`.

2. **Error rescue** (`rescue_errors=True` by default): the pipeline re-examines spots assigned as `Background` or `Unknown`. It compares each spot's rounded intensity vector against every barcode, by Hamming distance. If exactly one barcode sits at Hamming distance 1 (a single-bit error), the pipeline rescues the spot and assigns it to that gene. The source field then reads `'error rescue'`.

3. **Mixed spot rescue** (`rescue_mixed`, off by default): for low-confidence spots, the model tests whether the spot could be the sum of two overlapping barcodes. First the pipeline zeros the predicted barcode's positions in the residual intensity. If this residual then has Hamming distance 1 to another barcode, the pipeline adds a second entry to the output, for the same spot index, with source `'mixed rescue'`.

### Output

`SpotDecoding.predict()` returns a dictionary with five keys.

| Key              | Description                                               |
|------------------|-------------------------------------------------------------|
| `spot_index`     | Index of each spot (may repeat, for mixed rescue)          |
| `predicted_id`   | 1-indexed gene ID                                         |
| `predicted_name` | Gene name (`'Background'`, `'Unknown'`, or a gene name)   |
| `probability`    | Maximum posterior probability                             |
| `source`         | `'prediction'`, `'error rescue'`, or `'mixed rescue'`     |

---

## End-to-end usage

Run the two branches in sequence. The detection branch finds spot coordinates and extracts per-(round, channel) intensity vectors. The decoding branch takes these intensity vectors and returns gene assignments.

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

## Installation issues

If your GPU's CUDA drivers do not support the default PyTorch version, you get an error at the start of inference. To fix this, install a PyTorch version that matches your GPU.