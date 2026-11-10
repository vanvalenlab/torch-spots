# torch-spots

torch-spots is a PyTorch pipeline. It detects and decodes fluorescent spots in multiplexed imaging data. The library has two independent branches. The **detection** branch finds where spots are. The **decoding** branch identifies which gene each spot represents.

---

## Detection Branch (`torch_spots/detection/`)

The detection branch finds individual fluorescent spots in microscopy images with sub-pixel precision. It uses a fully-convolutional neural network called **SpotNet**. SpotNet predicts a spot probability map and a sub-pixel offset map together.

### Architecture

**SpotNet** ([dotnet.py](torch_spots/detection/dotnet.py)) has three parts.

1. **BNFeatureNetSkip2D backbone** ([modules.py](torch_spots/detection/modules.py)) is a sequential-refinement feature extractor. It runs `n_skips + 1` sub-networks (`BNFeatureNet2D`) in series. The first subnet processes the normalized input image alone. Each later subnet receives the original image joined with the previous subnet's output. This skip-connection design keeps raw pixel evidence visible at every stage of refinement.

   Each `BNFeatureNet2D` mirrors deepcell's `bn_feature_net_2D` with `receptive_field=13`. It mirror-pads its input once, by 6 pixels on every side. It then applies `4×4 conv (32) → BatchNorm → ReLU → 3×3 conv (32) → BatchNorm → ReLU → 2×2 max-pool (stride 1) → 4×4 conv with dilation 2 (128) → BatchNorm → ReLU`. No layer pads after the first reflection, so the output is the same height and width as the input, and each output pixel depends on exactly the 13×13 window centred on it. The full 4-sub-network stack sees ±24 pixels. Every BatchNorm uses Keras's constants (running statistics updated with weight 0.01, epsilon 1e-3).

   **Normalisation inside the model.** By default (`norm_method=None`) the model does no normalisation of its own. This matches the published model, whose five normalisation layers are all pass-through; the input is only min-max scaled to [0, 1] before it reaches the network. Two options remain for experiments: `'std'` (as deepcell's `'std'`: subtract the mean of a 13×13 window around each pixel and divide by that window's standard deviation) and `'whole_image'` (one mean and standard deviation per image). In the ablation rounds, `'std'` produced probabilities that peak at much lower thresholds than the published model's.

   **Initial weights.** Backbone and classification-head kernels start from Keras's `he_normal` (a normal distribution cut off at 2 standard deviations, with standard deviation √(2 / fan_in)). Regression-head kernels start from a normal with standard deviation 0.01. All biases start at 0.

   Sources: deepcell-tf 0.12.10 (`model_zoo/featurenet.py`, `layers/normalization.py`, `layers/pooling.py`) and deepcell_spots 0.4.2 (`dotnet.py`). Differences from the original that remain are listed in [PORT_DIFFERENCES.md](PORT_DIFFERENCES.md).

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

[train.py](torch_spots/detection/train.py) trains the detector. Its defaults are the settings recorded for the published SpotDetection-8 model (the weights deepcell_spots ships). Where the model files record nothing, the paper's Methods are used, and only after that the deepcell_spots 0.4.2 code defaults. Order of authority: the model's saved files, the application's `model_metadata`, the paper, the code defaults.

| Setting | Default | Source |
|---|---|---|
| Augmentation | horizontal and vertical flips (each probability 0.5), rotation uniform in ±180°, zoom uniform in 0.8–1.2 | flips: `training.py:38-42`; rotation and zoom: paper Methods |
| Loss | 1/6 · regression + 5/6 · classification, plus the L2 penalty | the model's saved `loss_weights` |
| Classification loss | deepcell weighted cross-entropy (class weights `T / (n·(count+1))` per batch) | `deepcell/losses.py` |
| Optimizer | SGD, Nesterov momentum 0.9, learning rate 0.01, no gradient clipping | the model's saved optimizer; `training.py:29-37` |
| Learning-rate schedule | `0.01 · 0.99^epoch / (1 + 1e-6 · step)` | application `model_metadata` (`lr_decay` 0.99); saved rate 0.01·0.99⁴ |
| L2 penalty | `1e-5 · Σw²` on the backbone convolution kernels and the two classification-head kernels (not biases, BatchNorm or the regression head); no weight decay | `featurenet.py:152,201`, `dotnet.py:62,66` |
| Input normalisation inside the model | none (`norm_method=None`) | all five saved normalisation layers are pass-through |
| Length | 10 epochs, batch size 1 (849 steps per epoch, 8,490 in total, on the v1.1 training split) | the saved optimizer state (0.01 · 0.99⁴) shows the shipped weights came from epoch 4; `model_metadata` plans 50 epochs, which the shipped weights never reached. Measured: mean test F1 0.8867 at 10 epochs, 0.7607 at 50 |
| Checkpoints | the best checkpoint is the epoch with the lowest validation objective (including the L2 penalty); the last epoch is saved separately | `training.py:194-196` |

The earlier defaults (flips only, 1 : 1 loss, decay 0.95, 10 epochs, `'std'` normalisation) can be reproduced by saving the following as a JSON file and passing it with `--config`:

   ```json
   {
     "augmentation": {"flip": true, "rotation_range": 0.0, "zoom_range": null},
     "loss": {"regression_weight": 1.0, "classification_weight": 1.0, "focal": false},
     "optimizer": {"lr": 0.01, "momentum": 0.9, "step_decay": 1e-06, "epoch_decay": 0.95, "kernel_l2": 1e-05, "grad_clip": null},
     "model": {"norm_method": "std"},
     "epochs": 10,
     "batch_size": 1
   }
   ```

Fifty epochs (`{"epochs": 50}`, the length `model_metadata` plans) trains about five times longer and scored worse; it is recorded as a non-reference setting.

**Changing settings (ablations).** Pass `--config` with a small JSON file. It is merged over the defaults, and `--seed`, `--data-path` and `--device` still apply on top. For example, to turn rotation off and weight the two loss terms equally:

```json
{
  "augmentation": {"rotation_range": 0},
  "loss": {"regression_weight": 1.0, "classification_weight": 1.0}
}
```

```bash
python -m torch_spots.detection.train --config ablation.json --seed 0 --device cuda:0
```

The settings you can change are `augmentation` (`flip`, `rotation_range`, `zoom_range`), `loss` (`regression_weight`, `classification_weight`, `focal`), `optimizer` (`lr`, `momentum`, `step_decay`, `epoch_decay`, `kernel_l2`, `grad_clip`), `model` (`norm_method`: `'std'`, `'whole_image'` or `null`), `epochs` and `batch_size`. A misspelled setting name is an error, not silently ignored.

**What gets recorded.** `training_config.json`, saved next to the checkpoint, holds every value used, the seed and the dataset provenance. Its `non_reference` field lists each setting that differs from the defaults above, with its value, so a run with no differences has an empty list. It also records the model settings, a `model_architecture_version` (see *Checkpoint compatibility* below), and `best_epoch` (counted from 0) and `best_val_objective`, which identify the checkpoint saved as `saved_model_best_dict.pth`.

Checkpoints trained under the earlier defaults (see above) stay loadable, because their `training_config.json` records their settings. They are not reference runs, and their results are not comparable with new ones.

[loss.py](torch_spots/detection/loss.py) defines `DotNetLosses`, which adds the regression and classification terms with the weights above.

- **Classification loss**: a weighted categorical cross-entropy (or focal loss) over all pixels. Per-class weights are inversely proportional to class frequency, computed for each batch. This compensates for the extreme imbalance between background and spot pixels.
- **Regression loss**: a smooth L1 loss, computed only on pixels near a true spot center (within `d_pixels` pixels). This limits the regression signal to the pixels where the offset carries meaning.

[loader.py](torch_spots/detection/loader.py) loads the data. `SpotDataset` converts raw point-list annotations `(y, x)` into dense label arrays. It uses a Euclidean distance transform to assign each pixel its signed offset `(Δy, Δx)` to the nearest spot.

### Inference

[inference.py](torch_spots/detection/inference.py) provides the `SpotDetection` class, with two main entry points.

- **`predict_transforms(X)`** returns the raw `detections` and `offsets` maps, for further custom processing.
- **`predict_points(X)`** runs the full pipeline and returns a list of sub-pixel `[y, x]` coordinates for all detected spots.

The inference pipeline works in five steps.

1. **Preprocess** ([loader.py](torch_spots/detection/loader.py), `preprocess_image`): min-max scale each image and channel to `[0, 1]`, on the full image before tiling. This is the same function training and evaluation use, with no percentile clipping and no offset. On SpotNet v1.1 every stored image already spans exactly `[0, 1]`, so this leaves those images unchanged, as in the original training. The original application also clipped each image at the 0.01st and 99.9th percentiles before scaling. That is available with `clip=True` on `predict_transforms` and `predict_points`, and is off by default so that inference sees the same kind of input as training.
2. **Tile**: split large images into 128×128 tiles. A spline window function blends overlapping tile outputs during reassembly. This avoids boundary artifacts.
3. **Infer**: run SpotNet on batches of tiles, under `torch.inference_mode()`.
4. **Untile**: reassemble the tiled predictions into the original image dimensions. The pipeline tracks each tile's `(x_offset, y_offset)`, so it places spot coordinates correctly in the full image.
5. **Postprocess**: find local maxima in the spot probability map with `skimage.feature.peak_local_max` (above a `threshold`, at least `min_distance` apart). Then add the predicted `(Δy, Δx)` offset at each peak. This yields sub-pixel coordinates. The implementation was deeply inspired by [PoSTcode](https://github.com/gerstung-lab/postcode).

**Checkpoint compatibility.** Each training run writes `training_config.json` next to its checkpoint, including `model_architecture_version` (currently 2) and the `model` settings used. `SpotDetection(model_path=...)` reads that file, builds the network from the recorded settings, and loads the weights strictly. It stops with an error if the file or the version is missing or different. Checkpoints trained before version 2 (including every earlier benchmark round and the published `torch-spots_2026-07-29.pth`) therefore cannot be loaded by this code. To use one, check out the commit it was trained at. For the same reason `SpotDetection()` with no `model_path` stops with an error until a model trained on the current architecture is published.

```python
from torch_spots.detection.inference import SpotDetection

detector = SpotDetection(model_path="data/model/<run>/saved_model_best_dict.pth", device="cuda:0")
points = detector.predict_points(images)                  # images: (B, 1, H, W), raw values
points = detector.predict_points(images, clip=True)       # with the original application's percentile clip
```

### Benchmark evaluation protocol

Detection results are scored against the **published SpotDetection-8 model**, run through its own deepcell_spots application and scored by this same pipeline on the same 94 test images. (The F1 = 0.83 figure from earlier versions of this protocol was never found in the paper, and the published model scores far lower at threshold 0.95, so it is no longer the target.) The scoring uses the **reference metric set** ([reference_metrics.py](torch_spots/detection/reference_metrics.py)), which reproduces the deepcell_spots 0.4.2 definitions: mutual-nearest-neighbour matching at τ = 1 px, precision/recall/F1, Jaccard, per-coordinate RMSE and d_md over matched pairs, macro-averaged over all test images. The port's other metrics ([metrics.py](torch_spots/detection/metrics.py)) are *additional* diagnostics and are never used to judge a benchmark or a divergence. The reference operating point is a probability threshold chosen once on the validation split **0.993**, selected on the validation split with the three round 6 seeds (`data/eval/round6_threshold_selection.json`; it is `EVAL_THRESHOLD` in `torch_spots/detection/inference.py`). The old 0.95 is not used, minimum peak distance 1 px, and τ = 1 px.

Benchmark data is **SpotNet v1.1** (`train.npz`, `val.npz`, `test.npz` in `~/.deepcell/spots/`). Training and evaluation both refuse stores that do not match the pinned image count, spot count and checksum.

1. **Convert the data** (the `.npz` files are never modified; stores record source checksum, version and counts):

   ```bash
   python -m torch_spots.preprocess            # or --data-dir PATH
   ```

2. **Train with a seed.** A benchmark configuration is trained with seeds 0, 1 and 2:

   ```bash
   python -m torch_spots.detection.train --seed 0 --device cuda:0
   ```

   The seed and the dataset provenance are written to `training_config.json` next to the checkpoint.

3. **Evaluate each seed.** This scores all 94 test images and writes a run directory with `per_image.csv` and `summary.json`. Checkpoints without recorded training provenance are marked `unknown_provenance` and are not benchmark-eligible.

   ```bash
   python -m torch_spots.detection.eval --model-path data/model/<run>/saved_model_best_dict.pth \
       --device cuda:0 --output-dir data/eval/seed0
   ```

   An operating-point sweep (always containing the reference point, reported separately) is available with `python -m torch_spots.detection.eval_sweep`.

   **Scoring the validation split.** Add `--split val` (to `eval` or `eval_sweep`) to score the pinned validation split (95 images) instead. This is how the reference threshold is chosen without looking at the test set. The store is checked against the pinned validation split just as the test store is, the summary records `"split": "val"`, and the result is never benchmark-eligible: `report` refuses it.

   ```bash
   python -m torch_spots.detection.eval --split val --model-path data/model/<run>/saved_model_best_dict.pth \
       --device cuda:0 --output-dir data/eval/seed0_val
   ```

4. **Build the 3-seed report.** This lists per-seed F1, mean and range, and a pass/fail line. The target is **one-sided**: the configuration reproduces the published model when its 3-seed mean F1 is at least the published model's F1 at the same operating point (scoring higher is not a failure). The published model's score is recorded in `torch_spots/detection/report.py` (`PUBLISHED_MODEL_F1`, with its provenance), and the report stops with an error if it has not been recorded. The report refuses runs with unknown provenance, mismatched seeds, a mismatched protocol, an operating point other than the reference, or a validation-split result:

   ```bash
   python -m torch_spots.detection.report report data/eval/seed0 data/eval/seed1 data/eval/seed2 \
       --output report.json
   ```

   **How the published score is measured.** SpotDetection-8 is run through its own deepcell_spots application (default preprocessing) in a separate TensorFlow environment, and its predicted points are written to `points.json`; `python -m torch_spots.detection.report published data/eval/published_spotdetection8/points.json` scores it on the pinned test split at every threshold and writes `summary.json` next to it.

5. **Judge a divergence** from the reference configuration by comparing two reports. An accuracy claim needs the candidate mean F1 to exceed the reference mean by more than the reference seed range. A quality-of-life claim needs a mean F1 no lower than the reference mean minus that range, plus a measured benefit:

   ```bash
   python -m torch_spots.detection.report compare reference_report.json candidate_report.json \
       --benefit-metric training_time_reduction --benefit-value 0.30
   ```

> **Note:** the `eval_results_*.csv` files from earlier versions are superseded. They used Hungarian matching at τ = 2 px, different RMSE/d_md definitions, different input preprocessing and a dataset of unverified version, so they are not comparable with results from this protocol.

---

## Decoding Branch (`torch_spots/decoding/`)

The decoding branch assigns gene identities to detected spots, in multiplexed FISH imaging. Each spot has an image from multiple rounds and channels. Together these produce one detection probability per image for each spot. The pipeline matches this vector of probabilities against a known binary codebook.

### Input format

`spot_probabilities` is a `[num_spots, rounds × channels]` array of detection probabilities, each between 0 and 1. These are the spot probabilities that detection assigns in each image (the `'probabilities'` entry of `get_spot_probabilities`), not raw pixel brightness. Column *i* must belong to codebook bit column *i*. Each row shows which (round, channel) combinations lit up for a given spot. The optional `locations` argument is the `[y, x]` array from the same hand-off (`'locations'`).

`df_barcodes` is a pandas DataFrame codebook. The first column holds the gene name. The remaining `rounds × channels` columns hold the binary `{0, 1}` barcode for each gene.

### Probabilistic model

[decoding_utils.py](torch_spots/decoding/decoding_utils.py) implements a **mixture of Relaxed Bernoulli distributions**, using [Pyro](https://pyro.ai/). The model treats each spot's probability vector as a draw from one of K mixture components, one per gene barcode plus a `Background` class.

The **Relaxed Bernoulli** is a continuous relaxation of the Bernoulli distribution. It takes a `probs` parameter (the location, derived from the binary barcode) and a `temperature` parameter (the sharpness). This lets the model handle soft probability inputs, rather than hard 0/1 calls, so the model stays robust to imaging noise.

The model has three parameters.

- **weights** `w`: a K-dimensional simplex, the mixture proportions across all gene categories.
- **sigma**: controls the location of each distribution component, relative to the ideal barcode (how much probability mass sits at the "on" state versus the "off" state).
- **temperature**: controls the sharpness of each Relaxed Bernoulli.

The `params_mode` argument (set when you create `SpotDecoding`) controls whether sigma and temperature stay shared globally, or vary per round or channel.

| Mode      | Parameters            | Description                                              |
|-----------|-----------------------|------------------------------------------------------------|
| `'2'`     | 2 total               | One sigma and one temperature, for each of {0,1} barcode states |
| `'2*R'`   | 2 × rounds            | Parameters vary per imaging round                        |
| `'2*C'`   | 2 × channels          | Parameters vary per imaging channel                      |
| `'2*R*C'` | 2 × rounds × channels | Parameters vary per (round, channel) pair (the default)  |

Known limitation: in the `'2*R'` and `'2*C'` modes, the bit columns that share a value are not the ones of one round or one channel, because of the order the columns are read in. With 10 rounds and 2 channels in `'2*R'`, columns 0 and 5 share a value, as do 1 and 6, and so on. Only the default `'2*R*C'` mode is unaffected. This is kept as it was, and not yet fixed.

### Inference procedure

Inference uses **Stochastic Variational Inference (SVI)**, with an `AutoDelta` guide (MAP estimation) and the Adam optimizer, on mini-batches, for `num_iter` iterations.

After training, an **E-step** (`rb_e_step`) uses the fitted parameters to compute the posterior probability `P(gene k | spot i)`, for every spot and every gene. This yields a `[num_spots, K]` class probability matrix.

### Postprocessing

[decode.py](torch_spots/decoding/decode.py) wraps the model in the `SpotDecoding` class. This class applies three steps after inference.

1. **Probability thresholding**: if a spot's maximum class probability falls below `pred_prob_thresh` (the default is 0.95), the pipeline relabels the spot as `Unknown`.

2. **Error rescue** (`rescue_errors=True` by default): the pipeline re-examines spots assigned as `Background` or `Unknown`. It compares each spot's rounded probability vector against every barcode, by Hamming distance. If exactly one barcode sits at Hamming distance 1 (a single-bit error), the pipeline rescues the spot and assigns it to that gene. The source field then reads `'error rescue'`.

3. **Mixed spot rescue** (`rescue_mixed`, off by default): for low-confidence spots, the model tests whether the spot could be the sum of two overlapping barcodes. First the pipeline zeros the predicted barcode's positions in the residual probabilities. If this residual then has Hamming distance 1 to another barcode, the pipeline adds a second entry to the output, for the same spot index, with source `'mixed rescue'`.

### Output

`SpotDecoding.predict()` returns one table (a pandas DataFrame), with one row per spot, in input order. If mixed rescue adds rows, they come at the end.

| Column           | Description                                               |
|------------------|-------------------------------------------------------------|
| `spot_index`     | Index of the input spot (repeats for a mixed-rescue row)   |
| `y`, `x`         | Row and column of the spot (only when you pass `locations`) |
| `predicted_id`   | 1-indexed gene ID                                         |
| `predicted_name` | Gene name (`'Background'`, `'Unknown'`, or a gene name)   |
| `probability`    | Maximum posterior probability (−1 for a mixed-rescue row) |
| `source`         | `'prediction'`, `'error rescue'`, or `'mixed rescue'`     |
| `prob_<name>`    | The spot's detection probability for each codebook bit, named after the codebook column (for example `prob_RS0015`) |

`y`, `x` and the `prob_*` values are looked up through `spot_index`, so a row added by mixed rescue has the location and probabilities of its own spot.

Set `genes_only=True` to drop every row labelled `Background`, `Unknown` or `Blank-*`. It is applied last, so it does not change which spots are rescued.

`predict` also takes `seed` (default 1), which sets which random group of spots each training step sees. The same input, settings and seed give the same table on the same device. After a run, `decoder.last_run` holds the settings used and the fitted `weights`, `sigma`, `temperature` and per-iteration `losses`.

------------------|-------------------------------------------------------------|
| `spot_index`     | Index of each spot (may repeat, for mixed rescue)          |
| `predicted_id`   | 1-indexed gene ID                                         |
| `predicted_name` | Gene name (`'Background'`, `'Unknown'`, or a gene name)   |
| `probability`    | Maximum posterior probability                             |
| `source`         | `'prediction'`, `'error rescue'`, or `'mixed rescue'`     |

---

## End-to-end usage

Run the two branches in sequence. The detection branch finds candidate spot locations and reads, at each one, the spot probability that detection assigned in every image of the stack. These are detection probabilities in [0, 1], not raw pixel brightness. The decoding branch takes those probabilities and returns gene assignments.

```python
from torch_spots.detection.inference import SpotDetection
from torch_spots.decoding.decode import SpotDecoding

# 1. Detect spots. X holds the N images of the stack, in the codebook's column order.
detector = SpotDetection(model_path='path/to/spotnet.pth')
transforms = detector.predict_transforms(X)           # raw output maps
handoff = detector.get_spot_probabilities(transforms,
                                          threshold=0.9,   # candidate threshold
                                          min_distance=1,
                                          pool_size=0)
# handoff['locations']      integer [y, x] (row, column) of each candidate, shape [n, 2]
# handoff['probabilities']  detection probability per image at each candidate, shape [n, N]
# handoff['settings']       the threshold, min_distance and pool_size that were used

# 2. Decode gene identities from the probabilities
app = SpotDecoding(df_barcodes=codebook, rounds=10, channels=2)
table = app.predict(handoff['probabilities'], handoff['locations'])
# table['predicted_name'] → gene name per spot; table has y, x, source, and one prob_* column per bit
```

The candidate threshold (default 0.9) is separate from the 0.993 threshold used to evaluate detection. Candidates are the local maxima of the highest probability across the images, so a spot that is bright in only one image is still a candidate. The decoding model then decides whether it is a barcode, `Background` or `Unknown`.

## Installation issues

If your GPU's CUDA drivers do not support the default PyTorch version, you get an error at the start of inference. To fix this, install a PyTorch version that matches your GPU.

## Joining the Hugging Face organzation and generating a Hugging Face access token

To use our models, you first need to create a free Hugging Face account and join the Van Valen Lab organization.

Navigate to the [Hugging Face org site](https://huggingface.co/vanvalenlab) and join the organization. You will be automatically added as a "Read" member. You will now have access to the model weights (which are private and only avaialble to members of the org).

Next, you will need to generate your Access Token. Navigate to the [Access Tokens](https://huggingface.co/settings/tokens) site and generate one. You will then export it as the environmental variable `HF_TOKEN`. This way, when you instantiate the model for the first time, the HF CLI will automatically check the value of this environmental variable and log you in before donwloading the model weights.