# deepcell_spots → torch_spots: differences

How the PyTorch port (`torch_spots/`) differs from the original TensorFlow package (deepcell_spots 0.4.2, with deepcell-tf 0.12.10 for the backbone, losses and tiling). The two were compared area by area (model, training, inference, metrics, decoding), and numerical claims were checked with small scripts.

**Reference configuration.** Where the original has more than one answer, the port follows the settings recorded for the published SpotDetection-8 model (the weights deepcell_spots ships), not the code defaults. In particular the reference training length is 10 epochs, and the in-model normalisation is off (the published model's normalisation layers are all pass-through).

## 1. Differences that were found and fixed

Each of these changed the numbers the network computes or is trained on. They were corrected so the port reproduces the original.

| Area | Difference found | Now |
|---|---|---|
| Model | The 2×2 max-pool after the second conv block of every sub-network was missing | Added (stride 1, dilated) |
| Model | The original pads once per sub-network, so each sub-network sees ±6 px. The port padded before every conv, giving an off-centre window ([−20, +24] instead of ±24 for the full stack) | One reflect-pad of 6 px per sub-network, then unpadded convs. Measured windows are symmetric |
| Model | Weights started from PyTorch defaults, not Keras `he_normal` / N(0, 0.01) with zero biases | Keras-style initialisation |
| Model | BatchNorm used PyTorch defaults (momentum 0.1, eps 1e-5) | Keras-equivalent (momentum 0.01, eps 1e-3) |
| Model | The in-model `'std'` normalisation was implemented as whole-image standardisation, not deepcell's local 13×13 window | The local version is implemented and kept as an option. The default is off, as in the published model |
| Training | Weighted cross-entropy summed over image width, not over classes, and used differently scaled class weights | Deepcell's formula, summed over classes. Focal loss uses the same weights |
| Training | Augmentation rotated points and pixels about centres 0.5 px apart, so labels were up to 1.4 px off. Fill value −1 was outside the data range | One shared transform about the true centre, edge-value fill. Defaults follow the published recipe |
| Training | The "best" checkpoint was overwritten every epoch | Written only when validation improves |
| Training | Keras `decay=1e-6` (learning-rate decay) was treated as weight decay. The original's L2 penalty on kernels was missing. Loss weighting, learning rate and gradient clipping differed | Learning-rate decay and explicit kernel L2 penalty restored, with no clipping. Loss weights follow the reference |
| Training | Epochs, batch size and handling of the last partial batch differed | Match the reference (10 epochs, batch size 1, partial batch dropped) |
| Inference / eval | Training used inputs in [−0.5, 0.5] with a two-sided clip, while inference and evaluation used [0, 1] with an upper-only clip over non-zero pixels | Training, evaluation and inference share one preprocessing function. The original application's 0.01/99.9 clip is available as `clip=True` |
| Metrics | Matching used Hungarian assignment (a close pair can be given up to lower total cost), τ = 2 px, RMSE √2× the original's, and a Chamfer distance over all points | The original's definitions are the reference metrics: mutual-nearest-neighbour matching, τ = 1 px, per-coordinate RMSE, Jaccard, macro averages. The Hungarian-based numbers remain as clearly labelled supplementary metrics, never used to judge a result |
| Metrics | `bias_x` / `bias_y` labels were swapped | Corrected |
| Decoding | The `distribution` argument was accepted but ignored (Gaussian and Bernoulli silently ran as Relaxed Bernoulli) | Argument removed, so passing it is an error |
| Decoding | Candidate spots were hard-wired, the output table had no coordinates or probabilities, and rows could misalign after mixed rescue | `get_spot_probabilities` hands over locations and probabilities with a tunable candidate threshold. `predict` returns one table and can drop Background, Unknown and Blank rows with `genes_only=True` |

## 2. Differences that remain

These are known and have not been changed.

### Inference

- **Tiled `predict_points` loses spots at tile edges.** Peak finding runs on each 128×128 tile, so spots on the first or last row or column of a tile are excluded. The original stitches the tiles first and finds peaks once on the whole image.
- **No tile overlap or blending.** The tile stride is 1.0 (tiles abut), where the original uses 0.75 with spline blending. The tiling code itself is numerically identical to the original's, and only the default stride differs.
- **Zero padding at the outer image margin** (`pad_mode='constant'`), where the original reflects.
- **Default probability threshold.** `predict_points` uses 0.99, the evaluation operating point is 0.993 (chosen on the validation set), and the original `SpotDetection.predict` uses 0.95.
- **Batch identity is lost.** For more than one image per call, points from all images are returned in one array without an image index. The single-image calls used in evaluation are unaffected.
- **Images smaller than 128 px** raise an error in `predict_points`, and an empty result is returned as a tuple where other results are a plain array.
- **Evaluation skips tiling.** It runs each whole image through the network. This is fine for a fully-convolutional model on 128×128 test images, but evaluation numbers do not test the tiling path.
- `detection/postprocessing.py` is not used by anything and is inconsistent with the live code in `utils.py` and `inference.py`.

### Decoding

- **Candidate spots.** The port decodes local peaks above a candidate threshold (default 0.9). The original decodes every max-projection pixel above 0.01, so the learned mixture weights and the Unknown threshold are calibrated on a different population.
- **No deduplication after decoding.** The original keeps one row per local maximum among decoded spots. The port keeps every candidate (after the optional `genes_only` filter).
- **No skipping of empty rounds.** The original zeroes image channels whose codebook column is all zero. The port does not. This cannot matter for the MERFISH codebook used here, which has no such column.
- **Device and random numbers.** The original fits on the GPU when one is available, and the port fits on the CPU. The seed is the same, but CPU and GPU random streams differ, so results are statistically equivalent but not bit-identical.
- The `[y, x]` convention is used throughout the port. The original's decoding table calls the row axis `x`.

### Intentional or harmless

- Redundant biases before BatchNorm are dropped. BatchNorm absorbs them, so outputs are the same, but parameter counts and checkpoint key names differ from the original.
- Outputs are a dictionary (`detections`, `offsets`) in channels-first layout, where the original returns a list (regression first) in channels-last. TensorFlow weights cannot be loaded by position.
- Validation points are not filtered to the image bounds, and train and validation come from pre-split files instead of a random 80/20 split.
- The original's probability renormalisation in the loss is omitted. It has no effect because the output is a softmax.
- Images with no spots produce a two-channel target, where the original's one-hot step would return a single channel.
- The port reports extra detection metrics (mean, median, 95th-percentile and maximum distance, bias, Chamfer) beside the reference set.
- The original's own decoding code has one quirk that the port inherits: with columns ordered round by round, the axis called "channel" is really the first versus second half of the columns. It is harmless for the default `'2*R*C'` parameterisation and would group parameters incorrectly for `'2*R'` / `'2*C'` in both versions.

## 3. Verified to match

- **Backbone:** the convolution geometry for receptive field 13, the number of sub-networks and filters, skip wiring (the image is concatenated first, then the previous output), and the classification and regression heads.
- **Losses and targets:** smooth-L1, the regression mask, the regularised classification loss (unused by default), target construction (`[y, x]` order, offsets as point minus pixel index, one-hot channel 0 = background) and the out-of-bounds filter.
- **Tiling code:** `tile_image`, `untile_image`, the spline window and the small-image path agree numerically with deepcell-toolbox.
- **Peak finding and offsets:** same arguments, same offset decoding and sign.
- **Decoding core:** the generative model, guide, optimiser settings, E-step, codebook handling and both rescue steps are line-for-line copies of the original for the default Relaxed Bernoulli path.
