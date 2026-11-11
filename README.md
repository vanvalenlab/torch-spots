# torch-spots

torch-spots is a PyTorch pipeline for detecting and decoding fluorescent spots in multiplexed imaging data. It has two independent branches:

- **Detection** finds where spots are, with sub-pixel precision.
- **Decoding** identifies which gene each spot represents.

---

## Detection (`torch_spots/detection/`)

Detection uses **SpotNet** ([dotnet.py](torch_spots/detection/dotnet.py)), a fully-convolutional network that predicts a spot probability map and a sub-pixel offset map. It has three parts:

1. **Backbone** ([modules.py](torch_spots/detection/modules.py)): `n_skips + 1` feature sub-networks in series. Each one after the first sees the original image together with the previous sub-network's output.
2. **Classification head**: outputs `(B, 2, H, W)`. Channel 0 is `p(background)` and channel 1 is `p(spot center)`.
3. **Offset regression head**: outputs `(B, 2, H, W)`, the `(Δy, Δx)` offset from each pixel to the nearest spot center.

### Inference

[inference.py](torch_spots/detection/inference.py) provides `SpotDetection`:

- **`predict_points(X)`** returns sub-pixel `[y, x]` coordinates for all detected spots.
- **`predict_transforms(X)`** returns the raw `detections` and `offsets` maps, for custom processing.

Images are min-max scaled to `[0, 1]`, tiled into 128×128 patches, run through SpotNet, and stitched back together. Spots are the local maxima of the probability map above `threshold`, shifted by the predicted offsets. Pass `clip=True` to also clip each image at the 0.01st and 99.9th percentiles before scaling. Postprocessing was inspired by [PoSTcode](https://github.com/gerstung-lab/postcode).

`SpotDetection()` downloads the published pretrained model from Hugging Face (see [below](#hugging-face-access)). `SpotDetection(model_path=...)` loads your own checkpoint. If that checkpoint was trained with in-model normalisation, also pass `norm_method=` (for example `'std'`).

```python
from torch_spots.detection.inference import SpotDetection

detector = SpotDetection(device="cuda:0")   # or model_path="path/to/saved_model_best_dict.pth"
points = detector.predict_points(images)    # images: (B, 1, H, W), raw values
```

### Training and evaluation

Training and evaluation use the **SpotNet v1.1** dataset (`train.npz`, `val.npz`, `test.npz`, in `~/.deepcell/spots/` by default).

1. **Convert the data** (the `.npz` files are not modified):

   ```bash
   python -m torch_spots.preprocess            # or --data-dir PATH
   ```

2. **Train** (optional):

   ```bash
   python -m torch_spots.detection.train --seed 0 --device cuda:0
   ```

   The defaults reproduce the published model. To change settings, pass a JSON file with `--config`. It is merged over the defaults:

   ```json
   {
     "augmentation": {"rotation_range": 0},
     "loss": {"regression_weight": 1.0, "classification_weight": 1.0}
   }
   ```

   You can set `augmentation` (`flip`, `rotation_range`, `zoom_range`), `loss` (`regression_weight`, `classification_weight`, `focal`), `optimizer` (`lr`, `momentum`, `step_decay`, `epoch_decay`, `kernel_l2`, `grad_clip`), `model` (`norm_method`: `'std'`, `'whole_image'` or `null`), `epochs` and `batch_size`. An unknown setting name is an error. Every run writes `training_config.json` next to its checkpoint, recording all settings used and listing any that differ from the defaults.

3. **Evaluate.** Without `--model-path`, this scores the published model on the test split. Results go to `per_image.csv` and `summary.json`:

   ```bash
   python -m torch_spots.detection.eval --device cuda:0
   python -m torch_spots.detection.eval --model-path data/model/<run>/saved_model_best_dict.pth \
       --device cuda:0 --output-dir data/eval/my_run
   ```

   Use `--split val` to score the validation split instead, and `--norm-method std` for checkpoints trained with in-model normalisation.

Metrics ([reference_metrics.py](torch_spots/detection/reference_metrics.py)) are precision, recall, F1, Jaccard, RMSE and d_md, using mutual-nearest-neighbour matching at 1 px and averaged over images. The default evaluation threshold is **0.993** (`EVAL_THRESHOLD` in [inference.py](torch_spots/detection/inference.py)).

---

## Decoding (`torch_spots/decoding/`)

Decoding assigns a gene to each detected spot. Each spot has one detection probability per image (one image per round and channel), and the pipeline matches this vector against a binary codebook.

### Inputs

- **`spot_probabilities`**: a `[num_spots, rounds × channels]` array of detection probabilities in `[0, 1]` (the `'probabilities'` entry from `get_spot_probabilities`). Column *i* must correspond to codebook column *i*.
- **`df_barcodes`**: a pandas DataFrame codebook. The first column is the gene name, and the remaining `rounds × channels` columns are each gene's `{0, 1}` barcode.
- **`locations`** (optional): the `[y, x]` array from the same hand-off.

### Model

[decoding_utils.py](torch_spots/decoding/decoding_utils.py) fits a mixture of Relaxed Bernoulli distributions with [Pyro](https://pyro.ai/), with one component per gene barcode plus a `Background` class. It uses SVI with MAP estimation, then computes the posterior `P(gene | spot)` for every spot.

`params_mode` controls how the sigma and temperature parameters are shared:

| Mode      | Parameters            | Description                                                     |
|-----------|-----------------------|-----------------------------------------------------------------|
| `'2'`     | 2 total               | One sigma and one temperature, for each of {0,1} barcode states |
| `'2*R'`   | 2 × rounds            | Parameters vary per imaging round                               |
| `'2*C'`   | 2 × channels          | Parameters vary per imaging channel                             |
| `'2*R*C'` | 2 × rounds × channels | Parameters vary per (round, channel) pair (the default)         |

Known limitation: in `'2*R'` and `'2*C'` modes, the columns that share parameters are not grouped by round or channel as intended. Only the default `'2*R*C'` mode is unaffected.

### Postprocessing

[decode.py](torch_spots/decoding/decode.py) wraps the model in `SpotDecoding`, which applies three steps after inference:

1. **Thresholding**: spots whose top probability is below `pred_prob_thresh` (default 0.95) are labelled `Unknown`.
2. **Error rescue** (`rescue_errors`, on by default): a `Background` or `Unknown` spot that is exactly one bit away from a single barcode is assigned to that gene.
3. **Mixed rescue** (`rescue_mixed`, off by default): checks whether a low-confidence spot is two overlapping barcodes, and if so adds a second row for that spot.

### Output

`SpotDecoding.predict()` returns a pandas DataFrame with one row per spot, in input order. Any mixed-rescue rows come at the end.

| Column           | Description                                                                                                         |
|------------------|---------------------------------------------------------------------------------------------------------------------|
| `spot_index`     | Index of the input spot (repeats for a mixed-rescue row)                                                            |
| `y`, `x`         | Row and column of the spot (only when you pass `locations`)                                                         |
| `predicted_id`   | 1-indexed gene ID                                                                                                   |
| `predicted_name` | Gene name (`'Background'`, `'Unknown'`, or a gene name)                                                             |
| `probability`    | Maximum posterior probability (−1 for a mixed-rescue row)                                                           |
| `source`         | `'prediction'`, `'error rescue'`, or `'mixed rescue'`                                                               |
| `prob_<name>`    | The spot's detection probability for each codebook bit, named after the codebook column (for example `prob_RS0015`) |

Set `genes_only=True` to drop `Background`, `Unknown` and `Blank-*` rows. `seed` (default 1) makes results reproducible on the same device. After a run, `decoder.last_run` holds the settings and fitted parameters.

---

## End-to-end usage

```python
from torch_spots.detection.inference import SpotDetection
from torch_spots.decoding.decode import SpotDecoding

# 1. Detect spots. X holds the N images of the stack, in the codebook's column order.
detector = SpotDetection()
transforms = detector.predict_transforms(X)
handoff = detector.get_spot_probabilities(transforms, threshold=0.9, min_distance=1, pool_size=0)
# handoff['locations']      [y, x] of each candidate, shape [n, 2]
# handoff['probabilities']  detection probability per image at each candidate, shape [n, N]

# 2. Decode gene identities
decoder = SpotDecoding(df_barcodes=codebook, rounds=10, channels=2)
table = decoder.predict(handoff['probabilities'], handoff['locations'])
```

Candidates are local maxima of the highest probability across all images, so a spot that is bright in only one image is still kept. The candidate threshold (0.9) is separate from the 0.993 detection evaluation threshold.

---

## Hugging Face access

The pretrained model weights are private to the Van Valen Lab Hugging Face organization.

1. Create a free Hugging Face account and join the [vanvalenlab organization](https://huggingface.co/vanvalenlab). You are added as a "Read" member.
2. Generate an [access token](https://huggingface.co/settings/tokens) and export it as `HF_TOKEN`. The weights download automatically the first time you create a `SpotDetection`.

## Troubleshooting

If your GPU's CUDA drivers do not support the default PyTorch version, you get an error at the start of inference. Install a PyTorch version that matches your GPU.
