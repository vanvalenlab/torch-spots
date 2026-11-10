# detection-model Specification

## Purpose

Defines the spot-detection network and the input it expects: the reference architecture and settings taken from the published SpotDetection-8 model's record (backbone geometry, in-model normalisation, which is off by default as in the published model, initialisation, normalisation-layer constants), the image preprocessing shared by training, evaluation and inference, and how model settings and the architecture version are recorded.

Source citations refer to the vendored `deepcell_spots/` package (version 0.4.2) as `deepcell_spots/<file>:<line>`, and to deepcell-tf 0.12.10 as `deepcell/<file>:<line>`.

## Requirements

### Requirement: Model outputs
For an input of shape (B, C, H, W), the model SHALL return a spot-probability map of shape (B, 2, H, W) that is a softmax over the class axis, where channel 1 is the spot class, and an offset map of shape (B, 2, H, W) holding (dy, dx) in pixels (source: `deepcell_spots/dotnet.py:27-103`).

#### Scenario: Shapes and softmax
- **WHEN** a batch of 2 single-channel 128 × 128 images is passed through the model
- **THEN** both outputs have shape (2, 2, 128, 128), and the two probability channels sum to 1 within 1e-6 at every pixel

#### Scenario: Non-square input
- **WHEN** a single-channel 96 × 160 image is passed through the model
- **THEN** both outputs are 96 × 160

### Requirement: Backbone sub-network layers
With receptive field 13, each of the 4 backbone sub-networks SHALL compute, in order: 4 × 4 conv (32 filters) → BatchNorm → ReLU → 3 × 3 conv (32) → BatchNorm → ReLU → 2 × 2 max-pool with stride 1 → 4 × 4 conv with dilation 2 (128) → BatchNorm → ReLU (source: `deepcell/model_zoo/featurenet.py:146-169,199-203`; `deepcell/layers/pooling.py:63-64`).

#### Scenario: Max-pool present
- **WHEN** the layers of any backbone sub-network are listed
- **THEN** a 2 × 2, stride-1 max-pool sits between the second ReLU and the dilated 4 × 4 conv

#### Scenario: Skip wiring unchanged
- **WHEN** the backbone runs
- **THEN** sub-network 0 receives the normalised image (C channels), and each later sub-network receives the normalised image concatenated with the previous sub-network's 128-channel output (C + 128 channels, image first)

### Requirement: Symmetric receptive field
Each sub-network's output at a pixel SHALL depend on exactly the 13 × 13 input window centred on that pixel (offsets −6 to +6 on each axis). The full backbone output SHALL depend on exactly the window from −24 to +24 on each axis (source: `deepcell/model_zoo/featurenet.py:100,132-136`).

#### Scenario: Single sub-network window
- **GIVEN** a single sub-network in evaluation mode with randomly initialised weights, and a 128 × 128 input
- **WHEN** the input pixels that influence the output at (64, 64) are found
- **THEN** they are exactly rows 58–70 and columns 58–70

#### Scenario: Full backbone window
- **GIVEN** the full model in evaluation mode with randomly initialised weights, and a 128 × 128 input
- **WHEN** the input pixels that influence the spot-probability output at (64, 64) are found, with in-model normalisation disabled for this check
- **THEN** they are exactly rows 40–88 and columns 40–88

### Requirement: Border padding once per sub-network
Each sub-network SHALL extend its input at the image border by mirror reflection by 6 px on every side (the edge pixel is not repeated), and SHALL apply no padding to any intermediate result (source: `deepcell/model_zoo/featurenet.py:132-136`; `deepcell/layers/padding.py:61`).

#### Scenario: Equivalent to one up-front reflection
- **GIVEN** a sub-network and an input image x
- **WHEN** its output on x is compared with the same layers applied with no padding at all to x mirror-reflected by 6 px on every side
- **THEN** the two outputs are equal within 1e-6

### Requirement: Weight initialisation
Backbone and classification-head kernels SHALL start from a normal distribution truncated at 2 standard deviations, scaled so that the resulting standard deviation is √(2 / fan_in) (Keras `he_normal`). Regression-head kernels SHALL start from N(0, 0.01). Every bias SHALL start at 0 (source: `deepcell/model_zoo/featurenet.py:58,150,199`; `deepcell_spots/dotnet.py:39,61,65,89-90`).

#### Scenario: Backbone kernel statistics
- **WHEN** a model is initialised and the kernel of a sub-network's first 4 × 4 conv with 129 input channels (fan_in 2,064) is inspected
- **THEN** its standard deviation is within 3% of 0.03113, its mean is within 0.002 of 0, and no value exceeds 0.07078 in magnitude

#### Scenario: Regression-head kernel statistics
- **WHEN** a model is initialised and the regression head's first 3 × 3 conv kernel (128 → 256) is inspected
- **THEN** its standard deviation is within 3% of 0.01

#### Scenario: Zero biases
- **WHEN** a model is initialised
- **THEN** every bias parameter is exactly 0, every normalisation-layer scale is 1 and every normalisation-layer shift is 0

#### Scenario: Seeded initialisation
- **WHEN** two models are initialised under the same seed
- **THEN** all their parameters are identical

### Requirement: Normalisation-layer constants
Every BatchNorm layer SHALL update its running statistics as `running = 0.99 · running + 0.01 · batch` and SHALL use ε = 1e-3 (Keras defaults; source: `deepcell/model_zoo/featurenet.py:153,202`; `deepcell_spots/dotnet.py:63`).

#### Scenario: Running-mean update
- **GIVEN** a BatchNorm layer with running mean 0 for a channel
- **WHEN** one training-mode batch whose mean for that channel is 2.0 passes through it
- **THEN** the running mean for that channel is 0.02

#### Scenario: Epsilon in evaluation mode
- **GIVEN** a BatchNorm channel in evaluation mode with running mean 0, running variance 0, scale 1 and shift 0
- **WHEN** a value of 0.1 passes through it
- **THEN** the output is 3.1623 ± 1e-4

### Requirement: Reference input preprocessing
Training, validation, evaluation and every inference entry point SHALL preprocess each image by default with the same transform: min-max scaling of each image and channel to [0, 1], with no percentile clipping and no offset. A constant image SHALL map to 0 everywhere (source: `deepcell_spots/training.py:89-90,135-155`; `deepcell_spots/utils/preprocessing_utils.py:56-67`).

#### Scenario: SpotNet v1.1 images unchanged
- **WHEN** any image from the pinned SpotNet v1.1 train, val or test split is preprocessed
- **THEN** the result is identical to the stored image

#### Scenario: Scaling
- **GIVEN** a single-channel image whose values range from 100 to 1,100
- **WHEN** it is preprocessed
- **THEN** each value v becomes (v − 100) / 1,000, and the brightest pixels are not clipped

#### Scenario: All paths agree
- **WHEN** one raw image is preprocessed by the training data path, the evaluation path and each inference entry point with default settings
- **THEN** the resulting arrays are identical

### Requirement: Optional percentile clip for inference
Inference entry points SHALL offer, off by default, the original application's clip: before min-max scaling, each image and channel is clipped to its 0.01st and 99.9th percentiles over all pixels. Training and benchmark evaluation SHALL NOT apply it (source: `deepcell_spots/utils/preprocessing_utils.py:60-61`; `deepcell_spots/applications/spot_detection.py:224`).

#### Scenario: Clip requested
- **GIVEN** a 100 × 100 image of value 10 everywhere except one pixel of value 10,000
- **WHEN** inference preprocessing runs with the clip enabled
- **THEN** the result is a constant image of 0

#### Scenario: Clip not requested
- **WHEN** inference preprocessing runs with default settings on the same image
- **THEN** the result equals the reference preprocessing

### Requirement: Recorded model settings
The in-model normalisation method SHALL be configurable in the training configuration, with none as the reference value and `'std'` and `'whole_image'` as alternatives. A non-reference value SHALL be listed by name in the training record's non-reference settings.

#### Scenario: Reference run
- **WHEN** training runs with default settings
- **THEN** the training record lists no non-reference settings and records the normalisation method as none

#### Scenario: Normalisation ablation
- **WHEN** training runs with the normalisation method set to `'std'`
- **THEN** the training record lists the normalisation method as non-reference, with its value

### Requirement: Recorded architecture version
The training record SHALL contain a model-architecture version. The version SHALL change whenever a code change alters the function the network computes for given weights and settings.

#### Scenario: Version recorded
- **WHEN** training finishes
- **THEN** its training record contains the current model-architecture version

### Requirement: Inference loads only compatible checkpoints
Inference SHALL load a checkpoint only if its training record names the current model-architecture version, and SHALL build the model with the settings recorded there. Otherwise it SHALL stop with an error naming the problem. This applies to the default pretrained model as well.

#### Scenario: Compatible checkpoint
- **GIVEN** a checkpoint trained with the normalisation method set to none, whose training record names the current architecture version
- **WHEN** it is loaded for inference
- **THEN** the model is built with no in-model normalisation and the weights load without missing or unexpected parameters

#### Scenario: Outdated default model
- **GIVEN** the published default pretrained model has no training record naming the current architecture version (as is true of `torch-spots_2026-07-29.pth`)
- **WHEN** inference is started without a checkpoint
- **THEN** it stops with an error saying that the default model predates the current architecture and that a checkpoint must be passed, and it does not load those weights

### Requirement: Optional in-model normalisation
By default the model SHALL apply no in-model normalisation: the preprocessed image is passed to the first sub-network unchanged (source: published SpotDetection-8 model, where every `ImageNormalization2D` layer has `norm_method` null).

#### Scenario: Default is off
- **WHEN** a model is built with default settings and given an input image
- **THEN** the first sub-network receives that image unchanged

### Requirement: Local 'std' normalisation option
When the `'std'` method is selected, the model SHALL normalise its raw input once, as deepcell's `'std'` method: `o = x − box(x)`, then `o / (sqrt(box(o²) − box(o)²) + 1e-7)`. `box` is a 13 × 13 window sum with zero padding outside the image, divided by 169 · C (source: `deepcell/layers/normalization.py:141-142,164-178,189-191`).

#### Scenario: Agreement with the deepcell formula
- **WHEN** the `'std'` normalisation is applied to randomized single-channel images (sizes 32, 64 and 128, including images with uneven background)
- **THEN** it equals a direct NumPy transcription of the deepcell formula within 1e-5 absolute error

#### Scenario: Selected method is applied
- **WHEN** a model is built with the normalisation method set to `'std'`
- **THEN** its input passes through the `'std'` normalisation before the first sub-network

#### Scenario: Constant images stay finite
- **WHEN** an all-zero image, and separately an image whose pixels all equal 0.7, are normalised with the `'std'` method
- **THEN** every output value is finite, and every output value of the all-zero image is 0
