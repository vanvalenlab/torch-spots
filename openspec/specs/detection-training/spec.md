# detection-training Specification

## Purpose
Defines how the spot detector is trained: the reference training configuration taken from the published SpotDetection-8 model's record (augmentation, losses, optimizer, schedule and checkpoint selection), how settings that differ from it are expressed and recorded, and the robustness guarantees of the training data path.

Source citations refer to the vendored `deepcell_spots/` package (version 0.4.2) as `deepcell_spots/<file>:<line>`, and to deepcell-tf 0.12.10 / Keras 2.8 as installed alongside it.

## Requirements

### Requirement: Reference flip augmentation
By default, each training image SHALL be flipped horizontally with probability 0.5 and, independently, vertically with probability 0.5, after the reference rotation and zoom. A flipped spot coordinate SHALL map to `L − 1 − c` on the flipped axis, where L is the image size on that axis (source: `deepcell_spots/training.py:38-42,144-145`; `deepcell_spots/utils/augmentation_utils.py:293-309`; Laubscher et al. 2024, Methods, "Model architecture").

#### Scenario: Horizontal flip moves pixels and points together
- **GIVEN** a 128 × 128 training image with a spot at (y = 10.3, x = 20.7), with rotation and zoom disabled
- **WHEN** the image is flipped horizontally
- **THEN** the spot label is at (10.3, 106.3) and the pixel previously in column c is in column 127 − c

#### Scenario: Flip frequency
- **WHEN** many training samples are drawn with the default augmentation
- **THEN** about half are flipped horizontally and about half vertically, independently of each other

#### Scenario: Validation is not augmented
- **WHEN** a validation sample is drawn
- **THEN** the image and its spot labels are returned unchanged

### Requirement: Aligned optional geometric augmentation
By default, each training image SHALL be rotated by an angle drawn uniformly from [−180°, 180°] and zoomed by a factor drawn uniformly from [0.8, 1.2], with no shear (source: Laubscher et al. 2024, Methods). Both SHALL be configurable and can be turned off. They SHALL transform image pixels and spot labels about the same centre, so that a spot's label stays on its image feature, and SHALL fill pixels outside the source image with the nearest edge value.

#### Scenario: Default ranges
- **WHEN** many training samples are drawn with the default augmentation
- **THEN** their rotation angles are spread over [−180°, 180°] and their zoom factors over [0.8, 1.2], with no angle or factor outside those ranges

#### Scenario: Rotation keeps labels on spots
- **GIVEN** a 128 × 128 image containing a single Gaussian spot centred at (30, 100)
- **WHEN** it is rotated by 180° with no zoom
- **THEN** the transformed label and the image's intensity centre of mass agree within 0.05 px on each axis, both at (97, 27)

#### Scenario: Fill does not create artificial edges
- **WHEN** a rotated image exposes corners outside the source image
- **THEN** those pixels take the value of the nearest source edge pixel, and no pixel falls outside the source image's value range

#### Scenario: Geometric augmentation turned off
- **WHEN** training runs with rotation range 0 and zoom disabled
- **THEN** no training sample is rotated or zoomed, and the training record lists both settings as non-reference

### Requirement: Reference classification loss
The classification loss SHALL be deepcell's weighted categorical cross-entropy. For each batch, class weights are `T / (n · (c_k + 1))`, where T is the number of pixels, n the number of classes and c_k the number of pixels of class k. The weighted log-likelihood is summed over the class axis and averaged over pixels (source: deepcell-tf 0.12.10 `deepcell/losses.py`, `weighted_categorical_crossentropy`; `deepcell_spots/dotnet_losses.py:120-143`).

#### Scenario: Worked example
- **GIVEN** one 2 × 2 image whose labels have one spot pixel and three background pixels, a predicted spot probability of 0.5 at the spot pixel, and a predicted background probability of 0.9 at each background pixel
- **WHEN** the classification loss is computed
- **THEN** the weights are 1.0 (spot) and 0.5 (background) and the loss is 0.2128 ± 1e-4

#### Scenario: Agreement with the deepcell formula
- **WHEN** the classification loss is computed on randomized batches of probability maps and one-hot labels with varying spot density and image width
- **THEN** it equals a direct NumPy transcription of the deepcell formula within 1e-6 relative error

#### Scenario: Focal loss uses the same weights
- **WHEN** focal loss is enabled
- **THEN** it uses the same class weights as the cross-entropy and equals the deepcell focal-loss formula (deepcell-tf 0.12.10 `deepcell/losses.py`, `weighted_focal_loss`) within 1e-6 relative error

### Requirement: Reference loss combination
By default, the training objective SHALL be 1/6 times the offset-regression loss plus 5/6 times the classification loss, plus the kernel L2 penalty with no extra weight (source: published SpotDetection-8 model, saved training configuration `loss_weights`: offset_regression 0.1667, classification 0.8333).

#### Scenario: Published weighting
- **GIVEN** a batch where the regression loss is 0.30, the classification loss is 0.20 and the L2 penalty is 0.01
- **WHEN** the training objective is computed with default settings
- **THEN** it is 0.22667 ± 1e-5

#### Scenario: Equal weighting
- **GIVEN** the same batch and loss weights set to regression 1.0 and classification 1.0 (weights are not renormalised to sum to 1)
- **WHEN** the training objective is computed
- **THEN** it is 0.51

### Requirement: Kernel L2 penalty
The training objective SHALL include `1e-5 · Σ w²` over the kernel weights of every backbone convolution and of the two classification-head projection layers. Biases, normalization parameters and regression-head layers SHALL NOT be penalized, and no other weight decay SHALL be applied (source: `deepcell/model_zoo/featurenet.py:152,201`; `deepcell_spots/dotnet.py:62,66`; see design D4).

#### Scenario: Only kernels are penalized
- **WHEN** any regression-head parameter, bias or normalization parameter is changed and no backbone or classification-head kernel is
- **THEN** the L2 penalty is unchanged

#### Scenario: Backbone and head kernels are penalized
- **WHEN** a backbone convolution kernel, or a classification-head kernel, is changed
- **THEN** the L2 penalty changes by `1e-5` times the change in that kernel's `Σ w²`

#### Scenario: Penalty value
- **GIVEN** penalized kernels (backbone and classification head together) whose squared weights sum to 250
- **WHEN** the penalty is computed
- **THEN** it is 0.0025

### Requirement: Reference optimizer and learning-rate schedule
Training SHALL use SGD with Nesterov momentum 0.9 and no gradient clipping. The learning rate at global step t during epoch e (both counted from 0) SHALL be `0.01 · 0.99^e / (1 + 1e-6 · t)` (source: `deepcell_spots/training.py:29-37`; Keras 2.8 `optimizer_v2/optimizer_v2.py:1003-1013`; epoch decay 0.99 from `deepcell_spots/applications/spot_detection.py` `model_metadata` and the published model's saved optimizer state, learning rate 0.01 · 0.99⁴).

#### Scenario: Learning rate mid-training
- **WHEN** training is at epoch 2 and global step 2000
- **THEN** the learning rate applied is 0.0097814 ± 1e-7

#### Scenario: First step
- **WHEN** the first optimizer step of training is taken
- **THEN** the learning rate applied is 0.01

#### Scenario: Last step of a default run
- **WHEN** training with default settings on the SpotNet v1.1 train split takes its last step (epoch 9, global step 8,489)
- **THEN** the learning rate applied is 0.0090583 ± 1e-7

#### Scenario: No clipping
- **WHEN** a batch produces a gradient with norm greater than 1
- **THEN** the gradient is applied unscaled

### Requirement: Reference training length and batch size
By default, training SHALL run 10 epochs with batch size 1. Each epoch SHALL present every training image once, in a seeded random order, and SHALL drop a final partial batch The source is the published model's saved optimizer state (see the "Recorded source" scenario).

#### Scenario: Recorded source
- **GIVEN** the published model's saved learning rate 0.01 · 0.99⁴, which shows the shipped weights were saved in epoch 4, and `model_metadata` (`batch_size` 1, `training_steps_per_epoch` 849, a planned `n_epochs` of 50 that the shipped weights never reached)
- **WHEN** the reference length is read
- **THEN** it is 10 epochs, and the measured 3-seed mean test F1 is 0.8451 (round 8) at 10 epochs against 0.7607 (round 6) at 50 epochs

#### Scenario: Default step count
- **WHEN** training runs with default settings on the SpotNet v1.1 train split (849 images)
- **THEN** 849 optimizer steps are taken per epoch and 8,490 in total

#### Scenario: Partial batch dropped
- **GIVEN** 849 training images and a batch size of 10
- **WHEN** one epoch runs
- **THEN** 84 optimizer steps are taken

### Requirement: Best-checkpoint selection
After each epoch, training SHALL compute the training objective on the validation split without augmentation. The best-model checkpoint SHALL hold the weights from the epoch with the lowest such value, and SHALL be rewritten only when that value improves. A separate checkpoint SHALL hold the final epoch's weights (source: `deepcell_spots/training.py:194-196`).

#### Scenario: Best is not last
- **GIVEN** validation objectives of 0.50, 0.40 and 0.45 over three epochs
- **WHEN** training finishes
- **THEN** the best-model checkpoint holds the epoch-2 weights (value 0.40) and the final checkpoint holds the epoch-3 weights

### Requirement: Recorded non-reference settings
Every setting covered by this capability SHALL be configurable, with the reference values as defaults. The training record SHALL contain the full set of values used, and SHALL list by name each setting whose value differs from the reference.

#### Scenario: Reference run
- **WHEN** training runs with default settings
- **THEN** the training record lists no non-reference settings

#### Scenario: Ablation run
- **WHEN** training runs with rotation augmentation turned off and the loss weights set to 1 : 1
- **THEN** the training record lists the rotation and loss-weight settings as non-reference, with their values

#### Scenario: Longer training
- **WHEN** training runs with 50 epochs and every other setting at its default
- **THEN** the training record lists only the number of epochs as non-reference, with the value 50

### Requirement: Finite preprocessing of constant images
Training preprocessing SHALL return finite values for every input image, including an image whose pixels all have the same value.

#### Scenario: Blank image
- **WHEN** an image with every pixel equal to 500 is preprocessed
- **THEN** every output value is finite and all are equal
