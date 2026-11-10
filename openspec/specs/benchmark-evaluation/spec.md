# benchmark-evaluation Specification

## Purpose

Defines how a trained spot detector is evaluated for publication: the pinned dataset, operating point and seed protocol, the published SpotNet benchmark it must reach, and the evidence required before torch_spots adopts any divergence from its reference configuration.

Source citations refer to the vendored `deepcell_spots/` package, version 0.4.2 (`deepcell_spots/_version.py:5`), as `deepcell_spots/<file>:<line>`. Scores are computed with the reference metric set defined by the `detection-metrics` capability.

## Requirements

### Requirement: Pinned benchmark dataset
Benchmark evaluation SHALL use the SpotNet v1.1 test split (`SpotNet-v1_1/test.npz`, CRC-32 `36c922d7`, 94 images of 128×128, 995 annotated spots; source: `~/.deepcell/spots/README.md`). The evaluation SHALL fail rather than score a test set that does not match these properties.

#### Scenario: Correct dataset
- **WHEN** the benchmark is run against the v1.1 test split
- **THEN** all 94 images are scored and the dataset version and checksum are recorded with the result

#### Scenario: Wrong dataset version
- **WHEN** the benchmark is pointed at a test split with a different image count, spot count or checksum (for example SpotNet v1.0, 100 images)
- **THEN** evaluation stops with an error naming the mismatch and no score is reported

### Requirement: Pinned training data
Any configuration scored for the benchmark or for a divergence decision SHALL be trained on the SpotNet v1.1 train split (`train.npz`, CRC-32 `4cb57ba8`, 849 images, 17,410 spots) and validated on the v1.1 val split (`val.npz`, CRC-32 `29836378`, 95 images, 1,957 spots). Training SHALL refuse data that does not match, and SHALL record the versions and checksums used.

#### Scenario: Checkpoint of unknown provenance
- **WHEN** a checkpoint has no recorded training-data version and checksum (for example, any checkpoint trained before this change)
- **THEN** it is not used for a benchmark or divergence result

#### Scenario: Mismatched training store
- **WHEN** training is pointed at train or val data whose image count, spot count or checksum differs from the pinned values
- **THEN** training stops with an error naming the mismatch before any weights are updated

### Requirement: All test images scored
Benchmark evaluation SHALL score every image in the pinned test split, with no images excluded.

#### Scenario: No silent filtering
- **WHEN** a benchmark result is produced
- **THEN** the number of scored images is 94

### Requirement: Reference operating point
Benchmark evaluation SHALL convert model outputs to points by taking local maxima of the spot-class probability above the reference threshold, with a minimum peak separation of 1 px, then adding the regressed offset at each peak. The reference threshold is 0.993, the single value chosen under "Operating-point selection on the validation split". It SHALL be the default for evaluation and SHALL be recorded with every result.

#### Scenario: Peak extraction source
- **GIVEN** the original application's peak extraction (`deepcell_spots/applications/spot_detection.py:266-269`; `deepcell_spots/utils/postprocessing_utils.py:85-119`)
- **WHEN** model outputs are converted to points
- **THEN** local maxima above the reference threshold are taken with minimum separation 1 px, and the regressed offset is added at each peak

#### Scenario: Recorded selection
- **GIVEN** the selection record `data/eval/round6_threshold_selection.json` (three 50-epoch seeds, mean validation F1 0.6862 at 0.993, 0.6858 at 0.99)
- **WHEN** the reference threshold is read
- **THEN** it is 0.993

#### Scenario: Default benchmark run
- **WHEN** the benchmark is run without overrides
- **THEN** the reference threshold, minimum distance 1 px and τ = 1 px are used and recorded with the result

#### Scenario: Sweeps include the reference point
- **WHEN** a sweep over probability threshold, minimum distance or τ is run
- **THEN** the grid includes (reference threshold, 1 px, 1 px), and the result at that point is reported separately

#### Scenario: Former threshold is not the default
- **WHEN** the benchmark is run without overrides
- **THEN** the threshold used is not 0.95, unless 0.95 is the value chosen on the validation split

### Requirement: Evaluation preprocessing matches training
Images SHALL be preprocessed for evaluation with exactly the same transformation used to preprocess training and validation images for the model being evaluated.

#### Scenario: Same image, two paths
- **WHEN** one raw image is preprocessed by the training data path and by the evaluation path
- **THEN** the two resulting arrays are identical

### Requirement: Seed protocol
Every configuration scored for the benchmark or for a divergence decision SHALL be trained independently with 3 random seeds. Each seed SHALL be evaluated separately. The result SHALL report each seed's test-set scores, their mean and their range (max − min).

#### Scenario: Reporting a configuration
- **WHEN** a configuration's benchmark result is reported
- **THEN** it lists 3 per-seed F1 values, their mean and their range, plus the seed values used

### Requirement: Result provenance
Every benchmark result SHALL record the code revision, full training and evaluation configuration, seeds, dataset version and checksum, and operating point. A result missing any of these SHALL NOT be used for a benchmark or divergence decision.

#### Scenario: Reproducing a result
- **WHEN** someone reads a recorded benchmark result
- **THEN** they can identify the commit, configuration, seeds and dataset needed to rerun it

### Requirement: Accuracy divergence evidence
A divergence from the reference configuration that claims better accuracy SHALL be adopted only if its mean F1 exceeds the reference mean F1 by more than the reference's seed range. Both configurations SHALL use the same seed values and protocol.

#### Scenario: Clear improvement
- **GIVEN** reference per-seed F1 of 0.82, 0.83 and 0.84 (mean 0.83, range 0.02)
- **WHEN** a divergence scores a mean F1 of 0.86
- **THEN** the accuracy claim is supported

#### Scenario: Within noise
- **GIVEN** the same reference
- **WHEN** a divergence scores a mean F1 of 0.845
- **THEN** the accuracy claim is not supported

### Requirement: Quality-of-life divergence evidence
A divergence that claims a non-accuracy benefit (such as speed, memory, stability or usability) SHALL be adopted only if its mean F1 is at least the reference mean F1 minus the reference's seed range, and the claimed benefit is measured and reported.

#### Scenario: Faster and non-inferior
- **GIVEN** reference per-seed F1 of 0.82, 0.83 and 0.84
- **WHEN** a divergence scores a mean F1 of 0.815 and a measured 30% reduction in training time
- **THEN** the divergence may be adopted

#### Scenario: Benefit not measured
- **WHEN** a divergence is non-inferior but its claimed benefit has no measurement
- **THEN** the divergence is not adopted

### Requirement: Checkpoint model compatibility
Evaluation SHALL score a checkpoint only if its training record names the current model-architecture version (defined by the `detection-model` capability). Otherwise it SHALL stop with an error naming the recorded and current versions, and report no score.

#### Scenario: Matching version
- **WHEN** a checkpoint whose training record names the current architecture version is evaluated
- **THEN** it is scored, and the architecture version is recorded with the result

#### Scenario: Checkpoint from before this change
- **GIVEN** a checkpoint whose training record has no architecture version (for example, any round 1 or round 2 checkpoint of training-parity)
- **WHEN** it is evaluated
- **THEN** evaluation stops with an error saying the version is missing, and no score is reported

#### Scenario: Different version
- **GIVEN** a checkpoint whose training record names an architecture version other than the current one
- **WHEN** it is evaluated
- **THEN** evaluation stops with an error naming both versions, and no score is reported

### Requirement: Published model reproduction target
The reference configuration SHALL reproduce the published model: its mean reference-set F1 over its 3 seeds SHALL be at least the published model's score (see "Published model score") on the pinned benchmark dataset at the reference operating point. Scoring above the published model SHALL NOT count against reproduction.

#### Scenario: Reference matches or beats the published model
- **GIVEN** a recorded published-model score of 0.842
- **WHEN** the reference configuration scores per-seed F1 of 0.86, 0.88 and 0.89 (mean 0.877)
- **THEN** the result is reported as reproducing the published model

#### Scenario: Reference falls short
- **GIVEN** a recorded published-model score of 0.842
- **WHEN** the reference configuration's mean F1 is 0.830
- **THEN** the result is reported as not reproducing the published model, and no divergence is adopted against that reference

#### Scenario: Judged at the same operating point
- **WHEN** the reference configuration is judged against the published-model score
- **THEN** both were scored at the same reference threshold, minimum distance and τ, on the same pinned test split, with the reference metric set

### Requirement: Published model score
The published-model score SHALL be measured once by running the published SpotDetection-8 model through its own deepcell_spots 0.4.2 application, with that application's default preprocessing, on the pinned test split at the reference operating point, and scoring its points with the reference metric set. The value and its provenance SHALL be recorded in this spec.

#### Scenario: Recorded value
- **GIVEN** SpotDetection-8 (archive sha256 `7aba51b9ea198e225abff220057a090f6dbb6dcc87448c8bffbc91ee1db3ae36`), the deepcell_spots 0.4.2 application with default preprocessing (percentile clip on, reflect padding), deepcell 0.12.10 and TensorFlow 2.8.4
- **WHEN** it is scored on the 94 test images (SpotNet v1.1, CRC-32 `36c922d7`) at threshold 0.993, minimum distance 1 px and τ = 1 px
- **THEN** the recorded mean reference-set F1 is 0.8100 (precision 0.7328, recall 0.9604), from `data/eval/published_spotdetection8/summary.json`

#### Scenario: Provenance recorded
- **WHEN** the published-model score is recorded
- **THEN** the record names the model (SpotDetection-8) and its archive hash, the application version, the preprocessing used (including the percentile clip), the operating point, the dataset version and checksum, and the 94 images scored

#### Scenario: Agreement with the earlier TensorFlow evaluation
- **WHEN** the published model is scored by this procedure at threshold 0.995
- **THEN** its macro F1 is 0.842 ± 0.001, matching the earlier evaluation in `torch_benchmark/results/tensorflow/spots`

### Requirement: Operating-point selection on the validation split
The reference threshold SHALL be the threshold, from the grid 0.90, 0.95, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998 and 0.999, with the highest mean reference-set F1 on the pinned validation split over the 3 reference-configuration seeds (minimum distance 1 px, τ = 1 px). Ties SHALL go to the lower threshold. Test-split results SHALL NOT be used in the choice.

#### Scenario: Selection
- **GIVEN** mean validation F1 of 0.85 at 0.99, 0.87 at 0.995 and 0.86 at 0.998, and lower values elsewhere on the grid
- **WHEN** the reference threshold is selected
- **THEN** it is 0.995

#### Scenario: Tie
- **GIVEN** the highest mean validation F1 is reached at both 0.995 and 0.997
- **WHEN** the reference threshold is selected
- **THEN** it is 0.995

#### Scenario: Recorded choice
- **WHEN** the reference threshold has been selected
- **THEN** the per-threshold validation F1 for each seed, the checkpoints used and the chosen value are recorded

### Requirement: Validation-split scoring
Evaluation and sweeps SHALL be able to score the pinned SpotNet v1.1 validation split (`val.npz`, CRC-32 `29836378`, 95 images, 1,957 spots), refusing data that does not match it. A validation-split result SHALL be labelled with its split and SHALL NOT be reported as a benchmark result.

#### Scenario: Scoring the validation split
- **WHEN** evaluation is run on the validation split
- **THEN** all 95 images are scored, and the result records the split as validation with its version and checksum

#### Scenario: Validation result is not a benchmark
- **WHEN** a validation-split result is passed to the benchmark report
- **THEN** the report refuses it, naming the split
