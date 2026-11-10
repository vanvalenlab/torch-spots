# detection-metrics Specification

## Purpose

Defines how predicted spot locations are matched to ground-truth annotations and scored, per image and across a test set. The reference metric set reproduces the SpotNet (deepcell_spots 0.4.2) definitions so results can be compared with the published benchmark. Additional metrics are reported alongside it but never used to judge a result.

Source citations refer to the vendored `deepcell_spots/` package, version 0.4.2 (`deepcell_spots/_version.py:5`), as `deepcell_spots/<file>:<line>`.

## Requirements

### Requirement: Point coordinate convention
Ground-truth and predicted spot locations SHALL be represented as `[y, x]` pixel coordinates, and all distances SHALL be Euclidean in pixels (source: `deepcell_spots/utils/postprocessing_utils.py:117-119`; `deepcell_spots/point_metrics.py:350`).

#### Scenario: Distance between two points
- **WHEN** a true point `[0, 0]` and a predicted point `[3, 4]` are compared
- **THEN** their distance is 5.0 px

### Requirement: Reference matching rule
The reference metric set SHALL match a true point and a predicted point only when each is the other's nearest neighbour and their distance is less than or equal to the distance threshold τ. Each point SHALL be matched at most once (source: `deepcell_spots/point_metrics.py:113-158`, comparison at `:153`).

#### Scenario: Close pair kept even when another assignment has lower total cost
- **GIVEN** true points `[[0,0],[10,0]]`, predicted points `[[1,0],[-8.5,0]]` and τ = 1
- **WHEN** reference matching is computed
- **THEN** exactly one pair, true `[0,0]` with predicted `[1,0]`, is matched

#### Scenario: Non-mutual neighbours in a dense cluster are not matched
- **GIVEN** true points `[[0,0],[1,0]]`, predicted points `[[0.6,0],[1.9,0]]` and τ = 1
- **WHEN** reference matching is computed
- **THEN** exactly one pair is matched, and precision and recall are both 0.5

#### Scenario: Threshold is inclusive
- **GIVEN** one true point `[0,0]`, one predicted point `[1,0]` and τ = 1
- **WHEN** reference matching is computed
- **THEN** the pair is matched

### Requirement: Default distance threshold
The reference metric set SHALL use τ = 1 px unless a different τ is explicitly requested (source: `deepcell_spots/point_metrics.py:442-446`, `:369`).

#### Scenario: No threshold given
- **WHEN** reference metrics are computed without specifying τ
- **THEN** τ = 1 px is used and is recorded with the result

### Requirement: Reference detection scores
For each image the reference metric set SHALL report precision = TP / N_pred, recall = TP / N_true, F1 = 2PR / (P + R) (0 when P or R is 0) and Jaccard index JAC = F1 / (2 − F1). TP is the number of matched pairs (source: `deepcell_spots/point_metrics.py:329-342`).

#### Scenario: Partial match
- **GIVEN** 4 true points, 2 predicted points and 2 matched pairs
- **THEN** precision is 1.0, recall is 0.5, F1 is 2/3 and JAC is 0.5

#### Scenario: No matches
- **GIVEN** non-empty true and predicted sets with no matched pairs
- **THEN** precision, recall, F1 and JAC are all 0

### Requirement: Reference localization error
For each image with at least one matched pair, the reference RMSE SHALL be the per-coordinate root-mean-square error over matched pairs: `sqrt(sum(dy² + dx²) / n_matched / 2)` (source: `deepcell_spots/point_metrics.py:349-351`).

#### Scenario: Single pair at unit distance
- **GIVEN** one matched pair at distance 1 px
- **THEN** the reference RMSE is 1/√2 ≈ 0.7071 px

### Requirement: Reference sum of minimal distances
For each image with at least one matched pair, the reference d_md SHALL be computed over the matched points only, as the mean of (a) the mean distance from each matched true point to its nearest matched predicted point and (b) the reverse (source: `deepcell_spots/point_metrics.py:17-56`, `:356-357`).

#### Scenario: Unmatched outliers do not affect d_md
- **GIVEN** true points `[[0,0]]`, predicted points `[[0,0.5],[50,50]]` and τ = 1
- **THEN** the reference d_md is 0.5 px

### Requirement: Degenerate images
When the true set or the predicted set is empty, the reference metric set SHALL report precision, recall, F1 and JAC as 0. RMSE and d_md SHALL be reported as undefined (not 0) whenever no pair is matched (source: `deepcell_spots/point_metrics.py:308-323`, `:345-347`).

#### Scenario: No predictions
- **GIVEN** a non-empty true set and an empty predicted set
- **THEN** F1 is 0 and RMSE and d_md are undefined

#### Scenario: Both sets empty
- **GIVEN** empty true and predicted sets
- **THEN** F1 is 0 and RMSE and d_md are undefined

### Requirement: Macro aggregation across images
A test-set score SHALL be the unweighted mean of per-image values (macro average) for precision, recall, F1 and JAC over all scored images (source: `deepcell_spots/point_metrics.py:397-400`). RMSE and d_md SHALL be averaged over the images where they are defined, and the number of such images SHALL be reported.

#### Scenario: Two images
- **GIVEN** image A with F1 = 1.0 and image B with F1 = 0.5
- **THEN** the test-set F1 is 0.75, regardless of how many spots each image contains

#### Scenario: Undefined RMSE excluded
- **GIVEN** image A with RMSE 0.4 px and image B with no matched pairs
- **THEN** the test-set RMSE is 0.4 px and it is reported as computed over 1 of 2 images

### Requirement: Micro counts reported
Alongside the macro scores, a test-set result SHALL report total TP, FP and FN summed over images. The reference metric set SHALL label these as supplementary.

#### Scenario: Totals available
- **WHEN** a test-set result is produced
- **THEN** it includes summed TP, FP and FN, and FP = N_pred − TP and FN = N_true − TP for each image

### Requirement: Additional metrics are separated from the reference set
Metrics outside the reference set (for example optimal-assignment matching, distance percentiles, signed bias, or Chamfer distance over all points) MAY be reported. They SHALL be labelled as additional, SHALL NOT share a name with a reference metric, and SHALL NOT be used for benchmark or divergence decisions.

#### Scenario: Chamfer distance reported
- **WHEN** the all-points Chamfer distance is reported
- **THEN** it appears under its own name, distinct from the reference d_md

### Requirement: Signed bias labels
Signed localization bias, when reported, SHALL label the mean of (predicted − true) along the row axis as `bias_y` and along the column axis as `bias_x`.

#### Scenario: Vertical offset only
- **GIVEN** every prediction lies exactly 0.5 px below (larger row index than) its matched true point
- **THEN** `bias_y` is +0.5 and `bias_x` is 0
