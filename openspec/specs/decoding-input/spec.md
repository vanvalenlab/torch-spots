# decoding-input Specification

## Purpose

Defines what the detection module hands to the decoding module for one field of view: the candidate spot locations, the candidate threshold that selects them, and the per-image spot probabilities read at each location.

Baseline citations refer to the torch-spots code at the start of this change as `torch_spots/<file>:<line>`.

## Requirements

### Requirement: Image stack for decoding
The hand-off SHALL take the detection output for one field of view, given as a stack of N images, one per codebook bit. The stack order SHALL be the codebook's column order. The spot-class probability map (channel 1 of the detection output) SHALL be the only value read from each image.

#### Scenario: MERFISH example stack
- **GIVEN** the MERFISH example image of shape (1, 500, 500, 20) with its last axis moved first
- **WHEN** it is passed through detection and handed off
- **THEN** 20 images are read, and image *i* corresponds to codebook column *i*

### Requirement: Candidate locations
Candidate locations SHALL be the local maxima of the pixel-wise maximum, over all N images, of the spot-class probability, keeping peaks above the candidate threshold with a minimum separation of 1 px. Each location SHALL be an integer `[y, x]` pixel (row, column), and the regressed offsets SHALL NOT be added (baseline: `torch_spots/detection/inference.py:62-78`).

#### Scenario: Peak in one image only
- **GIVEN** a pixel whose spot probability is 0.97 in image 3 and below 0.1 in every other image, and is a local maximum of the projection
- **WHEN** candidates are selected with the default candidate threshold
- **THEN** that pixel is a candidate

#### Scenario: Integer locations
- **WHEN** candidates are returned
- **THEN** every location is an integer `[y, x]` pixel inside the image

### Requirement: Separate candidate threshold
The candidate threshold SHALL be a setting of the hand-off, with default 0.9. It SHALL be independent of detection's evaluation threshold (0.993) and of the default threshold for plain point prediction. Changing either of those SHALL NOT change the candidate threshold.

#### Scenario: Default
- **WHEN** candidates are selected without an explicit threshold
- **THEN** the threshold used is 0.9

#### Scenario: Tuned threshold
- **WHEN** candidates are selected with threshold 0.7
- **THEN** every pixel with projected probability above 0.7 that is a local maximum is a candidate, including those between 0.7 and 0.9

### Requirement: Per-image spot probabilities
For each candidate, the hand-off SHALL return one value per image: the spot-class probability at that pixel. The result SHALL have shape [n_candidates, N], with column *i* from image *i*, and values in [0, 1]. An optional pool size *p* (default 0) SHALL instead take the maximum over the (2*p* + 1)² window, clipped to the image border.

#### Scenario: Default pool
- **GIVEN** a candidate at `[10, 20]`
- **WHEN** probabilities are extracted with the default pool size
- **THEN** column *i* equals image *i*'s spot probability at row 10, column 20

#### Scenario: Pool at the border
- **GIVEN** a candidate at `[0, 0]` and pool size 1
- **WHEN** probabilities are extracted
- **THEN** each value is the maximum over rows 0–1 and columns 0–1 only

### Requirement: Probabilities named as probabilities
The hand-off's public names, documentation and returned values SHALL describe the per-image values as spot probabilities, never as intensities. No public name containing "intensit" SHALL remain in the detection-to-decoding path.

#### Scenario: Public names
- **WHEN** the detection and decoding public interfaces and their documentation are searched for "intensit"
- **THEN** there are no matches

### Requirement: Hand-off settings reported
The hand-off SHALL return, with the locations and probabilities, the candidate threshold, minimum separation and pool size it used.

#### Scenario: Settings travel with the result
- **WHEN** candidates are selected with threshold 0.8
- **THEN** the returned settings include candidate threshold 0.8, minimum separation 1 px and pool size 0
