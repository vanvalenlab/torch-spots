# decoding-evaluation Specification

## Purpose

Defines how changes to the decoding module and to the detection-to-decoding hand-off are judged. It covers the pinned example data and detection checkpoint, the decoding seed protocol, the blank-barcode rate and gene-call count, the candidate-threshold sweep, and the evidence each kind of decoding change needs before it is adopted.

The baseline configuration is the hand-off and decoding defaults defined by the `decoding-input` and `spot-decoding` capabilities.

## Requirements

### Requirement: Pinned decoding example
Decoding evaluation SHALL use the MERFISH example image (`SpotNetExampleData-v1_0/MERFISH_example.npz`, array `spots_image`, shape (1, 500, 500, 20), MD5 `2cd7ce177b503fd0873125784097622b`). Its last axis, moved first, SHALL be the image stack. The codebook SHALL be `SpotNetExampleData-v1_0/MERFISH_codebook.csv` (MD5 `56f107ed562b73dc0037174c18f6f4dd`). A file with a different checksum SHALL stop evaluation.

#### Scenario: Checksum mismatch
- **WHEN** the example image's MD5 differs from the pinned value
- **THEN** evaluation stops with an error naming the file and both checksums, and reports no result

### Requirement: Pinned detection checkpoint
Decoding evaluation SHALL run detection with the release checkpoint (`data/release/saved_model_best_dict.pth`, round 8 seed 1, MD5 `f4713d8a7c552e68eecf63a880259ef1`, architecture version 2) and detection's default inference preprocessing.

#### Scenario: Default run
- **WHEN** decoding evaluation runs without overrides
- **THEN** detection uses the release checkpoint, and its checksum and architecture version are recorded with the result

### Requirement: Stored hand-off
For each hand-off configuration, detection and the hand-off SHALL run once, and their output (locations, probabilities, hand-off settings) SHALL be stored with its checksum. Every decoding run of that configuration SHALL read the stored hand-off, so seeds and decoding changes are compared on identical input.

#### Scenario: Decoding-only change
- **GIVEN** a stored baseline hand-off
- **WHEN** a change to decoding alone is evaluated
- **THEN** it is decoded from the stored hand-off, whose checksum is recorded with the result

#### Scenario: Hand-off change
- **WHEN** a change to the candidate threshold or pool size is evaluated
- **THEN** a new hand-off is produced and stored, and the result records its checksum

### Requirement: Decoding seed protocol
Every decoding configuration scored for a decision SHALL be decoded with seeds 0, 1 and 2 from the same stored hand-off and the same detection checkpoint. Each seed SHALL be scored separately, and each measure SHALL be reported per seed, with its mean and range (max − min).

#### Scenario: Reporting a configuration
- **WHEN** a decoding configuration's result is reported
- **THEN** it lists, for seeds 0, 1 and 2, the gene-call count and blank rate, plus the mean and range of each

### Requirement: Accuracy measures
Measures SHALL be computed on the full spot table (with `genes_only` off), counting every row, rescue rows included. A gene call is a row whose name is a codebook gene, and a blank call is a row whose name contains `Blank`. The blank rate SHALL be blank calls / (gene calls + blank calls).

#### Scenario: Counting
- **GIVEN** a table with 1,000 gene calls, 20 blank calls, 300 `Unknown` rows and 500 `Background` rows
- **WHEN** the measures are computed
- **THEN** the gene-call count is 1,000 and the blank rate is 20 / 1,020 = 0.0196

### Requirement: Reported breakdown
Each seed's result SHALL also report: the number of candidates; `Background` and `Unknown` row counts; gene and blank calls split by source (`prediction`, `error rescue`, `mixed rescue`); decoding wall-clock time; and peak memory.

#### Scenario: Breakdown present
- **WHEN** a seed's result is recorded
- **THEN** it contains each of the counts above, the time and the peak memory

### Requirement: Noise margin
The noise margin of a measure SHALL be the baseline's range of that measure over seeds 0, 1 and 2. If that range is 0, the margin SHALL be the effect of one call: 1 for the gene-call count, and 1 / (mean gene calls + mean blank calls) for the blank rate.

#### Scenario: Zero range
- **GIVEN** baseline gene-call counts of 1,200 at every seed, and mean blank calls 24
- **WHEN** the margins are computed
- **THEN** the gene-call margin is 1 and the blank-rate margin is 1 / 1,224

### Requirement: Accuracy divergence evidence
A decoding or hand-off change claiming better accuracy SHALL be adopted only if, against the baseline under the same protocol, one measure improves by more than its margin and the other does not worsen by more than its margin. For the blank rate, improving means a lower mean; for the gene-call count, a higher mean.

#### Scenario: Fewer false calls
- **GIVEN** baseline means of 1,200 gene calls and blank rate 0.020, with margins 15 and 0.002
- **WHEN** a change has means of 1,190 gene calls and blank rate 0.015
- **THEN** the accuracy claim is supported

#### Scenario: Fewer calls overall
- **GIVEN** the same baseline
- **WHEN** a change has means of 1,000 gene calls and blank rate 0.012
- **THEN** the claim is not supported, because the gene-call count fell by more than its margin

### Requirement: Quality-of-life divergence evidence
A change claiming a benefit other than accuracy (speed, memory, readability, usability) whose output differs from the baseline SHALL be adopted only if neither measure worsens by more than its margin and the benefit is measured and reported. Time and memory SHALL be measured on the same machine as the baseline, under the same thread limits.

#### Scenario: Faster and non-inferior
- **GIVEN** the baseline above
- **WHEN** a change has means of 1,195 gene calls and blank rate 0.021, and a measured 40% shorter decoding time
- **THEN** it may be adopted

#### Scenario: Benefit not measured
- **WHEN** a change is non-inferior but its benefit has no measurement
- **THEN** it is not adopted

### Requirement: Output-identical change evidence
A change that claims not to alter results (a rename, a new output-only switch left off, a restructure) SHALL be adopted only if, at the default seed (1) on the stored baseline hand-off, its spot table equals the baseline's in every row and value, after applying the change's documented column renames. No seed protocol is needed.

#### Scenario: Rename
- **GIVEN** a change that only renames public arguments
- **WHEN** its seed-1 table is compared with the baseline's
- **THEN** every row and value is equal, and the change may be adopted

#### Scenario: Any difference
- **WHEN** one row's `predicted_name` differs
- **THEN** the change is not output-identical and needs accuracy or quality-of-life evidence

### Requirement: Candidate-threshold sweep
A sweep SHALL score candidate thresholds 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99 and 0.993 under the seed protocol, reporting each threshold's measures and breakdown. The baseline 0.9 SHALL be reported separately. A new default SHALL need accuracy divergence evidence against 0.9.

#### Scenario: Sweep output
- **WHEN** the sweep finishes
- **THEN** each of the 8 thresholds has, per seed, the gene-call count, blank rate and candidate count, and 0.9 is marked as the baseline

### Requirement: Other detection checkpoints reported separately
The sweep and evaluation SHALL accept other detection checkpoints, such as round 8 seeds 0 and 2. Their results SHALL be labelled with the checkpoint and reported apart from the pinned-checkpoint results, and SHALL NOT be used as evidence under the divergence rules.

#### Scenario: Seed-0 detection checkpoint
- **WHEN** the sweep is run with the round 8 seed 0 checkpoint
- **THEN** its results name that checkpoint and do not appear in the pinned-checkpoint report

### Requirement: Decoding result provenance
Every decoding result SHALL record the code revision, every hand-off and decoding setting, the seeds, the checksums of the image, codebook, detection checkpoint and stored hand-off, the device, thread limits, and the torch and pyro versions. A result missing any of these SHALL NOT be used for a decision.

#### Scenario: Reproducing a result
- **WHEN** someone reads a recorded decoding result
- **THEN** they can identify the commit, inputs, settings and seeds needed to rerun it

### Requirement: Recorded baseline
A baseline result produced under this protocol, from the baseline configuration with the stored baseline hand-off, SHALL be recorded. Decisions SHALL compare against a recorded baseline that uses the same stored hand-off (for decoding-only changes) and, for time or memory claims, the same machine and thread limits.

#### Scenario: Time claim on another machine
- **GIVEN** a recorded baseline timed on machine A with 4 threads
- **WHEN** a change claims a speed-up measured on machine B
- **THEN** the claim is refused until the baseline is also timed on machine B with the same thread limits

#### Scenario: Device change
- **GIVEN** a change that moves fitting to a GPU
- **WHEN** its accuracy measures are compared with the CPU baseline
- **THEN** the comparison is allowed, and the change is judged under the quality-of-life rule
