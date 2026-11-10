# spot-decoding Specification

## Purpose

Defines the decoding module: how a codebook and the per-image spot probabilities from the detection hand-off become one gene assignment per spot, through a Relaxed Bernoulli mixture model fitted to each dataset, an Unknown threshold, two optional rescues, and an output spot table.

The baseline is the torch-spots decoding code at the start of this change (`torch_spots/decoding/decode.py`, `torch_spots/decoding/decoding_utils.py`), cited as `torch_spots/<file>:<line>`. Its numbers stay the same unless a later change justifies a difference under the `decoding-evaluation` evidence rules.

## Requirements

### Requirement: Codebook format
The codebook SHALL be a table with a 1-based index, a first column named `Gene` holding gene names, and B further columns holding a binary (0/1) barcode per gene, where B is the number of images in the hand-off. Anything else SHALL be rejected with an error naming the problem.

#### Scenario: Valid MERFISH codebook
- **WHEN** the pinned MERFISH codebook (265 rows, 20 bit columns, including 24 `Blank-*` rows) is loaded
- **THEN** it is accepted with B = 20

#### Scenario: Non-binary value
- **WHEN** a codebook contains the value 2 in a bit column
- **THEN** it is rejected with an error saying barcode values must be 0 or 1

### Requirement: Reserved class names
The names `Background` and `Unknown` SHALL be reserved. A codebook whose `Gene` column contains either name SHALL be rejected. The module SHALL append a `Background` class with an all-zero barcode and an `Unknown` class with an all −1 barcode after the codebook's own rows.

#### Scenario: Reserved name in the gene column
- **WHEN** a codebook has a row whose `Gene` value is `Background`
- **THEN** it is rejected with an error naming the reserved names

#### Scenario: Added classes
- **WHEN** a codebook with 265 rows is loaded
- **THEN** class 266 is `Background` and class 267 is `Unknown`

### Requirement: Decoding input
Decoding SHALL take spot probabilities of shape [n_spots, B] with every value in [0, 1], where column *i* matches codebook bit column *i*. A column count other than B, or any value outside [0, 1], SHALL be rejected with an error stating the expected and received shape or range.

#### Scenario: Column count mismatch
- **WHEN** probabilities with 18 columns are decoded against a 20-bit codebook
- **THEN** decoding stops with an error naming 20 expected and 18 received

#### Scenario: Value out of range
- **WHEN** probabilities contain 1.2
- **THEN** decoding stops with an error saying values must be in [0, 1]

### Requirement: Relaxed Bernoulli mixture model only
Decoding SHALL model each spot's probability vector as a draw from a mixture with one Relaxed Bernoulli component per codebook row plus `Background` (`Unknown` is not a component). No other distribution SHALL be offered, and there SHALL be no setting for choosing one.

#### Scenario: No distribution setting
- **WHEN** a user passes a distribution choice to decoding
- **THEN** it is rejected as an unknown argument

#### Scenario: Component count
- **WHEN** the pinned MERFISH codebook is used
- **THEN** the mixture has 266 components (265 codebook rows plus `Background`)

### Requirement: Mixture model parameters
The mixture weights SHALL start at 1/K and stay on the simplex. Each component's per-bit location SHALL be `b + (−1)^b · 0.3 · σ`, where b is the barcode bit, and its temperature τ SHALL be shared the same way as σ. Both σ and τ SHALL start at 0.5 and stay in [0, 1] (baseline: `torch_spots/decoding/decoding_utils.py:99-164,194`).

#### Scenario: Initial location
- **GIVEN** σ = 0.5 for a bit
- **WHEN** the component location is computed
- **THEN** it is 0.15 where b = 0 and 0.85 where b = 1

### Requirement: Parameter sharing modes
σ and τ SHALL each have one value per barcode state (0 or 1), further split by sharing mode: no split (`'2'`), per round (`'2*R'`), per channel (`'2*C'`), or per bit (`'2*R*C'`, the default). The per-round and per-channel modes SHALL group bits as in the baseline, a known limitation: bit columns are reshaped as [channel, round] but the model orders them [round, channel], so the bits sharing a value are not those of one round or channel. Fixing this is outside this change.

#### Scenario: Default mode
- **GIVEN** a 20-bit codebook
- **WHEN** decoding runs in the default mode
- **THEN** σ and τ each have 40 values, one per barcode state and bit

#### Scenario: Inherited grouping
- **GIVEN** rounds = 10, channels = 2 and sharing mode `'2*R'`
- **WHEN** bits are grouped
- **THEN** codebook bit columns 0 and 5 share a value, as do 1 and 6, and so on, and columns 0 and 10 do not (the baseline's column order, not one round per shared value)

#### Scenario: Invalid mode
- **WHEN** sharing mode `'3'` is requested
- **THEN** decoding stops with an error listing the valid modes

### Requirement: Model fitting settings
The model SHALL be fitted as a MAP estimate by stochastic variational inference, with the class variable summed out exactly. The optimizer SHALL be Adam with learning rate 0.085 and betas (0.85, 0.99). Fitting SHALL run 500 iterations by default, with minibatches of min(n_spots, 1000) spots by default (baseline: `torch_spots/decoding/decoding_utils.py:357-372`; `torch_spots/decoding/decode.py:375-376`).

#### Scenario: Defaults
- **WHEN** decoding runs without fitting overrides
- **THEN** it uses 500 iterations, a minibatch of min(n_spots, 1000), Adam learning rate 0.085 and betas (0.85, 0.99)

#### Scenario: Fewer spots than the minibatch
- **GIVEN** 600 spots
- **WHEN** decoding runs with the default minibatch size
- **THEN** each iteration uses all 600 spots

### Requirement: Decoding seed
The random seed for fitting SHALL be a setting, with default 1. Any earlier fitted state SHALL be discarded before fitting. Two runs with the same input, settings and seed on the same device SHALL give identical output.

#### Scenario: Repeat run
- **WHEN** the same probabilities are decoded twice with seed 0 on CPU
- **THEN** the two spot tables are identical

#### Scenario: Default seed
- **WHEN** decoding runs without a seed
- **THEN** the seed used is 1

### Requirement: Posterior assignment
After fitting, each spot SHALL be assigned the component with the highest posterior probability given the fitted weights, σ and τ, and its probability SHALL be that maximum posterior. Classes SHALL be numbered from 1 in codebook order, followed by `Background`.

#### Scenario: Clear barcode
- **GIVEN** a fitted model and a spot whose highest posterior is 0.99 for codebook row 46 (`Net1`)
- **WHEN** it is assigned
- **THEN** its class is 46, its name `Net1`, its probability 0.99 and its source `prediction`

### Requirement: Unknown threshold
A spot whose maximum posterior is below the Unknown threshold (default 0.95) SHALL be relabelled `Unknown`, keeping its maximum posterior as its probability.

#### Scenario: Low-confidence spot
- **GIVEN** a spot whose maximum posterior is 0.80, for gene `Apob`
- **WHEN** the default threshold is applied
- **THEN** it is labelled `Unknown`, with the `Unknown` class number and probability 0.80

### Requirement: Error rescue
Error rescue SHALL be on by default. For each spot labelled `Background` or `Unknown`, its probabilities SHALL be rounded to 0/1 (0.5 rounds to 0). If a class barcode (codebook rows, then `Background`, then `Unknown`) is at Hamming distance exactly 1, the first such class SHALL be assigned, with source `error rescue` and probability unchanged.

#### Scenario: One bit missing
- **GIVEN** an `Unknown` spot whose rounded vector equals gene `Cckar`'s barcode except one bit, and no earlier class is at distance 1
- **WHEN** error rescue runs
- **THEN** the spot is labelled `Cckar` with source `error rescue`

#### Scenario: Single lit bit
- **GIVEN** an `Unknown` spot whose rounded vector has exactly one 1 and is at distance 1 from no codebook row
- **WHEN** error rescue runs
- **THEN** it is labelled `Background` with source `error rescue` (distance 1 from the all-zero barcode)

#### Scenario: Rescue off
- **WHEN** decoding runs with error rescue off
- **THEN** no row has source `error rescue`

### Requirement: Mixed-barcode rescue
Mixed-barcode rescue SHALL be off by default. When on, for each spot with probability below 0.95 labelled with a codebook row, it SHALL zero that row's 1-bits in the spot's probabilities, round, and find the first class at Hamming distance 1. If one exists, it SHALL add a row with the same spot index, that class, probability −1 and source `mixed rescue`, keeping the original row.

#### Scenario: Two overlapping barcodes
- **GIVEN** rescue on, and a spot labelled gene A at probability 0.9 whose rounded probabilities with A's bits zeroed are 1 bit from gene B
- **WHEN** mixed rescue runs
- **THEN** the table has the original row for A and an added row for B with the same spot index, probability −1 and source `mixed rescue`

### Requirement: Output spot table
Decoding SHALL return one table with columns `spot_index`, `y`, `x`, `predicted_id`, `predicted_name`, `probability`, `source`, and one column per bit named `prob_<codebook column name>`. The rows SHALL be the input spots in input order, followed by any rows added by rescue. `y` and `x` SHALL be omitted when no locations are given.

#### Scenario: Columns
- **WHEN** the MERFISH example is decoded with locations
- **THEN** the table has the seven named columns followed by `prob_RS0015` … `prob_RS0707`

#### Scenario: No locations given
- **WHEN** probabilities are decoded without locations
- **THEN** the table has no `y` or `x` column and all other columns are present

### Requirement: Locations follow the spot index
Each row's `y`, `x` and `prob_*` values SHALL be those of the input spot named by its `spot_index`, including rows added by mixed rescue. `y` is the pixel row and `x` the column, as in the hand-off's `[y, x]` locations.

#### Scenario: Added row
- **GIVEN** mixed rescue adds a row for input spot 7, located at `[120, 45]`
- **WHEN** the table is built
- **THEN** the added row has `spot_index` 7, `y` 120 and `x` 45, and the same `prob_*` values as spot 7's original row

### Requirement: Genes-only output
A `genes_only` switch, off by default, SHALL remove every row whose `predicted_name` is `Background` or `Unknown`, or contains `Blank`. It SHALL NOT change any other row.

#### Scenario: Default keeps all rows
- **WHEN** decoding runs without the switch
- **THEN** rows labelled `Background`, `Unknown` and `Blank-*` are present

#### Scenario: Switch on
- **WHEN** the same input is decoded with the switch on, at the same seed
- **THEN** the table equals the default table without its `Background`, `Unknown` and `Blank-*` rows

### Requirement: Decoding settings reported
The result SHALL make available every decoding setting used (sharing mode, iterations, minibatch size, seed, Unknown threshold, both rescue switches, `genes_only`), together with the fitted weights, σ, τ and the per-iteration loss.

#### Scenario: Settings and fit available
- **WHEN** decoding finishes
- **THEN** the settings above and the fitted parameters and losses can be read from the result without rerunning
