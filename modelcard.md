---
license: other
license_name: modified-apache-2.0-noncommercial
license_link: https://github.com/vanvalenlab/deepcell-auth/blob/main/ASSET_LICENSE
---

# Model Card for Model ID

This is a model that detects sub-pixel coordinates for spots in spatial transcriptomics experiments. It is based on a FeatureNet backbone with semantic heads predicting spot probability and subpixel offset.

## Model Details

### Model Description




- **Developed by:** The Van Valen Lab
- **Model type:** weakly-supervised deep learning spot detection
- **Language(s) (NLP):** en
- **License:** [Modified Apache 2.0 Noncommercial](https://github.com/vanvalenlab/deepcell-auth/blob/main/ASSET_LICENSE)

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [Github](https://github.com/vanvalenlab/torch-spots)
- **Paper [optional]:** [Accurate single-molecule spot detection for image-based spatial transcriptomics with weakly supervised deep learning](https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00121-2)

## Uses

This model is useful for fast and accurate detection of spots generated in spatial transcriptomics datasets. Spots constitute single "reads" that, over many rounds, code for individual gene products. This model takes in an image of any shape and identifies the spots at sub-pixel resolution.

### Direct Use

This model can be used out of the box in the following way: 

1. `pip` install the package from the github repository.
2. Ensure you have a Hugging Face API key connected to the computer you want to do inference on.
3. When you instantiate the `SpotDetection` application, the model weights will be downloaded directly to your computer in the canonical `~/.deepcell` location.
4. Run the `predict_points` method by supplying an image of shape `(B, 1, H, W)`
5. The points will be returned as an `(N,2)` array containing the point coordinates.

### Out-of-Scope Use

Spot detection has been trained, evaluated, and benchmarked on spatial transcriptomics data. Attempting to detect spots from immunofluorescence or any other modality may yield incorrect results.

## Bias, Risks, and Limitations

As stated previously, the spot detection has been trained, evaluated and benchmarked on spatial transcriptomic data. This means that, although other data may form foci that look similar, the results will not be accurate.

The model tiles the input image under the hood, conducts inference on each `(128x128)` tile of the image. This is done batchwise, so one should limit the batch number if vRAM is at a premium. In addition, the current implementation requires the user to read the whole image into memory before inference, so if RAM is also at a premium, consider tiling the full image before sending a smaller tile through the `predict_points` method.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import tifffile
from torch_spots.detection.inference import SpotDetection
app = SpotDetection() # This will download the model weights if necessary

img = tifffile.imread('your_image.tiff')

print(img.shape) 

# should be shape 1, 1, H, W. 
# If multi-channel, process each channel separately

spots = app.predict_points(img)

```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Model Card Contact

[More Information Needed]