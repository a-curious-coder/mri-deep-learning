# Alzheimer's Classification

![Python](https://img.shields.io/badge/Made%20by-my%20brain-blue)

Alzheimer's in an incurable disease ravaging the elderly population. Through the power of deep learning, I hope to build a neural network that detects Alzheimer's in MRI scans of the brain.

## Table of Contents
---

- [Alzheimer's Classification](#alzheimer-s-classification)
  * [Table of Contents](#table-of-contents)
  * [Setup](#setup)
  * [Dataset Description](#dataset-description)
    + [Images](#images)
  * [Objectives](#objectives)
    + [Frame Selection](#frame-selection)
      - [How to tackle this problem?](#how-to-tackle-this-problem-)
      - [Problems / Solutions](#problems---solutions)
    + [Image segmentation](#image-segmentation)
    + [Image Augmentation](#image-augmentation)
    + [Image Fabrication](#image-fabrication)
  * [Problems](#problems)
    + [1. How to select the best frames from scans?](#1-how-to-select-the-best-frames-from-scans-)
  * [Limitations](#limitations)
    + [Are MRI Scans Appropriate Data](#are-mri-scans-appropriate-data)
  * [Exploratory Data Analysis](#exploratory-data-analysis)
    + [Tabular Data](#tabular-data)
  * [Process](#process)
    + [Appendices](#appendices)

## Setup
---
Create an [AWS](https://aws.amazon.com) account, create a user profile through IAM and create a storage bucket through S3. Upload your MRI data to the bucket and change the load_data function according to your bucket name and directories to file(s)

Create and initialise virtual environment

```bash
# Create virtual environment
virtualenv .venv

# Initialise .venv WINDOWS
.venv\Scripts\activate
# Initialise .venv MAC OSX
source .venv/bin/activate
```

Download and install all required libraries

```bash
pip install -r requirements.txt
```

Create ".env" file and populate it with variables containing your AWS api keys.

```bash
ACCESS_KEY=*API KEY GOES HERE EXCLUDING ASTERISKS*
SECRET_ACCESS_KEY=*SECRET API KEY GOES HERE EXCLUDING ASTERISKS*
```

Run the main python file

```bash
python main.py
```

## Dataset Description
---
There are two datasets: Images and Tabular Data.

### Images
I've been provided with around 6000 MRI scans of peoples brains. There's a small variance regarding their resolutions, however there's a consistency in that all scans have the quantity of frames; 256. The tabular data dataset is a collection of ADNI scans for the brain. 

There are three angles for each scan: axial, coronal and sagittal. I suspect I will need to train several models, one for each angle. 



## Objectives
---
### Frame Selection

When classifying any disease/problem with the brain via an MRI scan, there are specific frame(s) selected to analysis by the doctor. The goal here is to teach the computer to select these frame(s) independently throughout all 6000 scans. Accuracy here is important as we want to optimise the data we pass to the final CNN model when predicting Alzheimer's during experimentation.

#### How to tackle this problem?
Choosing the appropriate frame / frames from an MRI scan for the task of diagnosis is important as it is the key to the success of the model. The goal is to select the best frame(s) from each MRI scan to provide the best training data for the CNN model. 

To save time with potential minor sacrifice to accuracy, I have chosen to train a CNN model on pre-existing brain MRI scan data-sets sourced online only consisting of the frames used in kaggle projects. The data-sets are available from the [Kaggle](https://www.kaggle.com/code/hachemsfar/alzheimer-mri-model-data-exploration/data).

#### Problems / Solutions
I found the data-set I downloaded from kaggle provided images with various diagnosis (which is fine) however, they all shared the same resolution; my mri scans didn't. To solve this, I will have to reshape/resize and potentially sheer my MRI scan images to match the resolution of the training data-set. 

I will potentially rescale the brain images and fill in any blank space with black pixels (values of 0) which will be ignored by the trained CNN when stripping the skull. This is to maintain the aspect ratio of the brain scans; thus maintaining as much information as we can of each patient's brain.

### Image segmentation

The main goal of segmentation in the context of MRI brain scans is to separate the brain from the skull in each of the images. This is done to keep only the relevant data in each image when passing it to a CNN.

<!-- Link papers talking about scan segmentation -->
<!-- https://pubmed.ncbi.nlm.nih.gov/25945121/ -->

### Image Augmentation

To increase the size of the training set, the training images will need to be augmented so there are multiple variations of each MRI scan. The idea of this is to provide more data to the Neural Network for Alzheimer's classification.

### Image Fabrication

General Adversial Neural Networks are popular for fabricating entirely new data. The goal is to provide a GAN with labelled brain images (Each label representing the angle of the scan) as to generate further training data.

## Problems
---
### 1. How to select the best frames from scans?
The initial goal was difficult to determine without more information from the literature. I can select the appropriate frames first or strip the skull from each frame in each scan to select the brain. If we select the appropriate frames first, we can then strip the skull from these images which is far less data to process versus skull stripping entire scans with 256 frames each. 

## Limitations
---
### Are MRI Scans Appropriate Data

During the research phase of this project, I came across multiple papers sharing similar results in regards to the effectiveness of MRI brain scans for identifying Alzheimer's. The general consensus is that deep learning models are able to identify Alzheimer's from MRI scans of the brain; however, if the model is shown a brain scan of a disease it has been trained on, it struggles to differentiate between the different types of disease, thus consistently misclassifying MRI scans, having a detrimental impact on the accuracy results.

Furthermore, in the real world, in the context of brain scans, doctors traditionally use CT Scans over MRI scans as they're better suited for identifying diseases. A final note... Alzheimer's Disease begins in the gut of humans, not the brain. Thus, I feel this project could have used scans of alternative regions of the body to identify the disease before it reaches the brain.

<!-- ## What is Alzheimer's Disease? -->

## Exploratory Data Analysis

---

Classification of Alzheimer's Disease from the MRI datasets is a binary classification problem as the patients either have Alzheimer's Disease (AD) or they are Cognitive Normal (CN).

### Tabular Data

Each patient is represented by a unique Security ID (SID). Each patient has had at least 2 scan sessions, thus has 2 or more MRI scans. In the effort of dimensionality reduction, I should try and represent each patient within a single row.

<!-- Pie chart for Diagnosis to represent binary ratio -->
<!--  Will Cross Validation be needed due to size/ratio of diagnosis -->
<!-- Something to do with Age -->


## Process

---

This section will outline my thoughts behind what I've done and why I've done it for classifying the various types of data.

I've stored the MRI tabular data in Amazon's AWS cloud storage for safe keeping. Using various API keys, I can securely load in the data to this python script.

Before the data is given to a deep learning model, I need to clean and simplify the data such that there's enough data there to make a classification but not too much it's too much.


### Appendices