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
      - [Tackling the problem](#how-to-tackle-this-problem-)
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
I've sourced data from <sources> however, ADNI required a request to access to data; thus making it private.

### Images

I've been provided with around 6000 MRI scans of peoples brains. There's a small variance regarding their resolutions, however there's a consistency in that all scans have the quantity of frames; 256. The tabular data dataset is a collection of ADNI scans for the brain. 

There are three angles for each scan: axial, coronal and sagittal. I suspect I will need to train several models, one for each angle. 

## Literature Review
<!-- NEW TERMS -->
<!-- MULTIMODAL - https://towardsdatascience.com/multimodal-deep-learning-ce7d1d994f4 -->
<!-- ATTENTION MECHANISM - https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/ -->
<!-- SELF ATTENTION -->
<!-- CROSS MODALITY ATTENTION -->

Classification of Alzheimer's Disease from the MRI datasets is a binary classification problem as the patients either have Alzheimer's Disease (AD) or they are Cognitive Normal (CN).

### Literature 1
<!-- Paper Link 1 : https://ieeexplore.ieee.org/abstract/document/7950647?casa%5C_token=nOv1QYjkYwkAAAAA:Ln3Jr6dUyYV7iyCMs9wMQWKwPb4Vz-UwdyI3QfUcRCon6vqwE2FsE%5C_ctoOhfu6bk9XFg9wvW-->

#### Data
The author uses a specific subset of 3d structural MRI brain scans from ADNI which has been *preprocessed with alignment and skull-stripping marked as “Spatially Normalized, Masked and N3 corrected T1 images"*. The data was used to train a neural network that detects Alzheimer's Disease from the MRI scans. To mitigate data leaks, the author ensured only the first scan of a patient was used since using multiple scans from the patients who had them would result in a slight bias.

The data-set used consisted of 231 images over 4 classes. 
> 50 of Alzheimer’s Disease (AD) patients
  43 of Late Mild Cognitive Impairment (LMCI)
  77 of Early Mild Cognitive Impairment (EMCI) 
  61 of Normal Cohort (NC)


<!-- Data Analysis / Visualisation -->
The author acknowledges various details of the data and provides minimal visualisations of this data to help convey the data's nature.

<!-- Preprocessing Method -->
Data came pre-processed.
<!-- What models were used -->
The author created 6 binary classification tasks using all combinations of class labels against eachother and the results are shown in Table 1 within the paper.
<!-- Experimentation / Results -->

<!-- Limitations/Improvements -->

The author uses 3D structural MRI brain scans as the data-set to train and test the deep learning models used for classification. This data was sourced from ADNI (Alzheimer's Disease Neuroimaging Initiative) and has 4 labels

The author(s) of this paper have moved away from the obvious multi-class classification and created 6 binary classification problems for each permutation of the labels.

Since an MRI scan has different 'layers' in the scan, the first image for each patient was retrieved, thus the same area/layer of the brain was used for each patient as the data-set. This data was preprocessed using 'alignment' and 'skull-stripping'  which are marked as “Spatially Normalized, Masked and N3 corrected T1 images”. This removes unnecessary information from the data, allowing the classification methods to handle the data that's important / relevant.

Due to the complexity of the data, deep learning methods such as Convolutional and Residual neural networks are currently the ideal candidates for handling and classifying this image data. Standard machine learning methods would be limited in their capabilities of classifying new MRI image data.

% Comparison of models / model prediction accuracy (Which models were better)
\citeauthor{paper1}\cite{paper1} developed a few different classification models to classify Alzheimer's in MRI scans. VoxCNN and Resnet
% Transfer learning? (Used pre-trained models) Isolated Learning?
The models described in this paper are examples of isolated learning meaning they were trained from scratch.

### Literature 2
<!-- Paper Link 2 : https://www.nature.com/articles/s41467-022-31037-5 -->
This [paper](https://www.nature.com/articles/s41467-022-31037-5) was published June 20, 2022.

Considering this is a multilabel classification problem, author has split the problem into a bunch of binary classification subtasks as to delineate the different types of AD. E.g. separate MCI from NC/DE, separate DE from NC/MCI and NC from MCI/DE cases.

Moreover, author has included a Non-imaging classification model that takes scalar-valued clinical variables such as past medical history, neuropsychological scores and other clinical variables and predicts the AD status of the patient.

<!-- What data was used -->
The author has managed to access data from eight separate sources, thus accumulating a lot of data to train various machine/deep learning models with. These sources are: AIBL, ADNI, FHS, LBDSU, NACC, NIFD, OASIS, PPMI. Data from ADNI, AIBL, NACC, NIFD, OASIS and PPMI are public. However, data from FHS and LBDSU have to be requested.

<!-- Pros/Cons of the data?-->
Table 1 in the paper shows various statistics of each feature in each dataset. All of the datasets have multiple classes and the data is not balanced. However, with regards to gender for each diagnosis, 

<!-- What models were used -->
CNN, Catboost and a Fusion of the two.

<!-- Experimentation / Results -->
Initially separated the problem into three separate binary classificatiion tasks in the effort to effectively delineate each of the three labels; NC, MCI, AD.

<!-- What is the best model? -->
The best model is the fusion of the image (CNN) and non-image models (CatBoost). 

<!-- Limitations/Improvements -->
The author acknowledges an improvement for the project; to allow for the identification of co-occuring dementing conditions within the same individual.
- Wait for further data that allows for the investigation of identifying the distinct signatures within diagnosed MCI patients who have prodromal AD.
- Since the data is retrieved from studies that primarily collect data focussing on Alzheimer's disease, it could detract from the accuracy in nADDs diagnosis'.
- MMSE; well-known limitations in specificity, may bias toward more common forms of dementia like AD.
- Future models could be optimised provided additional clinical data tailored to their diagnosis; further specific tests. E.g. motor examination to assess parkinsons.



### Literature 3
<!-- Paper Link 3 : https://arxiv.org/abs/2206.08826-->
This [paper](https://arxiv.org/abs/2206.08826) was published June 17, 2022.
<!-- What data was used -->
The author used clinical, genetic and image datasets sourced from ADNI (Alzheimer's Disease Something Something). 
<!-- Pros/Cons of the data?-->
Albeit rare to have a perfectly balanced data-set, there were imbalances in the labels across all dataset types.
<!-- What models were used -->
Takes a multimodal approach using separate neural networks for each type of data (two feedforward, one CNN). 
<!-- How were the models trained and tested -->
To train the CNN, the author took the middle slice from each scan of each patient at each of the three angles. 
<!-- Experimentation / Results -->

<!-- Limitations/Improvements -->
After speaking with the author, Michal, of the paper, she stated that certain things weren't done such as skull-stripping, improved slice selection or any other pre-processing due to time limitations which is mainly due to working with other data. The impact the other data's predictive models had on the overall classification far outweighed further time expenditure on the intracies of the image classification model; despite the image model providing the highest independent accuracy.



## Objectives

### Skull stripping

As a form of pre-processing to optimise the training of the CNN, segmenting the brain from the skull of each mri-scan leaves the important and necessary data behind. Any other features other than the brain could negatively influence the classification of the patient as the skull doesn't exhibit any differences whether the patient has alzheimer's or not; this won't be obvious to a computer.

#### Tackling the problem
There are various techniques to segmenting objects from surrounding objects in images; this will likely require a deep learning model. 


### Frame Selection

When classifying any disease/problem with the brain via an MRI scan, there are specific frame(s) selected to analysis by the doctor. The goal here is to teach the computer to select these frame(s) independently throughout all 6000 scans. Accuracy here is important as we want to optimise the data we pass to the final CNN model when predicting Alzheimer's during experimentation.

#### Tackling the problem

Manually select *x* amount of scan images from mri scan files to create a data-set to train a CNN on which will be used to predict/select the better frames for Alzheimer's classification in new MRI scans. Might require data augmentation to compound number of images are in the data-set.

To save time with potential minor sacrifice to accuracy, I have chosen to train a CNN model on pre-existing brain MRI scan data-sets sourced online only consisting of the frames used in kaggle projects. The data-sets are available from the [Kaggle](https://www.kaggle.com/code/hachemsfar/alzheimer-mri-model-data-exploration/data).

#### Problems / Solutions

I found the data-set I downloaded from kaggle provided images with various diagnosis (which is fine) however, they all shared the same resolution; my mri scans didn't. To solve this, I will have to reshape/resize and potentially sheer my MRI scan images to match the resolution of the training data-set. 

I will potentially rescale the brain images and fill in any blank space with black pixels (values of 0) which will be ignored by the trained CNN when stripping the skull. This is to maintain the aspect ratio of the brain scans; thus maintaining as much information as we can of each patient's brain.

### Image segmentation

The main goal of segmentation in the context of MRI brain scans is to separate the brain from the skull in each of the images. This is done to keep only the relevant data in each image when passing it to a CNN.

<!-- Link papers talking about scan segmentation -->
<!-- https://pubmed.ncbi.nlm.nih.gov/25945121/ -->
<!-- https://ieeexplore.ieee.org/abstract/document/9234262-->

### Image Augmentation

To increase the size of the training set, the training images will need to be augmented so there are multiple variations of each MRI scan. The idea of this is to provide more data to the Neural Network for Alzheimer's classification.

### Image Fabrication

General Adversial Neural Networks are popular for fabricating entirely new data. The goal is to provide a GAN with labelled brain images (Each label representing the angle of the scan) as to generate further training data.

## Problems

### 1. How to select the best frames from scans?

The initial goal was difficult to determine without more information from the literature. I can select the appropriate frames first or strip the skull from each frame in each scan to select the brain. If we select the appropriate frames first, we can then strip the skull from these images which is far less data to process versus skull stripping entire scans with 256 frames each. 

## Limitations

### Are MRI Scans Appropriate Data

During the research phase of this project, I came across multiple papers sharing similar results in regards to the effectiveness of MRI brain scans for identifying Alzheimer's. The general consensus is that deep learning models are able to identify Alzheimer's from MRI scans of the brain; however, if the model is shown a brain scan of a disease it has been trained on, it struggles to differentiate between the different types of disease, thus consistently misclassifying MRI scans, having a detrimental impact on the accuracy results.

Furthermore, in the real world, in the context of brain scans, doctors traditionally use CT Scans over MRI scans as they're better suited for identifying diseases. A final note... Alzheimer's Disease begins in the gut of humans, not the brain. Thus, I feel this project could have used scans of alternative regions of the body to identify the disease before it reaches the brain.

<!-- ## What is Alzheimer's Disease? -->

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
