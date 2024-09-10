<!-- 
Miguel notes
- Pre-trained model (Transfer Learning) : 
- My own CNN model
  - Neurons 
  - epochs
  - dis rate
- LSTM model
  - epochs
  - dis rate
- FRAME SELECTION
  - Most area of scan
  - Choose best middle frame of mri scan
  - Average image of 10 or so frames
- Max/Average Pooling
- PCA (Principal Component Analysis)

 -->
# Neurosight: Experimental Alzheimer's Detection Platform

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6%2B-yellow)
![Three.js](https://img.shields.io/badge/Three.js-Latest-lightgrey)

Neurosight is an experimental platform developed as part of an MSc dissertation project, exploring the potential of deep learning in Alzheimer's Disease detection from MRI brain scans. This project combines backend processing with an interactive front-end to investigate neurological image analysis techniques.

## Abstract

This research investigates the application of deep learning techniques to MRI brain scans for the detection of Alzheimer's Disease. While results were promising, they also highlighted significant challenges and limitations in this approach. The project developed a web-based platform for data visualization and analysis, contributing to the ongoing exploration of AI in neurological diagnostics.

## Key Findings

- Developed and tested multiple deep learning models for MRI analysis
- Implemented an interactive 3D MRI viewer for enhanced data exploration
- Identified limitations in using MRI scans alone for Alzheimer's detection
- Explored the potential of transfer learning and custom CNN architectures
- Highlighted the need for multi-modal approaches in neurological diagnostics

## Methodology

The project employed a multi-faceted approach:

1. Data Preprocessing: Skull stripping, image segmentation, and frame selection
2. Model Development: Custom CNNs, transfer learning, and LSTM models
3. Feature Engineering: Exploration of PCA and pooling techniques
4. Data Augmentation: Techniques to address limited dataset size
5. Visualization: Development of a web-based 3D MRI viewer

[Further details on methodology]

## Results and Discussion

While the models showed promise in certain aspects, overall results were not definitive. Key challenges included:

- Limited dataset size and quality
- Difficulty in differentiating Alzheimer's from other neurological conditions
- Potential overfitting due to data augmentation techniques

[Expand on results and their implications]

## Limitations and Future Work

This project highlighted several important limitations:

- MRI scans alone may not be sufficient for accurate Alzheimer's diagnosis
- Need for multi-modal data integration (e.g., PET scans, clinical data)
- Importance of larger, more diverse datasets

Future work could explore:

- Integration of additional imaging modalities and clinical data
- Longitudinal studies to track disease progression
- Exploration of explainable AI techniques for result interpretation

## Tech Stack

### Backend
- Python 3.8+
- TensorFlow 2.x
- Flask 2.x
- Pandas, NumPy, SciPy
- OpenCV
- Scikit-learn, Scikit-image
- Nibabel (for NIfTI file handling)

### Frontend
- HTML5, CSS3, JavaScript (ES6+)
- Three.js for 3D rendering
- Tailwind CSS for styling
- Chart.js for data visualization

### Data Processing
- Custom image preprocessing pipeline
- Advanced data augmentation techniques
- Transfer learning with pre-trained models

### Deployment
- Docker for containerization
- AWS S3 for data storage
- GitHub Actions for CI/CD

## Docker Setup

1. Build and run the Docker container:
   ```
   docker-compose up --build
   ```

2. Access the application at `http://localhost:5000`

## Development Setup

1. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

2. Install system dependencies (on Ubuntu/Debian):
   ```
   sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 tk
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Flask application:
   ```
   flask run
   ```

Visit `http://localhost:5000` in your browser to access the Neurosight platform.

## Usage

[Add detailed usage instructions here, including how to use the 3D viewer, upload scans, and interpret results]

## Data Description

Neurosight uses a combination of MRI scans and clinical data for analysis:

- MRI Scans: 3D structural brain scans in NIfTI format
- Clinical Data: Demographic information, cognitive test scores, and diagnosis labels

[Add more details about your dataset, including sources and preprocessing steps]

## Acknowledgements

This project was completed as part of an MSc dissertation at [University Name]. Special thanks to [Supervisor Name] for their guidance and support.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
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

This section will outline my thoughts behind what I've done and why I've done it for classifying the various types of data.

I've stored the MRI tabular data in Amazon's AWS cloud storage for safe keeping. Using various API keys, I can securely load in the data to this python script.

Before the data is given to a deep learning model, I need to clean and simplify the data such that there's enough data there to make a classification but not too much it's too much.

### Appendices

