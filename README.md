# MRI Deep Learning
I hope to develop a deep learning model that classifies abnormalities (Specifically Alzheimer's Disease) in MRI scans. 
## Setup
Create AWS account, create user profile through IAM and create a bucket through S3. Upload MRI data to bucket and change the load_data function according to your bucket name and directories to file(s)

Create a virtual environment
```bash
virtualenv .venv
```
Download and install all required libraries
```bash
pip install -r requirements.txt
```
Create ".env" file and populate it with your AWS api keys within the following environment variables:
```bash
ACCESS_KEY=*API KEY GOES HERE EXCLUDING ASTERISKS*

SECRET_ACCESS_KEY=*SECRET API KEY GOES HERE EXCLUDING ASTERISKS*
```
Run the python file
```bash
python main.py
```
<!-- ## What is Alzheimer's Disease?p -->

## Exploratory Data Analysis
Classification of Alzheimer's Disease from the MRI datasets is a binary classification problem as the patients either have Alzheimer's Disease (AD) or they are Cognitive Normal (CN).
### Tabular Data (ADNI/IXI)
Each patient is represented by a unique Security ID (SID). Each patient has had at least 2 scan sessions, thus has 2 or more MRI scans. In the effort of dimensionality reduction, I should try and represent each patient within a single row.
<!-- Pie chart for Diagnosis to represent binary ratio -->
<!--  Will Cross Validation be needed due to size/ratio of diagnosis -->
<!-- Something to do with Age -->

## Process
This section will outline my thoughts behind what I've done and why I've done it for classifying the various types of data.
### Tabular Data (ADNI/IXI)
I've stored the MRI tabular data in Amazon's AWS cloud storage for safe keeping. Using various API keys, I can securely load in the data to this python script.

Before the data is given to a deep learning model, I need to clean and simplify the data such that there's enough data there to make a classification but not too much it's too much. 

