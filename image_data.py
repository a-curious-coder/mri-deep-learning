import os
import time
from distutils.util import strtobool

import boto3
import pandas as pd
import SimpleITK as sitk
from dotenv import load_dotenv

from misc_functions import *
from model import *
from plot import *


# PRE-PROCESSING
def handle_null_values(data):
    """Handles null values from data

    Args:
        data (pd.DataFrame): original data

    Returns:
        pd.DataFrame: identifies and removes null/nan values
    """
    null_val_per_col = data.isnull().sum().to_frame(
        name='counts').query('counts > 0')
    # print(null_val_per_col)
    # NOTE: Null value quantity is same as max number of rows
    # NOTE: Thus, delete whole columns instead
    # Get column names
    columns = null_val_per_col.index.tolist()
    # Drop columns with null values
    data.drop(columns, axis=1, inplace=True)
    return data


def filter_original_data(data):
    """Filters full data-set to records we want

    Args:
        data (pd.DataFrame): full data-set
    """
    # If the filtered data is not already available, run the data filtering
    if not exists("data/filtered_data.csv"):
        print("[*]\tRefining big data-frame to SCAN_NUM: 1, PROJECT: AIBL")
        # Filter data by scan number and study
        data = data[data['PROJECT'] == "AIBL"]
        data = data[data['SCAN_NUM'] == 1]
        # Perform eda
        image_data_eda(data)
        # Remove rows/columns with null values
        data = handle_null_values(data)

        # NOTE: EDA says max age is 1072, remove ages more than 125
        # Remove rows where age is less than 125
        data = data[data['AGE'] < 125]

        # Remove irrelevant columns to this data
        del data['PROJECT']
        del data['SCAN_NUM']

        # Extract patient ids from each directory path
        patients = [txt.split("\\")[-1] for txt in data['Path']]
        # Drop path column
        del data['Path']
        # Replace path values with patient ids
        data['PATIENT_ID'] = patients
        # Save filtered data
        data.to_csv("data/filtered_data.csv", index=False)
    else:
        print("[*]\tLoading filtered data-frame from file")
        data = pd.read_csv("data/filtered_data.csv", low_memory=False)
    return data
    

def tabularise_image_data(data):
    """ Assigns the corresponding labels from the tabular data to the MRI scan file names
    
    Args:
        data (pd.DataFrame): tabular mri data
    
    Returns:
        pd.DataFrame: mri data with labels
    """
    if not exists("data/tabular_image_data.csv"):
        # Get all folder (patient) names in current directory
        patient_ids = [
            patient_id for patient_id in os.listdir(MRI_IMAGE_DIR)
            if os.path.isdir(MRI_IMAGE_DIR + patient_id)
        ]
        # Load in MRI Scans
        # NOTE: Multiple frames per mri scan (Typically 256)
        images = get_mri_scans(patient_ids)

        # Image resolutions to evidence image is loaded
        image_shapes = [image.GetSize()[:2] for image in images]

        # Using patient names from directory's folder names, filter dataframe
        patients = data[data['PATIENT_ID'].isin(patient_ids)]

        # Get classification results for each patient in dataframe
        classifications = list(patients['GROUP'])
        genders = list(patients['GENDER'])

        data = pd.DataFrame({"NAME": patient_ids, "SHAPE": image_shapes, "GENDER": genders, "DIAGNOSIS": classifications})
        
        # Save image details dataframe to file
        data.to_csv('data/image_details.csv', index=False)
    else:
        # Load data in
        data = pd.read_csv("data/image_details.csv", low_memory=False)

    return data


def image_data_eda(data):
    """Exploratory Data Analysis on dataframe

    Args:
        data (pd.DataFrame): mri data
    """
    if EDA:
        print(data.info())
        print(data.describe())
        data.describe().to_csv("data/dataset-description.csv", index=True)


# PLOTS
def compare_mri_images(image_details):
    """ Compares images from different scans """
    details = []
    images = []

    print("Diagnosis\tFrequency")
    # For each unique diagnosis
    for diagnosis in image_details['diagnosis'].unique():
        temp = image_details[image_details['diagnosis'] == diagnosis]
        print(f"{diagnosis}\t{len(temp)}")
        # Get patient id
        patient_id = temp.head(1)['name'].values[0]
        # Append patient's details
        details.append(temp.head(1))
        # Load MRI scan from patient_id
        image = get_mri_scan(patient_id)
        # Rearrange 3D array
        slices = sitk.GetArrayFromImage(image)
        # plot_mri_image(patient_id, patient_diagnosis, slices)
        # Append rearranged image format to images
        images.append(slices)


def get_mri_scan(patient_id):
    """Loads in MRI scan

    Parameters
    ----------
    patient_id : str
        Patient ID

    Returns
    -------
    numpy.ndarray
        MRI scan
    """
    # ! Have assumed each patient folder has only one MRI scan file
    files = os.listdir(MRI_IMAGE_DIR + patient_id)
    if len(files) > 1:
        print(f"[!]\tMultiple MRI scans found for patient: {patient_id}")
        return

    for file in files:
        # If file is an MRI scan (With .nii extension)
        if file.endswith(".nii"):
            return sitk.ReadImage(MRI_IMAGE_DIR + patient_id + "/" + file)


def get_mri_scans(patient_ids: list):
    """Loads in mri scans

    Parameters
    ----------
    patient_ids : list
        Directories

    Returns
    -------
    list
        MRI Scans
    """
    return [get_mri_scan(patient_id) for patient_id in patient_ids]


def get_n_mri_scans(n):
    """ Loads n mri scans from the data directory 

    Parameters
    ----------  n : int
        Number of scans to load

    Returns
    -------
    list
        MRI scans
    """
    # Load image_details
    image_details = pd.read_csv('data/image_details.csv', low_memory = False)
    # image_details equals where diagnosis is not MCI
    image_details = image_details[image_details['diagnosis'] != 'MCI']

    # Get first n names from image_details
    patient_ids = image_details['name'].head(n).tolist()
    patient_scans = [get_mri_scan(patient_id) for patient_id in patient_ids]
    # Get patient id classification from image_details
    patient_diagnosis = image_details.loc[image_details['name'].isin(
        patient_ids), 'diagnosis']

    return zip(patient_ids, patient_diagnosis, patient_scans)


def get_best_mri_frame():
    """ Trains CNN to classify best frame of MRI scan for classification model """
    contents = get_n_mri_scans(1)
    # Need a training set for the CNN to recognise what a "best frame" looks like

def initialise_settings():
    """Loads environment variables to settings"""
    global TEST_SIZE
    global RANDOM_STATE
    global DR
    global PCA
    global SVD
    global BALANCE_TRAINING
    global NORMALISATION
    global ML
    global DL
    global TABULAR
    global IMAGE
    global EDA
    global CLIENT
    global MRI_IMAGE_DIR
    global VERBOSE
    global PREPROCESSING
    global TEST_ALL_CONFIGS

    # Loads access keys in from .env file
    load_dotenv()

    # Load in environment variables
    EDA = strtobool(os.getenv("EDA"))
    VERBOSE = strtobool(os.getenv("VERBOSE"))
    MRI_IMAGE_DIR = os.getenv("MRI_IMAGES_DIRECTORY")

    NORMALISATION = strtobool(os.getenv("NORMALISATION"))
    DR = strtobool(os.getenv("DIMENSIONALITY_REDUCTION"))
    # Dimensionality Reduction
    PCA = strtobool(os.getenv("PCA"))
    SVD = strtobool(os.getenv("SVD"))

    BALANCE_TRAINING = strtobool(os.getenv("BALANCE_TRAINING"))
    TEST_SIZE = float(os.getenv("TEST_SIZE"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

    ML = strtobool(os.getenv("MACHINE_LEARNING"))
    DL = strtobool(os.getenv("DEEP_LEARNING"))

    TABULAR = strtobool(os.getenv("TABULAR_DATA"))
    IMAGE = strtobool(os.getenv("IMAGE_DATA"))

    # All preprocessing settings as booleans
    PREPROCESSING = [NORMALISATION, BALANCE_TRAINING, DR]
    TEST_ALL_CONFIGS = strtobool(os.getenv("TEST_ALL_CONFIGS"))
    access_key = os.getenv("ACCESS_KEY")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY")
    # Initialise AWS client to access Tabular Data
    CLIENT = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_access_key)


def strip_skull_from_mri(image):
    """Strips skull from MRI scan

    Parameters
    ----------
    image : numpy.ndarray
    """
    pass


def select_best_scan_frames(image_details):
    """Selects best frames from MRI scans"""
    # Load in image_details
    image_details = pd.read_csv('data/image_details.csv', low_memory = False)
    # image_details equals where diagnosis is not MCI
    image_details = image_details[image_details['diagnosis'] != 'MCI']
    # dictionary of patient_id and best frames
    best_frames = {}
    # For each patient
    for patient_id in image_details['name']:
        # Load MRI scan from patient_id
        image = get_mri_scan(patient_id)
        # Rearrange 3D array
        slices = sitk.GetArrayFromImage(image)
        # Get best frames
        best_frames[patient_id] = get_best_frames(slices)

def get_best_frames(slices):
    """Selects best frames of MRI scan using Neural Network
    """
def get_mri_scan_frames(patient_id):
    """Loads in MRI scan

    Parameters
    ----------
    patient_id : str
        Patient ID

    Returns
    -------
    numpy.ndarray
        MRI scan
    """
    # ! Have assumed each patient folder has only one MRI scan file
    files = os.listdir(MRI_IMAGE_DIR + patient_id)
    if len(files) > 1:
        print(f"[!]\tMultiple MRI scans found for patient: {patient_id}")
        return

    for file in files:
        # If file is an MRI scan (With .nii extension)
        if file.endswith(".nii"):
            return sitk.ReadImage(MRI_IMAGE_DIR + patient_id + "/" + file)


def get_mri_scan_framess(patient_ids: list):
    """Loads in mri scans

    Parameters
    ----------
    patient_ids : list
        Directories

    Returns
    -------
    list
        MRI Scans
    """
    return [get_mri_scan_frames(patient_id) for patient_id in patient_ids]


def get_n_mri_scan_frames(n):
    """ Loads n mri scans from the data directory 

    Parameters
    ----------  n : int
        Number of scans to load

    Returns
    -------
    list
        MRI scans
    """
    # Load image_details
    image_details = pd.read_csv('data/image_details.csv', low_memory = False)
    # image_details equals where diagnosis is not MCI
    image_details = image_details[image_details['diagnosis'] != 'MCI']

    # Get first n names from image_details
    patient_ids = image_details['name'].head(n).tolist()
    patient_scans = [get_mri_scan_frames(patient_id) for patient_id in patient_ids]
    # Get patient id classification from image_details
    patient_diagnosis = image_details.loc[image_details['name'].isin(
        patient_ids), 'diagnosis']

    return zip(patient_ids, patient_diagnosis, patient_scans)

    
def train_cnn(data):
    """Trains CNN to classify best frame of MRI scan for classification model"""
    

def main():
    """Image data classification"""
    print("[IMAGE DATA CLASSIFICATION]")
    initialise_settings()

    # Load in mri data schema
    data = pd.read_csv("data/adni_all_aibl_all_oasis_all_ixi_all.csv", low_memory = False)
    
    # Filter data by study/scan
    data = filter_original_data(data)
    # Create a tabular representation of the classification for each image in the data
    data = tabularise_image_data(data)
    # Count quantity for each unique scan resolution in dataset
    data_shape = data.groupby("SHAPE").size()
    print(f"[!]\tNumber of scans per resolution: {data_shape}")
    # Train CNN on images in mri_kaggle_dataset folder
    train_cnn(data)

    # get_best_mri_frame()
    # ! Compare MRI images from each diagnosis
    # compare_mri_images(data)
    print(data.head())
    # Save the same slice of each patient's MRI scan to file
    for index, row in data.iterrows():
        progress_bar(index, data.shape[0])
        patient_id = row['NAME']
        patient_diagnosis = row['DIAGNOSIS']
        image = get_mri_scan(patient_id)
        # Load in image
        slices = sitk.GetArrayFromImage(image)
        # ! Figure out which slice is most appropriate per patient
        plotted = plot_mri_slice(patient_id, patient_diagnosis, slices[128], directory=f"plots/{row['GENDER']}")

if __name__ == "__main__":
    main()