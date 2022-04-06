import os
from distutils.util import strtobool
from os.path import exists
import numpy as np
import boto3
import nibabel as nib
import pandas as pd
from dotenv import load_dotenv
import SimpleITK as sitk
from model import *
import tabular_data
from visualisations import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def filter_original_data(data):
    """Filters full data-set to records we want

    Args:
        data (pd.DataFrame): full data-set

    Returns:
        pd.DataFrame: filtered data-set
    """
    # Filter data to AIBL project
    data = data[data['PROJECT'] == "AIBL"]
    # Filter data to scan number 1
    data = data[data['SCAN_NUM'] == 1]
    return data


# Image data
def image_data(client):
    """Image data classification

    Args:
        client (botocore.client.S3): API client to access image data
    """

    # If the filtered data is not already available, run the data filtering
    if not exists("filtered_data.csv"):
        dprint("[*]\tRefining big data-frame to SCAN_NUM: 1, PROJECT: AIBL")
        # Load in all data
        data = pd.read_csv("adni_all_aibl_all_oasis_all_ixi_all.csv")
        # Filter data by scan number and study
        data = filter_original_data(data)
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

        data.to_csv("filtered_data.csv", index=False)

    # If we haven't made an image details file
    if not exists('image_details.csv'):
        dprint("[*]\tGenerating details associated with image")
        data = pd.read_csv("refined_data.csv")

        # Get all folder/patient names in current directory
        dirs = [
            item for item in os.listdir(MRI_IMAGE_DIR)
            if os.path.isdir(MRI_IMAGE_DIR + item)
        ]
        # Using patient names from directory's folder names, filter dataframe
        patients = data[data['PATIENT_ID'].isin(dirs)]
        # Get classification results for each patient in dataframe
        classifications = [label for label in patients['GROUP']]

        # Load in MRI Scans
        # NOTE: Multiple frames per mri scan (Typically 256)
        images = load_mri_scans(dirs)

        # Image shapes to evidence image is loaded
        image_shapes = [image.shape for image in images]

        # new dataframe for images to corresponding labels
        image_details = pd.DataFrame({
            "name": dirs,
            "image": image_shapes,
            "classification": classifications
        })

        # Save image details dataframe to file
        image_details.to_csv('image_details.csv', index=False)

    image_details = pd.read_csv('image_details.csv')
    #
    details = []
    images = []
    # For each diagnosis
    for classification in image_details['classification'].unique():
        temp = image_details[image_details['classification'] == classification]
        print(f"{classification}: {len(temp)}")
        details.append(temp.head(1))
        image = load_mri_scan(temp.head(1)['name'].values[0])
        slices = sitk.GetArrayFromImage(image)
        images.append(slices)

    plot_mri_comparison(images, details)
    return
    patient_details = image_details[image_details['name'] == patient_id]
    patient_diagnosis = patient_details['classification'].values[0]

    image = load_mri_scan(patient_id)
    # Load in image
    # print(image.GetSize())
    slices = sitk.GetArrayFromImage(image)
    print(slices.shape)
    # print(*slices[128], sep="\n")
    plot_mri_image(patient_id, patient_diagnosis, slices)
    # print(image_details)
    # cnn()


def cnn(image):
    """ Convolutional Neural Network classifying image """
    print("CNN")


def load_mri_scan(patient_id):
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
            # return nib.load(MRI_IMAGE_DIR + "/" + patient_id + "/" + file)


def load_mri_scans(patient_ids):
    """Loads in mri scans

    Parameters
    ----------
    dirs : list
        Directories

    Returns
    -------
    list
        MRI Scans
    """
    return [load_mri_scan(patient_id) for patient_id in patient_ids]


def image_data_eda(data):
    """Exploratory Data Analysis on dataframe

    Args:
        data (pd.DataFrame): mri data
    """
    if EDA:
        dprint(data.info())
        dprint(data.describe())
        data.describe().to_csv("dataset-description.csv", index=True)


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


###### Misc functions ######
def cls():
    """Clear terminal"""
    # clear the terminal before running
    os.system("cls" if os.name == "nt" else "clear")


def dprint(text):
    """Prints text during verbose mode

    Args:
        text (str): text
    """
    if VERBOSE:
        print(text)


def print_title(title):
    """Print title

    Args:
        title (str): title
    """
    dprint("------------------------------------------------"
           f"\n\t{title}\n"
           "------------------------------------------------")


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
    global client
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
    client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_access_key)


def prepare_directory():
    """Creates necessary folders in preparation for data/models saved"""
    directories = ["plots", "models", "data"]
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    if not os.path.isdir("plots/confusion_matrices"):
        os.mkdir("plots/confusion_matrices")
    if not os.path.isdir("models"):
        os.mkdir("models")
    if not os.path.isdir("optimal_parms"):
        os.mkdir("optimal_parms")
    # Prepare files
    if not exists('model_metrics.csv'):
        file_object = open('model_metrics.csv', 'w')
        file_object.write(
            "classifier,acc,auc_roc,log_loss,normalisation,balance_training,pca,svd\n"
        )
        # Close the file
        file_object.close()


def main():
    """Main"""
    cls()
    initialise_settings()
    print_title("Alzheimer's Classification Project")
    prepare_directory()

    print("[*]\tClient initialised")
    if TABULAR:
        tabular_data.main()

    if IMAGE:
        image_data(client)


if __name__ == "__main__":
    main()
