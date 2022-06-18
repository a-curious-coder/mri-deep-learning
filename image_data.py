import os
import time
from distutils.util import strtobool
import boto3
import pandas as pd
import SimpleITK as sitk
from dotenv import load_dotenv
import skimage.segmentation as seg
from PIL import Image

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


def filter_data(data, SCAN_NUM = None, PROJECT = None):
    """Filters full data-set to records we want

    Args:
        data (pd.DataFrame): full data-set
    """
    # If the filtered data is not already available, run the data filtering
    print(f"[*]\tRefining big data-frame to SCAN_NUM: {SCAN_NUM}, PROJECT: {PROJECT}")
    if not exists("data/filtered_data.csv"):
        # Filter data by scan number and study
        # if PROJECT is not none
        if PROJECT is not None:
            data = data[data['PROJECT'] == PROJECT]
        if SCAN_NUM is not None:
            data = data[data['SCAN_NUM'] == SCAN_NUM]
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
    

def tabularise_image_data(data = None):
    """ Assigns the corresponding labels from the tabular data to the MRI scan file names
    
    Args:
        data (pd.DataFrame): tabular mri data
    
    Returns:
        pd.DataFrame: mri data with labels
    """
    if not exists("data/image_details.csv"):
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


import cv2
import matplotlib.pyplot as plt

def show_image(title, img, ctype):
    plt.figure (figsize =  (10 , 10 ) )
    if ctype == 'bgr':
        b , g , r = cv2.split ( img )
        # get b , g , r
        rgb_img = cv2.merge ( [ r , g , b ] )
        # switch it to rgb
        plt.imshow (rgb_img)
    elif ctype == "hsv":
        rgb = cv2.cvtColor ( img , cv2.COLOR_HSV2RGB )
        plt.imshow ( rgb )
    elif ctype == "gray":
        plt.imshow ( img , cmap = "gray" )
    elif ctype == "rgb" :
        plt.imshow ( img )
    else:
        raise Exception ( "Unknown colour" )
    plt.axis ( 'off' )
    plt.title ( title )
    plt.show ( )

def circle_points(resolution, center, radius):
    """Generate points which define a circle on an image.Centre refers to the centre of the circle"""
    radians = np.linspace ( 0 , 2*np.pi, resolution )
    c =  center [1] + radius*np.cos ( radians ) #polar co - ordinates
    r = center [0] + radius*np.sin ( radians )
    return np.array ( [ c , r ] ) . T


def image_show (image , nrows=1 , ncols=1 , cmap='gray' ) :
    """ Show the image presented in the parameters
    
    Args:
        image (np.array): image to be shown
        nrows (int): number of rows in the image
        ncols (int): number of columns in the image
        cmap (str): colour map
    
    Returns:
        None    
    """
    fig , ax = plt.subplots ( nrows - nrows , ncols - ncols , figsize=( 9 , 9 ) )
    ax.imshow ( image , cmap - ' gray ' )
    ax.axis ( 'off' )
    return fig , ax


def strip_skull_from_mri(image):
    """Strips skull from MRI scan

    Parameters
    ----------
    image : numpy.ndarray
    """
    image_path = "plots/F/AD/S232906.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert to float
    image = image.astype(float)
    # Rescale gray scale between 0-255
    image = (image - image.min()) / (image.max() - image.min())
    # Convert to uint8
    image = (image * 255).astype(np.uint8)
    # show_image("gray", image, "gray")
    # to binary using Otsu's method
    #Threshold the image
    ret , thresh_custom = cv2.threshold (image, 0,255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    #ShowImage ( " Applying Otsu ' , thresh_custom , ' gray " )
    points = circle_points ( 200 , [ 128 , 128 ] , 35 ) [ : -1 ]
    #fig , ax image_show ( gray )
    # ax.plot ( points [ :, 0 ] , points [ :, 1 ] , ' --r ' , Lw = 3 )
    snake =  seg.active_contour ( thresh_custom , points )
    fig , ax =  image_show ( thresh_custom )
    # ax.plot ( points [ :, 0 ] , points [ :, 1 ] , ' --r ' , Lw = 3 )
    ax.plot ( snake [ :, 0 ] , snake [ :, 1 ] , ' b ' , lw = 3 ) ;
    # OTSU THRESHOLDING
    _, binarized =  cv2.threshold (filtered , 0 , 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    print ( binarized.shape , ' is Otsu thresholding value ' )
    #ShowImage ( ' Brain with Skull XXX ' , binarized , ' gray ' )
    plt.imshow ( binarized )


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


def train_cnn(data):
    """Trains CNN to classify best frame of MRI scan for classification model
    
    Parameters
    ---------- 
    data : list
        List of tuples containing patient_id, diagnosis, and MRI scan
    """

    # Read in all MRI scans using patient IDs from data
    mri_scans = get_mri_scans(data['PATIENT_ID'])
    # Remove last 64 frames from each MRI scan
    mri_scans = [(patient_id, diagnosis, scan[:-64]) for patient_id, diagnosis, scan in mri_scans]
    # Remove first 64 frames from each MRI scan
    mri_scans = [(patient_id, diagnosis, scan[64:]) for patient_id, diagnosis, scan in mri_scans]
    # Detect which frame has largest surface area
    mri_scans = [(patient_id, diagnosis, scan[np.argmax(scan.sum(axis=1))]) for patient_id, diagnosis, scan in mri_scans]


def find_bounding_box(image):
    """Finds bounding box of image
    
    Parameters
    ---------- 
    image : numpy.ndarray
        Image to find bounding box of
    
    Returns
    -------
    numpy.ndarray
        Bounding box of image
    """
    # Find bounding box of image
    bounding_box = np.zeros(image.shape)
    bounding_box[image > 0] = 1
    # Find bounding box of image
    bounding_box = np.argwhere(bounding_box)
    # Find bounding box of image
    bounding_box = np.min(bounding_box, axis=0)
    # Find bounding box of image
    bounding_box = np.max(bounding_box, axis=0)
    # Return bounding box
    return bounding_box

def main():
    """Image data classification"""
    #! TEMP TEST FOR SKULL STRIP
    initialise_settings()
    # image = get_mri_scan('S63525')
    # slices = sitk.GetArrayFromImage(image)
    # # NOTE: FOR LOOP THIS
    # single_slice = slices[len(slices)//2]
    # im = Image.fromarray(single_slice)
    # im = im.convert("RGB")
    # im.save("your_file.jpg")
    # # plotted = plot_mri_slice("S63525", "AD", single_slice, directory="plots/F")

    # # Apply median filter
    # filtered = cv2.medianBlur(single_slice, 5)
    # # Calculate mean intensity value
    # mean_intensity = np.mean(filtered)
    # print(mean_intensity)
    # return
    try:
        print("[IMAGE DATA CLASSIFICATION]")
        
        # if filtered data exists
        if not os.path.exists("data/filtered_data.csv"):
            # Load in mri data schema
            data = pd.read_csv("data/adni_all_aibl_all_oasis_all_ixi_all.csv", low_memory = False)
            # Filter data by study/scan
            data = filter_data(data, SCAN_NUM = 1, PROJECT = "AIBL")
        # Create a tabular representation of the classification for each image in the data
        data = tabularise_image_data()
        # Count quantity for each unique scan resolution in dataset
        data_shape = data.groupby("SHAPE").size()
        print(f"[!]\tNumber of scans per resolution: {data_shape}")
        # Count number of rows in data
        print(f"[!]\tNumber of frames to process: {len(data)*256}")

        # Train CNN to identify best frame(s) of MRI scans
        # train_cnn(data)
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
            # strip_skull_from_mri(image)
            # Load in image
            slices = sitk.GetArrayFromImage(image)
            single_slice = slices[128]
            # ! Figure out which slice is most appropriate per patient
            im = Image.fromarray(single_slice)
            im = im.convert("RGB")
            
            im.save(f"plots/{row['GENDER']}/{patient_diagnosis}/{patient_id}.jpg")
            # plotted = plot_mri_slice(patient_id, patient_diagnosis, slices[128], directory=f"plots/{row['GENDER']}")
    except KeyboardInterrupt as keyboard_interrupt:
        print("[EXIT] User escaped")

if __name__ == "__main__":
    main()