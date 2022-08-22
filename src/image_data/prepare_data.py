""" Prepare data for project"""
import os
import shutil

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from plot import *


def normalise_data(data):
    """ Normalises the data

    Args:
        data (list): List of data to normalise
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def preprocess_tab_data(data, scan_num=None, project=None):
    """Filters full data-set to records we want

    Args:
        data (pd.DataFrame): full data-set
    """
    # If the filtered data is not already available, run the data filtering
    print(f"[INFO]  Filtering tabular data to SCAN_NUM: {scan_num}, PROJECT: {project}")
    
    # Filter data by scan number and study
    if project is not None:
        data = data[data['PROJECT'] == project]
    if scan_num is not None:
        data = data[data['SCAN_NUM'] == scan_num]
    
    # Remove irrelevant columns in this data
    del data['PROJECT']
    del data['SCAN_NUM']

    # Remove rows/columns with null values
    null_val_per_col = data.isnull().sum().to_frame(name='counts').query('counts > 0')
    # Drop columns with null values
    columns = null_val_per_col.index.tolist()
    data.drop(columns, axis=1, inplace=True)

    # NOTE: EDA says max age is 1072, remove ages more than 125
    # Remove rows where age is less than 125
    data = data[data['AGE'] < 125]

    # Extract patient ids from each directory path
    patients = [txt.split("\\")[-1] for txt in data['Path']]
    del data['Path']
    # Replace path values with patient ids
    data['PATIENT_ID'] = patients
    
    # Move patient_id to the front of the dataframe
    data = data[['PATIENT_ID'] + data.columns.tolist()[1:-1]]
    # Sort data by patient id
    data.sort_values(by=['PATIENT_ID'], inplace=True)
    # Capitalise all column headers
    data.columns = [x.upper() for x in data.columns]
    # Save data to file
    data.to_csv("../data/filtered_data.csv", index=False)

    return data


def binarise_labels(labels):
    """ Prepares labels for
        Binary classification
    Args:
        labels (list): labels

    Returns:
        list: labels
    """
    # Replace labels in series with categorical labels
    labels = pd.Series(labels)
    # Replace AD and MCI with 1
    # labels[labels == "AD"] = 1
    labels[labels == "MCI"] = "AD"
    # Replace NL with 0
    # labels[labels == "NL"] = 0
    # Print counts
    print(labels.value_counts())
    return np.array(labels)


def resize_slices(slice_list, new_shape=(72, 72)):
    """ Resize the slices to the new shape

    Args:
        slice_list (list): list of slices
        new_shape (tuple): new shape of the image

    Returns:
        list: list of resized slices
    """
    # Resizes images for the purpose of being more visible in the plot
    im1 = resize(slice_list[0], new_shape, order=1, preserve_range=True)
    im2 = resize(slice_list[1], new_shape, order=1, preserve_range=True)
    im3 = resize(slice_list[2], new_shape, order=1, preserve_range=True)
    return [im1, im2, im3]


def get_mri_scan(patient_id, data_dir=None):
    """Loads in MRI scan

    Args:
        patient_id (str): patient id
        data_dir (str): directory to load data from
    Returns:
        np.array: MRI scan
    """

    mri_scan = None
    if data_dir is not None:
        MRI_IMAGE_DIR = data_dir
    # print(f"[INFO] Loading MRI scan for patient {patient_id} ")
    files = os.listdir(MRI_IMAGE_DIR + "/" + patient_id)

    # Collect all .nii files
    files = [file for file in files if file.endswith(".nii")]

    # If no mri scan file found
    if len(files) == 0:
        print(f"[!]\tNo MRI scan file found for patient: {patient_id}")

    # If a single mri scan file is found
    if len(files) == 1:
        for file in files:
            # If file is an MRI scan (With .nii extension)
            if file.endswith(".nii"):
                mri_scan = nib.load(MRI_IMAGE_DIR + "/" + patient_id + "/" +
                                    file)

    # If multiple mri scan files are found
    if len(files) > 1:
        print(f"[!]\tMultiple MRI scan files found for patient: {patient_id}")
        for file in files:
            print(file[:-10])

    return mri_scan


def get_mri_data(patient_id, data_dir=None, verbose=False):
    """ Loads in MRI scan data

    Args:
        patient_id (str): patient ID
    Returns:
        numpy.ndarray: MRI scan data
    """
    mri_scan = None
    try:
        # Load MRI scan
        mri_scan = get_mri_scan(patient_id, data_dir)
        # Get MRI scan data
        mri_scan = mri_scan.get_fdata()
    except Exception as err:
        if verbose:
            print(f"{patient_id:<10} {err}")
        mri_scan = None
        if verbose:
            print(f"[!]\tNo MRI scan file found for patient: {patient_id}")
        # ! If invalid_files.csv doesn't exist, create it
        if not os.path.isfile("../data/invalid_files.csv"):
            with open("../data/invalid_files.csv", "w",
                      encoding="utf-8") as file:
                file.write("patient_id\n")
            # Close file
            file.close()
        
        # ! If patient_id is not in invalid_files.csv, add it
        if patient_id not in open("../data/invalid_files.csv",
                                  "r",
                                  encoding="utf-8").read():
            with open("../data/invalid_files.csv", "a",
                      encoding="utf-8") as file:
                file.write(patient_id + "\n")
            # Close file
            file.close()
        return mri_scan
    
    # If mri scan is 4D, remove 4th dimension as it's useless
    if len(mri_scan.shape) == 4:
        mri_scan = mri_scan[:, :, :, 0]

    return mri_scan


def get_mri_scans(patient_ids: list):
    """Loads in multiple mri scans

    Args:
        patient_ids (list): list of patient IDs
    Returns:
        list: list of mri scans
    """
    return [
        get_mri_scan(patient_id, MRI_IMAGE_DIR) for patient_id in patient_ids
    ]


def get_mri_scans_data(patient_ids: list):
    """Loads in mri scans data

    Parameters
    ----------
    patient_ids : list
        Directories

    Returns
    -------
    list
        MRI Scans
    """
    return [
        get_mri_data(patient_id, MRI_IMAGE_DIR) for patient_id in patient_ids
    ]


def get_files(dir_path, filetype):
    """Extract nii files from directory

    Args:
        dir_path (str): Path to directory

    Returns:
        list: List of nii files
    """
    # Get all nii files contained in subdirectories of dir_path
    mri_scans = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(filetype):
                mri_scans.append(os.path.join(root, file))
    # Return nii files
    return mri_scans


def get_center_slices(mri_scan_data):
    """Returns the center slices of the scan
    Args:
        scan (np.array): scan to be processed
    Returns:
        list: center slices of the scan
    """
    # Store mri_scan_data dimensions to individual variables
    n_i, n_j, n_k = mri_scan_data.shape

    # Calculate center frames for each scan
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2

    slice_0 = mri_scan_data[center_i, :, :]
    slice_1 = mri_scan_data[:, center_j, :]
    slice_2 = mri_scan_data[:, :, center_k]

    return [slice_0, slice_1, slice_2]


def get_average_center_slices(mri_scan_data):
    """Returns the average of the middle 20 frames from scan
    Args:
        mri_scan_data (np.array): scan to be processed
    Returns:
        list: average center slices of the scan
    """
    # Store mri_scan_data dimensions to individual variables
    n_i, n_j, n_k = mri_scan_data.shape

    # Calculate center frames for each scan
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2

    # Get average of middle 20 frames
    slice_0 = np.mean(mri_scan_data[center_i - 10:center_i + 10, :, :], axis=0)
    
    # Roll axis so mean function operates correctly
    temp = np.rollaxis(mri_scan_data, 0, start=2)
    slice_1 = np.mean(temp[center_j - 10:center_j + 10, :, :], axis=0)

    temp = np.rollaxis(mri_scan_data, 2, start=0)
    slice_2 = np.mean(temp[center_k - 10:center_k + 10, :, :], axis=0)
    
    return [slice_0, slice_1, slice_2]


def get_area_slices(mri_scan_data):
    """ Extract slices from mri scan with the least background
    Args:
        mri_scan_data (np.array): 3D array of scan data 
    """
    # ! Rule 1 - Only consider middle 25% of mri_scan_data for each dimension
    middle = 0.33
    # ! Rule 2 - Apply threshold to slices to lesser features
    threshold = 0.15
    # ! Rule 3 - Select slices with the least background from frames

    # Normalise data
    temp = normalise_data(mri_scan_data)

    # ! 1 Select middle 25% of mri_scan_data for each dimension
    n_i, n_j, n_k = temp.shape
    # Calculate center frames for each dimension
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2

    # Frame range to cover
    frange = int(len(temp) * middle)

    # Set all values in temp to 0 if they are not in the middle for each dimension
    temp[:center_i - frange, :, :] = 0
    temp[center_i + frange:, :, :] = 0
    temp[:, :center_j - frange, :] = 0
    temp[:, center_j + frange:, :] = 0
    temp[:, :, :center_k - frange] = 0
    temp[:, :, center_k + frange:] = 0

    # ! 2 If pixel values are less than threshold of 0.3 then set to 0
    temp[temp < threshold] = 0

    # ! 3 Select 2d array from mri_scan_data with most non-zero values
    i_min = np.argmax(np.count_nonzero(temp, axis=(1, 2)))
    j_min = np.argmax(np.count_nonzero(temp, axis=(0, 2)))
    k_min = np.argmax(np.count_nonzero(temp, axis=(0, 1)))

    # Print frames selected for debugging
    # print(f"{'saggital':<5} {'coronal':<5} {'axial':<5}")
    # print(f"{i_min:<5} {j_min:<5} {k_min:<5}")

    # Select frames from original mri_scan_data using the indices found above
    slice_0 = mri_scan_data[i_min, :, :]
    slice_1 = mri_scan_data[:, j_min, :]
    slice_2 = mri_scan_data[:, :, k_min]

    # Return slices for preprocessing
    return [slice_0, slice_1, slice_2]


def delete_all_npy_files():
    """ Delete all npy files in the current directory"""
    # List all .npy files in dataset
    npy_files = get_files("../data/mri_images", ".npy")
    # Delete all .npy files
    for file in npy_files:
        os.remove(file)


def merge_all_data(slice_mode, image_size, patient_ids):
    """Merge all slices into one file
    Args:
        slice_mode (str): Mode of slice extraction
        image_size (int): Size of image to be extracted
        patient_ids (list): List of patient ids
    """
    # List all .npy files in dataset
    npy_files = get_files("../data/mri_images", ".npy")
    # Filter npy_files to image_size
    npy_files = [npy_file for npy_file in npy_files if npy_file.endswith(
        f"{slice_mode}_{image_size[0]}.npy")]
    print(f"{slice_mode}_{image_size[0]}.npy")
    # Filter npy_files if name contains slice_mode
    npy_files = [npy_file for npy_file in npy_files if slice_mode in npy_file]

    # Count number of .npy files
    count = 0
    for npy_file in npy_files:
        # Replace slash depending on OS
        npy_file = npy_file.replace("\\", "/")
        filename = npy_file.split("/")[-2]
        if filename in patient_ids:
            count += 1
    
    print(f"[INFO] \t{count}/{len(patient_ids)} patients have center slices available for use")
    # Merge all .npy files into one array
    npy_files = np.stack([np.load(npy_file, allow_pickle=True) for npy_file in npy_files])
    # Save npy file
    np.save(f"../data/dataset/all_{slice_mode}_slices_{image_size[0]}.npy", npy_files, allow_pickle=True)


def process_patient_slices(patient_id, slice_mode, image_size, patient_slices_dir):
    """ Processes the slices of a patient
    Args:
        patient_id (str): patient's id
        slice_mode (str): mode to extracting and processing the slices
        patient_slices_dir (str): path to save patient slices
    
    Returns:
        bool: True if successful, False otherwise
    """

    # Get mri scan data
    mri_data = get_mri_data(patient_id, "../data/mri_images")

    if mri_data is not None:
        # ! 1 Extract slices from MRI scan data dependent on slice_mode
        if slice_mode == "center":
            scan_slices = get_center_slices(mri_data)
        elif slice_mode == "average_center":
            scan_slices = get_average_center_slices(mri_data)
        elif slice_mode == "area":
            scan_slices = get_area_slices(mri_data)
        # Get smallest dimension of from all elements in scan_slices
        smallest_dim = min(min(scan_slices[0].shape), min(scan_slices[1].shape), min(scan_slices[2].shape))
        image_size = (smallest_dim, smallest_dim)
        # print(type())
        # ! 2 Normalise pixel values in each scan slice
        # for index, scan_slice in enumerate(scan_slices):
        #     scan_slices[index] = normalise_data(scan_slice)
        
        # ! 3 Resize each slice
        im1, im2, im3 = resize_slices(scan_slices, image_size)

        # ! 3.5 Rotate slices 180 degrees (Not necessary)
        im1 = np.rot90(im1, 2)
        im2 = np.rot90(im2, 2)
        im3 = np.rot90(im3, 2)

        # ! 4 Convert these image slices of scan to a concatenated np array
        processed_slices = np.array([im1, im2, im3]).T
        # ! 5 Save center slices to (npy) *png file
        # np.save(patient_slices_dir, processed_slices)
        cv2.imwrite(patient_slices_dir, processed_slices)

        return True
    
    # Failed to retrieve MRI data
    return False


def create_dataset_folders(labels, slice_mode, image_size):
    """ Create dataset folder for each label
    Args:
        labels (list): list of labels
        slice_mode (str): mode to extracting and processing the slices
    """
    if not os.path.exists(f"../data/dataset/{slice_mode}_{image_size[0]}"):
        os.mkdir(f"../data/dataset/{slice_mode}_{image_size[0]}")

    datasets = ['train', 'test']
    
    for dataset in datasets:
        if not os.path.exists(f"../data/dataset/{slice_mode}_{image_size[0]}/{dataset}"):
            os.mkdir(f"../data/dataset/{slice_mode}_{image_size[0]}/{dataset}")
        for label in labels:
            if not os.path.exists(f"../data/dataset/{slice_mode}_{image_size[0]}/{dataset}/{label}"):
                os.mkdir(f"../data/dataset/{slice_mode}_{image_size[0]}/{dataset}/{label}")


def link_tabular_to_image_data(data=None):
    """ Assigns the corresponding labels from the tabular data to the MRI scan file names
    Args:
        data (pd.DataFrame): tabular mri data
    Returns:
        pd.DataFrame: mri data with labels
    """

    # Collect all the MRI scan file names
    patient_ids = [
        patient_id for patient_id in os.listdir("../data/mri_images")
        if os.path.isdir("../data/mri_images/" + patient_id)
    ]

    # Using patient names from directory's folder names, filter dataframe
    patients = data[data['PATIENT_ID'].isin(patient_ids)]
    # Get classification results for each patient in dataframe
    classifications = list(patients['GROUP'])
    # Get genders of each patient
    genders = list(patients['GENDER'])

    # Bring data together into single dataframe
    data = pd.DataFrame({
        "PATIENT_ID": patient_ids,
        "GENDER": genders,
        "LABEL": classifications
    })

    # ! Remove invalid labels from data
    # Collect indices where DIAGNOSIS is TBD (aka "No diagnosis")
    to_be_classified = data[data["LABEL"] == "TBD"].index
    # Remove rows where DIAGNOSIS is TBD
    data = data.drop(to_be_classified)
    
    # Save dataframe to file
    data.to_csv('../data/image_to_label.csv', index=False)

    return data


def prepare_tabular(slice_mode, image_size):
    """ Prepares the tabular data for creating the dataset
    Args:
        slice_mode (str): mode to extracting and processing the slices
        image_size (tuple): size of the image slices
    Returns:
        pd.DataFrame: tabular data with labels
    """
    # if ../data/slice_mode doesn't exist, create it
    if not os.path.exists(f"../data/{slice_mode}"):
        os.makedirs(f"../data/{slice_mode}")

    # ! If clinical_data does not exist for these settings, create it
    if not os.path.isfile(f'../data/clinical_data.csv'):
        # ! Load clinical data associated with mri scans
        # Load in mri data schema
        clinical_data = pd.read_csv("../data/tabular_data.csv", low_memory=False)
        # NOTE: Filters data for the first scan of each patient for AIBL project
        clinical_data = preprocess_tab_data(clinical_data, scan_num=1, project="AIBL")
        # Create a tabular representation of the classification for each image in the data
        clinical_data = link_tabular_to_image_data(clinical_data)

        print(f"{'Size':<10} {'Mode':<10} {'# Images':<10}")
        patient_ids = clinical_data['PATIENT_ID'].unique()
        print(f"{image_size[0]:<10} {slice_mode:<10} {len(patient_ids):<10}")
        
        # ! Binarise labels (MCI & AD = AD)
        clinical_data['LABEL'] = np.asarray(binarise_labels(clinical_data['LABEL']).tolist())
        # Save clinical data to file
        clinical_data.to_csv(f'../data/clinical_data.csv', index=False)


def prepare_images(test_size, slice_mode, image_size, verbose=1):
    """ prepares images for the project
    
    Args:
        image_size (tuple): size of the image
        slice_mode (str): mode of slicing
        verbose (int): verbosity level
            1 - print progress
            2 - print progress and errors
        
    Returns:
        list: list of images
    """
    # ! Careful here
    delete = False
    if delete:
        delete_all_npy_files()
        print("[INFO] Deleted all npy files")

    
    clinical_data = pd.read_csv(f'../data/clinical_data.csv')
    # Get unique labels from tabular data
    labels = clinical_data["LABEL"].unique()
    create_dataset_folders(labels, slice_mode, image_size)
    # Get all patient ids from tabular data
    patient_ids = clinical_data["PATIENT_ID"]

    # Split patient_ids stratified by label into train and test sets
    train_patient_ids, _ = train_test_split(patient_ids, test_size=test_size, stratify=clinical_data["LABEL"])
    
    print(f"[INFO] Preparing mri slices of size {image_size}")

    # !  Re-organises MRI data away from AIBL folder structure
    if os.path.exists("../AIBL"):
        # If dataset folder doesn't exist
        if not os.path.exists("../data/mri_images"):
            # Create dataset folder
            os.mkdir("../data/mri_images")

        # Create dictionary of patient_ids with 0 scans
        patient_dict = {patient_id: 0 for patient_id in patient_ids}

        mri_scans = get_files("../AIBL", ".nii")
        # Generalises the directory string format across all OSs
        mri_scans = [mri_scan.replace("\\", "/") for mri_scan in mri_scans]

        # ! Move mri scan files from AIBL to mri_images folder
        for count, mri_scan in enumerate(mri_scans):
            # Extract patient_id from scan path
            patient_id = mri_scan.split("/")[-2]
            patient_folder = f"../data/mri_images/{patient_id}"
            # If patient's folder doesn't exist
            if not os.path.exists(patient_folder):
                # If patient id is in our list of patients
                if patient_id in patient_ids:
                    # Chaange patient_dict value to true because patient has an MRI scan
                    patient_dict[patient_id] += 1

                    # If the patient doesn't have their own folder in the new location
                    if not os.path.exists(f"../data/mri_images/{patient_id}"):
                        # Create folder
                        os.mkdir(f"../data/mri_images/{patient_id}")
                    # Move mri file to patient's folder in mri_images
                    shutil.move(mri_scan, patient_folder)
            
            progress = count / len(mri_scans) * 100
            print(f"[INFO]  {progress:.2f}%", end="\r")

        # ! Check if all patients MRI scans were found
        if verbose == 2:
            for patient_id, scans in patient_dict.items():
                if scans > 1:
                    print(f"[INFO]  Patient {patient_id} have {scans} scans")
                elif scans == 0:
                    print(f"[INFO]  Patient {patient_id} have no scans")
    
    missed = 0
    processed = 0
    overwrite = False
    # ! Extract slices from mri scans in dataset
    print(f"[INFO] {'Processed':<10} {'Missed':<10} {'# Images':<10}")
    for count, patient_id in enumerate(patient_ids):
        # Determine if slices of patient are in train or test set
        dataset = "train" if patient_id in train_patient_ids.values else "test"
        # Get patient's diagnosis label
        patient_class = clinical_data[clinical_data["PATIENT_ID"] == patient_id]["LABEL"].values[0]
        # Create patient's folder in dataset folder
        patient_slices_dir = f"../data/dataset/{slice_mode}_{image_size[0]}/{dataset}/{patient_class}/{patient_id}.png"
        # If the patient doesn't have their scan slices captured
        if not os.path.exists(patient_slices_dir) or overwrite:
            result = process_patient_slices(patient_id, slice_mode, image_size, patient_slices_dir)
            if result:
                processed += 1
            else:
                missed += 1

        covered = f"{count} / {len(patient_ids)}"
        print(f"[INFO]  {processed:<10} {missed:<10} {covered:<10}", end = "\r")
    print(f"[INFO]  {processed:<10} {missed:<10} {len(patient_ids):<10}")
    
    # merge_all_data(slice_mode, image_size)
    print("[INFO] \tFinished preparing images")


def prepare_data(test_size, slice_mode, image_size):
    """ Prepares data for use in model 
    
    Args:
        slice_mode (str): The mode of the slices to be used
        image_size (tuple): The size of the images to be used
    
    Returns:
        None
    """
    global MRI_IMAGE_DIR
    MRI_IMAGE_DIR = "../data/mri_images"
    prepare_tabular(slice_mode, image_size)
    prepare_images(test_size, slice_mode, image_size)


def main(patient_id = "S231111"):
    """ Main function """
    if patient_id == "":
        patient_id = "S231111"
    print("[TEST] Get average center slices")
    image_size = (150, 150)
    # ! Testing the creation of all slice modes
    # TODO: Create a front-end to allow user to select which slices were best as to 
    #      create a training dataset for a YOLO model for frame selection
    slice_modes = ["center", "average_center", "area"]
    for slice_mode in slice_modes:
        result = process_patient_slices(patient_id, slice_mode, image_size)
        if result:
            slices = np.load(f"../data/mri_images/{patient_id}/{patient_id}_{slice_mode}_{image_size[0]}.npy", allow_pickle=True)
            plot_mri_slices(slices, "testlabel", patient_id=patient_id, slice_mode=slice_mode)
        else:
            print(f"[INFO] Patient {patient_id} has no {slice_mode} slices")


if __name__ == "__main__":
    main()
