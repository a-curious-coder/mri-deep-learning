""" Prepare data for project"""
import os
from re import T
import shutil

import numpy as np
import pandas as pd
from skimage.transform import resize
from image_data.image_data import filter_data, get_mri_data
from plot import *


def normalise_data(data):
    """ Normalises the data

    Args:
        data (list): List of data to normalise
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


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


def process_patient_slices(patient_id, slice_mode, image_size):
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
    patient_slices_dir = f"../data/mri_images/{patient_id}/{patient_id}_{slice_mode}_{image_size[0]}.npy"
    if mri_data is None:
        return False
    else:
        # ! 1 Extract slices from MRI scan data dependent on slice_mode
        if slice_mode == "center":
            scan_slices = get_center_slices(mri_data)
        elif slice_mode == "average_center":
            scan_slices = get_average_center_slices(mri_data)
        elif slice_mode == "area":
            scan_slices = get_area_slices(mri_data)

        # ! 2 Normalise pixel values in each scan slice
        for index, scan_slice in enumerate(scan_slices):
            scan_slices[index] = normalise_data(scan_slice)
        
        # ! 3 Resize each slice
        im1, im2, im3 = resize_slices(scan_slices, image_size)

        # ! 3.5 Rotate slices 180 degrees (Not necessary)
        im1 = np.rot90(im1, 2)
        im2 = np.rot90(im2, 2)
        im3 = np.rot90(im3, 2)

        # ! 4 Convert these image slices of scan to a concatenated np array
        processed_slices = np.array([im1, im2, im3]).T

        # ! 5 Save center slices to npy file
        np.save(patient_slices_dir, processed_slices)
        
    return True


def prepare_images(image_size=(72, 72), slice_mode = "center", verbose = 1):
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
    
    # if all_slices.npy exists, return
    if os.path.isfile(f"../data/dataset/all_{slice_mode}_slices_{image_size[0]}.npy"):
        print("[INFO] Images prepared")
        return
    print(f"[INFO] Preparing mri slices of size {image_size}")
    print(f"{'Size':<10} {'Mode':<10} {'# Images':<10}")

    patient_ids = []
    tabular_data = pd.read_csv("../data/tabular_data.csv", low_memory=False)
    # Filter data to only include patients we want
    tabular_data = filter_data(tabular_data, scan_num=1, project="AIBL")
    patient_ids = tabular_data['PATIENT_ID'].unique()
    print(f"{image_size[0]:<10} {slice_mode:<10} {len(patient_ids):<10}")

    # ! Preprocess organisation of MRI data from AIBL dataset
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

        # ! Copy mri scan files from AIBL to mri_images folder
        for count, mri_scan in enumerate(mri_scans):
            # Extract patient_id from scan path
            patient_id = mri_scan.split("/")[-2]
            output_folder = f"../data/mri_images/{patient_id}"
            # If output_folder doesn't exist
            if not os.path.exists(output_folder):
                # If patient id is in our list of patients
                if patient_id in patient_ids:
                    # Chaange patient_dict value to true because patient has an MRI scan
                    patient_dict[patient_id] += 1

                    # If the patient doesn't have their own folder in the new location
                    if not os.path.exists(f"../data/mri_images/{patient_id}"):
                        # Create folder
                        os.mkdir(f"../data/mri_images/{patient_id}")

                    # Copy mri file to patient's folder in mri_images
                    shutil.copy(mri_scan, output_folder)
            
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
        patient_slices_dir = f"../data/mri_images/{patient_id}/{patient_id}_{slice_mode}_{image_size[0]}.npy"
        # If the patient doesn't have their scan slices captured
        if not os.path.exists(patient_slices_dir) or overwrite:
            result = process_patient_slices(patient_id, slice_mode, image_size)
            if result:
                processed += 1
            else:
                missed += 1
        covered = f"{count} / {len(patient_ids)}"
        print(f"[INFO]  {processed:<10} {missed:<10} {covered:<10}", end = "\r")
    print(f"[INFO]  {processed:<10} {missed:<10} {len(patient_ids):<10}")
    # ! Merge all slices into one file
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

    print("[INFO] \tFinished preparing images")


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
