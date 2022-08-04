""" Prepare data for project"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
from skimage.transform import resize


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
    nii_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(filetype):
                nii_files.append(os.path.join(root, file))
    # Return nii files
    return nii_files


def get_slices(mri_scan):
    """Returns the center slices of the scan
    Args:
        scan (np.array): scan to be processed
    Returns:
        list: center slices of the scan
    """
    # Store mri_scan dimensions to individual variables
    n_i, n_j, n_k = mri_scan.shape

    # Calculate center frames for each scan
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2

    slice_0 = mri_scan[center_i, :, :]
    slice_1 = mri_scan[:, center_j, :]
    slice_2 = mri_scan[:, :, center_k]

    return [slice_0, slice_1, slice_2]


def prepare_images(image_size=(72, 72)):
    """ Main """
    print(f"[INFO] Preparing mri slices of size {image_size}")
    # if all_slices.npy exists, return
    if os.path.isfile(f"../data/dataset/all_slices_{image_size[0]}.npy"):
        print("[INFO]  Images were already prepared")
        return
    from image_data.image_data import filter_data, get_mri_data

    # Request folder name from user if AIBL folder doesn't exist
    if not os.path.exists("../AIBL"):
        print("AIBL Folder Doesn't exist")
        return
    nii_files = get_files("../AIBL", ".nii")
    # Separate nii files by slash depending on OS
    nii_files = [nii_file.replace("\\", "/") for nii_file in nii_files]

    # If dataset folder doesn't exist
    if not os.path.exists("../data/mri_images"):
        # Create dataset folder
        os.mkdir("../data/mri_images")

    # ! Check which patients we want from tabular data file
    tabular_data = pd.read_csv("../data/tabular_data.csv")

    # Filter data to only include patients we want
    tabular_data = filter_data(tabular_data, scan_num=1, project="AIBL")
    # Collect all patient ids from filtered data
    patient_ids = tabular_data['PATIENT_ID'].unique()
    # Create dictionary of patient_ids to false
    patient_dict = {patient_id: False for patient_id in patient_ids}

    # ! Copy mri scan files to dataset folder
    for count, nii_file in enumerate(nii_files):
        patient_id = nii_file.split("/")[-2]
        output_folder = f"../data/mri_images/{patient_id}"
        progress = count / len(nii_files) * 100
        print(f"[INFO]  {progress:.2f}%", end="\r")
        # If output_folder doesn't exist
        if not os.path.exists(output_folder):
            # If patient id is in our list of patients
            if patient_id in patient_ids:
                patient_dict[patient_id] = True
                # If folder doesn't exist
                if not os.path.exists(f"../data/mri_images/{patient_id}"):
                    # Create folder
                    os.mkdir(f"../data/mri_images/{patient_id}")
                # Copy nii file to dataset folder
                shutil.copy(nii_file, output_folder)

    # ! Check if all patients were found
    for patient_id, found in patient_dict.items():
        if not found:
            # Check if patient id is in our mri_images
            if os.path.exists(f"../data/mri_images/{patient_id}"):
                patient_dict[patient_id] = True

    # ! Extract center slices from nii files in dataset
    # for mri scan in dataset
    for count, patient_id in enumerate(patient_ids):
        progress = count / len(patient_ids) * 100
        print(f"[INFO] \t{progress:.2f}%", end="\r")
        # if center slice doesn't exist
        if not os.path.exists(
                f"../data/mri_images/{patient_id}/{patient_id}_center_slices_{image_size[0]}.npy"
        ):
            # If MRI scan exists for patient
            if patient_dict[patient_id]:
                # Get mri scan data
                mri_data = get_mri_data(patient_id, "../data/mri_images")
                if mri_data is None:
                    continue
                # Get center slices from nii files
                center_slices = get_slices(mri_data)
                # For every slice in center slices
                for index, slice in enumerate(center_slices):
                    # Normalise slice in center slices
                    center_slices[index] = normalise_data(slice)
                # Resize each slice
                im1, im2, im3 = resize_slices(center_slices, image_size)

                # Convert these image slices of scan to a concatenated np array
                processed_slices = np.array([im1, im2, im3]).T

                # Save center slices to npy file
                np.save(
                    f"../data/mri_images/{patient_id}/{patient_id}_center_slices_{image_size[0]}.npy",
                    processed_slices)
                print(f"[INFO]  Saved {patient_id}'s slices")

    # List all .npy files in dataset
    npy_files = get_files("../data/mri_images", ".npy")
    # Filter npy_files to image_size
    npy_files = [npy_file for npy_file in npy_files if npy_file.endswith(
        f"_{image_size[0]}.npy")]
    # Count number of .npy files
    count = 0
    for npy_file in npy_files:
        filename = npy_file.split("/")[-2].split(".")[0]
        if filename in patient_ids:
            count += 1
    print(
        f"[INFO] \t{count}/{len(patient_ids)} patients have center slices available for use")
    # Merge all .npy files into one array
    npy_files = [np.load(npy_file, allow_pickle=True)
                 for npy_file in npy_files]
    # Save npy file
    np.save(
        f"../data/dataset/all_slices_{image_size[0]}.npy", npy_files, allow_pickle=True)

    print("[INFO] \tFinished preparing images")


if __name__ == "__main__":
    print("[INFO] image_prepare.py cannot be run directly")
    sys.exit()
