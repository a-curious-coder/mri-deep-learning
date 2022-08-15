""" Prepare data for project"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
from skimage.transform import resize
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


def get_center_slices(mri_scan):
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


def get_average_center_slices(mri_scan):
    """Returns the average of the middle 20 frames from scan
    Args:
        mri_scan (np.array): scan to be processed
    Returns:
        list: average center slices of the scan
    """
    # Store mri_scan dimensions to individual variables
    n_i, n_j, n_k = mri_scan.shape

    # Calculate center frames for each scan
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2

    # Get average of middle 20 frames
    slice_0 = np.mean(mri_scan[center_i - 10:center_i + 10, :, :], axis=0)
    
    # Roll axis so mean function operates correctly
    temp = np.rollaxis(mri_scan, 0, start=2)
    slice_1 = np.mean(temp[center_j - 10:center_j + 10, :, :], axis=0)

    temp = np.rollaxis(mri_scan, 2, start=0)
    slice_2 = np.mean(temp[center_k - 10:center_k + 10, :, :], axis=0)
    
    return [slice_0, slice_1, slice_2]


def get_area_slices(mri_scan):
    """ Get frame from scan with the least background
    Args:
        mri_scan (np.array): scan to be processed
    """
    return None


def prepare_images(image_size=(72, 72), slice_mode = "center"):
    """ Main """

    print(f"[INFO] Preparing mri slices of size {image_size}")
    # if all_slices.npy exists, return
    if os.path.isfile(f"../data/dataset/all_{slice_mode}_slices_{image_size[0]}.npy"):
        print("[INFO]  Images were already prepared")
        return
    from image_data.image_data import filter_data, get_mri_data

    patient_ids = []
    # ! Preprocess organisation of MRI data from AIBL dataset
    if os.path.exists("../AIBL"):
        # If dataset folder doesn't exist
        if not os.path.exists("../data/mri_images"):
            # Create dataset folder
            os.mkdir("../data/mri_images")

        # ! Filter patients we want from tabular data file
        tabular_data = pd.read_csv("../data/tabular_data.csv")

        # Filter data to only include patients we want
        tabular_data = filter_data(tabular_data, scan_num=1, project="AIBL")
        patient_ids = tabular_data['PATIENT_ID'].unique()
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
        for patient_id, scans in patient_dict.items():
            if scans > 1:
                print(f"[INFO]  Patient {patient_id} has {scans} scans")
            elif scans == 0:
                print(f"[INFO]  Patient {patient_id} has no scans")

    tabular_data = pd.read_csv("../data/tabular_data.csv", low_memory=False)
    # Filter data to only include patients we want
    tabular_data = filter_data(tabular_data, scan_num=1, project="AIBL")
    # Collect all patient ids from filtered data
    patient_ids = tabular_data['PATIENT_ID'].unique()
    
    # ! Extract slices from mri scans in dataset
    overwrite = True
    missed = 0
    for count, patient_id in enumerate(patient_ids):
        # If the patient doesn't have their scan slices captured
        if not os.path.exists(
                f"../data/mri_images/{patient_id}/{patient_id}_{slice_mode}_slices_{image_size[0]}.npy"
        ) or overwrite:

            # Get mri scan data
            mri_data = get_mri_data(patient_id, "../data/mri_images")
            if mri_data is None:
                missed += 1
                continue

            # ! Get slices from image
            if slice_mode == "center":
                slices = get_center_slices(mri_data)
            elif slice_mode == "average_center":
                slices = get_average_center_slices(mri_data)
            elif slice_mode == "area":
                slices = get_area_slices(mri_data)
            
            # For every slice in center slices
            for index, slice in enumerate(slices):
                # Normalise slice in center slices
                slices[index] = normalise_data(slice)
            
            # Resize each slice
            im1, im2, im3 = resize_slices(slices, image_size)

            # Convert these image slices of scan to a concatenated np array
            processed_slices = np.array([im1, im2, im3]).T
            # processed_slices = np.array([im1, im2, im3])

            # Save center slices to npy file
            np.save(
                f"../data/mri_images/{patient_id}/{patient_id}_{slice_mode}_slices_{image_size[0]}.npy",
                processed_slices)
        
        progress = count / len(patient_ids) * 100
        print(f"[INFO] \t{progress:.2f}%\tMissed: {missed}", end="\r")
    # ! Merge all slices into one file
    # List all .npy files in dataset
    npy_files = get_files("../data/mri_images", ".npy")
    # Filter npy_files to image_size
    npy_files = [npy_file for npy_file in npy_files if npy_file.endswith(
        f"_{image_size[0]}.npy")]
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
    print(
        f"[INFO] \t{count}/{len(patient_ids)} patients have center slices available for use")
    # Merge all .npy files into one array
    npy_files = [np.load(npy_file, allow_pickle=True)
                 for npy_file in npy_files]
    # Save npy file
    np.save(
        f"../data/dataset/all_{slice_mode}_slices_{image_size[0]}.npy", npy_files, allow_pickle=True)

    print("[INFO] \tFinished preparing images")


# def plot_mri_slices(image, label, patient_id=None):
#     """ Plots an image and its label

#     Args:
#         image (np array): image
#         label (str): label
#     """
#     print(f"[INFO] Plotting slices of mri scan for patient \'{patient_id}\'")
#     # Plot each channel separately
#     fig, axs = plt.subplots(1, 3, figsize=(15, 15))
#     axs[0].imshow(image[:, :, 0], cmap="gray")
#     axs[0].set_title("Axial")
#     axs[1].imshow(image[:, :, 1], cmap="gray")
#     axs[1].set_title("Coronal")
#     axs[2].imshow(image[:, :, 2], cmap="gray")
#     axs[2].set_title("Saggital")
#     # remove axis
#     for ax in axs:
#         ax.axis("off")
#     # Tight layout
#     fig.tight_layout()
#     # Sup title
#     fig.suptitle("Alzheimer's" if label == 1 else "Non-Alzheimer's")
#     fig.suptitle(label if label != 0 else "Non-Alzheimer's")
#     plt.show()


def main():
    """ Main function """
    print("[TEST] Get average center slices")
    image_size = (150, 150)
    slice_mode = "average_center"
    from image_data.image_data import filter_data, get_mri_data
    # Get mri scan data
    mri_data = get_mri_data("S231111", "../data/mri_images")
    if mri_data is None:
        print("[ERROR] No mri scan found")
        return
    # ! Get slices from image
    if slice_mode == "center":
        slices = get_center_slices(mri_data)
    elif slice_mode == "average_center":
        slices = get_average_center_slices(mri_data)
    elif slice_mode == "area":
        pass
        # slices = get_area_slices(mri_data)
    
    # For every slice in center slices
    for index, slice in enumerate(slices):
        # Normalise slice in center slices
        slices[index] = normalise_data(slice)
    
    # Resize each slice
    im1, im2, im3 = resize_slices(slices, image_size)

    # Convert these image slices of scan to a concatenated np array
    processed_slices = np.array([im1, im2, im3]).T

    # Save center slices to npy file
    np.save(
        f"../data/mri_images/S231111/S231111_{slice_mode}_slices_{image_size[0]}.npy",
        processed_slices)
    # Load mri scan from ../data/mri_images/S231111
    slices = np.load(f"../data/mri_images/S231111/S231111_{slice_mode}_slices_{image_size[0]}.npy", allow_pickle=True)
    # Plot slices
    plot_mri_slices(slices, "S231111")


if __name__ == "__main__":
    main()
