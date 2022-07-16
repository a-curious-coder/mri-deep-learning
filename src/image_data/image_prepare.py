""" Prepare data for project"""
import os
import shutil
import sys
import time

import numpy as np


def extract_nii_files(dir_path):
    """Extract nii files from directory

    Args:
        dir_path (str): Path to directory

    Returns:
        list: List of nii files
    """
    start = time.time()
    # Get all nii files contained in subdirectories of dir_path
    nii_files = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".nii"):
                nii_files.append(os.path.join(root, file))

    end = time.time()
    print(
        f"[INFO] \tExtracted {len(nii_files)} nii files in: {end - start:.2f}s")
    # Return nii files
    return nii_files


def prepare_images():
    """ Main """
    print("[INFO] Preparing images")
    from image_data.image_data import (filter_data, get_center_slices,
                                       get_mri_scan_data)

    # Request folder name from user if AIBL folder doesn't exist
    if not os.path.exists("../AIBL"):
        print("AIBL Folder Doesn't exist")
        return
    nii_files = extract_nii_files("../AIBL")
    # Separate nii files by slash depending on OS
    nii_files = [nii_file.replace("\\", "/") for nii_file in nii_files]

    # If dataset folder doesn't exist
    if not os.path.exists("../data/mri_images"):
        # Create dataset folder
        os.mkdir("../data/mri_images")

    # ! Check which patients we want from tabular data file
    with open("../data/tabular_data.csv", "r", encoding="utf-8") as file:
        tabular_data = file.read()

    # Filter data to only include patients we want
    tabular_data = filter_data(tabular_data, scan_num=1, project="AIBL")
    # Collect patient ids from filtered data
    patient_ids = tabular_data['PATIENT_ID'].unique()

    # ! Copy mri scan nii files to dataset folder
    for nii_file in nii_files:
        patient_id = nii_file.split("/")[-2]
        output_folder = f"../data/mri_images/{patient_id}"
        # If output_folder doesn't exist
        if not os.path.exists(output_folder + "/" + patient_id):
            # If folder doesn't exist
            if not os.path.exists(f"../data/mri_images/{patient_id}"):
                # Create folder
                os.mkdir(f"../data/mri_images/{patient_id}")
            # Get patient id from nii file based on OS
            # If patient id is in our list of patients
            if patient_id in patient_ids:
                # Copy nii file to dataset folder
                shutil.copy(nii_file, output_folder)

    # ! Compare patient_ids with patient ids in nii_files
    # If patient_ids is not in nii_files, remove patient_id from dataset
    count = 0
    patient_ids2 = [nii_file.split("/")[-2] for nii_file in nii_files]
    for patient_id in patient_ids:
        if patient_id in patient_ids2:
            count += 1

    print(f"[INFO] \t{count}/{len(patient_ids)} patients in dataset")

    # ! Extract center slices from nii files in dataset
    # for mri scan in dataset
    for count, patient_id in enumerate(patient_ids2):
        progress = count/len(patient_ids2)*100
        print(f"[INFO] \t{progress:.2f}%", end="\r")
        # if center slice doesn't exist
        if not os.path.exists(f"../data/mri_images/{patient_id}/{patient_id}_center_slices.npy"):
            # Get mri scan data
            mri_data = get_mri_scan_data(patient_id, "../data/mri_images")
            # Get center slices from nii files
            center_slices = get_center_slices(mri_data)
            # Save center slices to npy file
            np.save(
                f"../data/mri_images/{patient_id}/{patient_id}_center_slices.npy", center_slices)


if __name__ == "__main__":
    print("[INFO] image_prepare.py cannot be run directly")
    sys.exit()
