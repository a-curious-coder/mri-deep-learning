""" Prepare data for project"""
import os
import sys
import time
import pandas as pd
import shutil


def filter_data(data, scan_num=None, project=None):
    """Filters full data-set to records we want

    Args:
        data (pd.DataFrame): full data-set
    """
    # If the filtered data is not already available, run the data filtering
    print(
        f"[*]\tRefining big data-frame to SCAN_NUM: {scan_num}, PROJECT: {project}"
    )
    if not os.path.exists("../data/filtered_data.csv"):
        # Filter data by scan number and study
        # if PROJECT is not none
        if project is not None:
            data = data[data['PROJECT'] == project]
        if scan_num is not None:
            data = data[data['SCAN_NUM'] == scan_num]

        # Remove rows/columns with null values
        null_val_per_col = data.isnull().sum().to_frame(
            name='counts').query('counts > 0')
        # NOTE: Null value quantity is same as max number of rows
        # NOTE: Thus, delete whole columns instead
        # Get column names
        columns = null_val_per_col.index.tolist()
        # Drop columns with null values
        data.drop(columns, axis=1, inplace=True)

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
        data.to_csv("../data/filtered_data.csv", index=False)
    else:
        print("[*]\tLoading filtered data-frame from file")
        data = pd.read_csv("../data/filtered_data.csv", low_memory=False)
    return data


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
        f"[INFO] Extracted {len(nii_files)} nii files in: {end - start:.2f}s")
    # Return nii files
    return nii_files


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


def main():
    """ Main """
    # Request folder name from user if AIBL folder doesn't exist
    if not os.path.exists("../AIBL"):
        print("AIBL Folder Doesn't exist")
        return
    nii_files = extract_nii_files("../AIBL")
    # Separate nii files by slash depending on OS
    if sys.platform == "win32":
        nii_files = [nii_file.replace("\\", "/") for nii_file in nii_files]
        # Get the folders each nii file is in
        nii_folders = [nii_file.split("/")[-2] for nii_file in nii_files]

    else:
        nii_files = [nii_file.replace("/", "\\") for nii_file in nii_files]
        # Get the folders each nii file is in
        nii_folders = [nii_file.split("\\")[-2] for nii_file in nii_files]

    # NOTE: The folders also represent the patient name
    # Save nii files and folders to text file
    with open("nii_files.txt", "w", encoding="utf-8") as file:
        for nii_folder in nii_folders:
            file.write(f"{nii_folder}\n")

    # If dataset folder doesn't exist
    if not os.path.exists("../dataset"):
        # Create dataset folder
        os.mkdir("../dataset")

    # ! Check which patients we want from tabular data file
    with open("../data/tabular_data.csv", "r", encoding="utf-8") as file:
        tabular_data = file.read()

    # Filter data
    tabular_data = filter_data(tabular_data, scan_num=1, project="AIBL")
    # Collect patient ids from filtered data
    patient_ids = tabular_data['PATIENT_ID'].unique()
    # Create folders for each patient
    for patient_id in patient_ids:
        # If folder doesn't exist
        if not os.path.exists(f"../dataset/{patient_id}"):
            # Create folder
            os.mkdir(f"../dataset/{patient_id}")

    # Copy nii files to dataset folder
    for nii_file in nii_files:
        # Get patient id from nii file
        patient_id = nii_file.split("/")[-2]
        # If patient id is in our list of patients
        if patient_id in patient_ids:
            # Copy nii file to dataset folder
            shutil.copy(nii_file, f"../dataset/{patient_id}")


if __name__ == "__main__":
    main()
    print("[INFO] Finished")
    input("[PRESS ENTER TO EXIT]")
    sys.exit()
