""" MRI Scan images will be preprocessed here """
import numpy as np


def generate_dataset(patient_ids):
    """ Generates dataset for CNN

    Args:
        patient_ids (list): patient IDs
    """
    batch = "all"
    # if ../data/dataset/all.npy doesn't exist
    if not os.path.exists(f"../data/dataset/{batch}.npy"):
        print("[INFO] Generating image dataset")
        # Create final data-set for classification
        # Split patient_ids into batches of 10
        batches = np.array_split(patient_ids, len(patient_ids)//10)
        for batch, patient_ids in enumerate(batches):
            print(f"[{batch}] Saving  mri scan data")
            if os.path.exists(f"../data/dataset/{batch}_batch_data.npy"):
                continue
            final = []
            all_mri_center_slices = []
            # print(f"[INFO] Getting all {len(patient_ids)} mri scans worth of data")
            mri_scans_data = get_mri_scans_data(patient_ids)

            # print("[INFO] Extracting center slices of each mri scan angle")
            for mri_scan in mri_scans_data:
                all_mri_center_slices.append(get_center_slices(mri_scan))

            # print("[INFO] Concatenating all center slices for each scan")
            for i, center_slices in enumerate(all_mri_center_slices):
                print(f"{i/len(all_mri_center_slices)*100}%", end="\r")
                # Resizing each center slice to 72/72
                # TODO: Determine an optimal image size
                # NOTE: Could it be plausible to suggest a size closest to native scan resolution is best?
                #   Maintain as much quality?
                im1, im2, im3 = resize_slices(center_slices, (72, 72))
                # Convert these image slices of scan to concatenated np array for CNN
                all_angles = np.array([im1, im2, im3]).T
                # print(type(all_angles))
                final.append(all_angles)
            # Save final data-set to file
            np.save(
                f"../data/dataset/{batch}_batch_data.npy", final, allow_pickle=True)

        npfiles = glob.glob("../data/dataset/*.npy")
        npfiles.sort()
        # Merge all .npy files into one file
        all_arrays = []
        for npfile in npfiles:
            if "all" in npfile:
                continue
            all_arrays.append(np.load(npfile, allow_pickle=True))
        # Flatten 2d array
        all_arrays = [item for sublist in all_arrays for item in sublist]
        # Print length of all_arrays
        print(f"[INFO] Length of all_arrays: {len(all_arrays)}")
        np.save("../data/dataset/all.npy", all_arrays)
# Unused


def main():
    """Main"""
    pass


if __name__ == "__main__":
    main()
    print("[INFO] Images preprocessed")
    exit(0)
