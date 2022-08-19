#! usr/bin/python3
""" Alzheimer's Classification Project

Author: Curious Coder

NOTE:   This and all dependent files have been coded under
        the assumption main.py is executed from the src folder
"""

import os
import sys
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Settings
image_size = (72, 72)
slice_mode = "average_center" # center, average_center, area
mri_scans = "../data/mri_images"

def main():
    """Main"""

    # if misc exists, import it
    try:
        from misc import cls, prepare_dir_structure, print_title, print_time_left
        cls()
        # Prettiness for terminal
        print_title()
        print_time_left()
    except ImportError:
        print("[ERROR]\tCould not import misc.py")
    
    # Menu
    if sys.argv[1:]:
        # Read in arg
        arg = sys.argv[1]
    else:
        print("\n")
        print("[1]\tImage data")
        print("[2]\tTabular data")
        print("[3]\tExit")
        print("\n")
        arg = input("[*]\tSelect an option: ")
        print("\n")

    # Preliminary setup
    prepare_dir_structure()

    # Lower case arg
    arg = arg.lower()

    if arg in ["image", "1"]:
        start = time.time()
        # ! Prepare data
        from image_data.image_prepare import prepare_images
        prepare_images(image_size, slice_mode)
        from image_data.image_data import image_data_classification
        print(f"[INFO] Image data file imported in {time.time() - start:.2f} seconds")
        image_data_classification(image_size, slice_mode)
    elif arg in ["tabular", "2"]:
        from tabular_data.tabular_data import main as tmain
        tmain()
    elif arg in ["", "3", "exit"]:
        print("[EXIT]")
    elif arg in ["4", "testplot"]:
        from image_data.image_prepare import main as m
        patient_id = input("[*]\tEnter patient ID: ")
        # Does patient ID directory exists
        if os.path.exists(f"../data/mri_images/{patient_id}"):
            m(patient_id)
        else:
            print(f"[INFO] Patient ID {patient_id} directory does not exist. Running default.")
            m()
    elif arg in ["5"]:
        # Test all combinations of parameters
        from image_data.image_prepare import prepare_images
        from image_data.image_data import image_data_classification

        image_sizes = [(72, 72), (144, 144), (150, 150)]
        slice_modes = ["center", "average_center", "area"]

        for img_size in image_sizes:
            for slc_mode in slice_modes:
                print(f"[INFO] Images with size {img_size} and slice mode {slc_mode}")
                prepare_images(img_size, slc_mode)
                image_data_classification(img_size, slc_mode)


    else:
        print(f"[ERROR] Invalid argument '{arg}'")


if __name__ == "__main__":
    main()
