#! usr/bin/python3
""" Alzheimer's Classification Project

Author: Curious Coder

NOTE:   This and all dependent files have been coded under
        the assumption main.py is executed from the src folder
"""

import os
import sys
import time

from image_data.prepare_data import prepare_data
from image_data.train_test_models import main as image_data_classification
from utils.misc import cls, prepare_dir_structure, print_time_left, print_title
from tabular_data.tabular_data import main as tmain
import image_data.constants as constants
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def main():
    """Main"""
    # if misc exists, import it
    try:
        cls()
        # Prettiness for terminal
        print_title()
        print_time_left()
        # Print settings headers
        print(f"{'Image Size':<10} {str(constants.IMAGE_SIZE):<5}")
        print(f"{'Slice Mode':<10} {constants.SLICE_MODE:<5}")
        print(f"{'Test Size':<10} {str(constants.TEST_SIZE):<5}")
        print(f"{'Validation Size':<10} {str(constants.VAL_SIZE):<5}")

        print("\n")
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
        print("[4]\tTrain All Models")
        print("[5]\tPrepare data")
        print("\n")
        arg = input("[*]\tSelect an option: ")
        print("\n")

    # Preliminary setup
    prepare_dir_structure()

    # Lower case arg
    arg = arg.lower()

    if arg in ["image", "1"]:
        start = time.time()
        print(f"[INFO] Image data file imported in {time.time() - start:.2f} seconds")
        prepare_data()
        image_data_classification()
    elif arg in ["tabular", "2"]:
        tmain()
    elif arg in ["", "3", "exit"]:
        print("[EXIT]")
    elif arg in ["4"]:
        image_sizes = [(72, 72), (144, 144), (150, 150)]
        slice_modes = ["center", "average_center", "area"]
        test_sizes = [0.1, 0.2]

        for img_size in image_sizes:
            IMAGE_SIZE = img_size
            for slc_mode in slice_modes:
                SLICE_MODE = slc_mode
                for tst_size in test_sizes:
                    TEST_SIZE = tst_size
                    VAL_SIZE = 0.2
                print(f"[INFO] Images with size {img_size} and slice mode {slc_mode}")
                prepare_data()
                image_data_classification(IMAGE_SIZE, SLICE_MODE, ts = TEST_SIZE, vs = VAL_SIZE)
                
    elif arg in ["5"]:
        # Prepare data
        prepare_data()

    else:
        print(f"[ERROR] Invalid argument '{arg}'")


if __name__ == "__main__":
    main()
