#! usr/bin/python3
""" Alzheimer's Classification Project

Author: Curious Coder

NOTE:   This and all dependent files have been coded under
        the assumption main.py is executed from the src folder
"""

import datetime
import os
import sys
import time

import numpy as np

from misc import cls, prepare_dir_structure


def print_title():
    """ Prints program title """
    # Read title.txt
    # If terminal size is small, print title
    if os.get_terminal_size().columns < 80:
        with open("../small_title.txt", encoding="utf-8") as file_object:
            print(file_object.read())
    else:
        with open("../title.txt", "r", encoding="utf-8") as file:
            title = file.read()
        # Print title
        print(title)


def print_time_left():
    """Print time left until September 1st, 2022"""
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    # Calculate time left
    time_left = datetime.datetime(year=2022, month=9, day=1, hour=0, minute=0, second=0) - \
        datetime.datetime(year=year, month=month, day=day,
                          hour=hour, minute=minute, second=second)
    # Convert time left into months, then weeks, then days, then hours, then minutes, then seconds
    months = time_left.days // 30
    weeks = (time_left.days % 30) // 7
    days = (time_left.days % 30) % 7
    # Print time left
    print(f"[TIME LEFT]\t{months} months, {weeks} weeks, {days} days")
    # Get today in string format
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    time_left = np.busday_count(today, '2022-09-01') * 3
    # Print working days
    print(
        f"[WORK HOURS]\tAssuming you're working at least 3 hours a day, you have at least {time_left} working hours left"
    )


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    """Main"""
    cls()
    # Prettiness for terminal
    print_title()
    print_time_left()

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
        from image_data.image_data import image_data_classification
        print(
            f"[INFO] Image data file imported in {time.time() - start:.2f} seconds")
        image_data_classification()
    elif arg in ["tabular", "2"]:
        from tabular_data.tabular_data import main as tmain
        tmain()
    elif arg in ["", "3", "exit"]:
        print("[EXIT]")
    # elif arg in["4", "test"]:

    else:
        print(f"[ERROR] Invalid argument '{arg}'")


if __name__ == "__main__":
    main()
