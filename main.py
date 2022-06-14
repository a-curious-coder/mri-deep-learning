#!/usr/bin/env python
import datetime
from misc_functions import cls, prepare_dir_structure, print_title
cls()
# Read title.txt
with open("title.txt", "r") as file:
    title = file.read()
# Print title
print(title)
# Print time left until September 1st, 2022
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
    time_left = datetime.datetime(year=2022, month=9, day=1, hour=0, minute=0, second=0) - datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
    # Convert time left into months, then weeks, then days, then hours, then minutes, then seconds
    months = time_left.days // 30
    weeks = (time_left.days % 30) // 7
    days = (time_left.days % 30) % 7
    # Print time left
    print(f"[TIME LEFT]\t{months} months, {weeks} weeks, {days} days")
    
print_time_left()

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    """Main"""
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

    if arg in ["tabular", "2"]:
        from tabular_data import main as tmain
        tmain()
    elif arg in ["image", "1"]:
        from image_data import main as imain
        imain()
    elif arg == "":
        print("[EXIT]")
    else:
        print(f"[ERROR] Invalid argument '{arg}'")


# if __name__ == "__main__":
main()
