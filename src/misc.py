""" Miscellaneous functions"""
import os
import datetime
import numpy as np


def cls():
    """Clear terminal"""
    # clear the terminal before running
    os.system("cls" if os.name == "nt" else "clear")

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



def prepare_dir_structure():
    """Creates necessary folders in preparation for data/models saved"""
    directories = [
        "../plots", "../plots/confusion_matrices", "../plots/F",
        "../plots/F/NL", "../plots/F/MCI", "../plots/F/AD", "../plots/M",
        "../plots/M/NL", "../plots/M/MCI", "../plots/M/AD", "../models",
        "../data", "../optimal_parms", "../data/results", "../data/history"
    ]

    for directory in directories:
        if not os.path.isdir(directory):
            os.mkdir(directory)

    # Prepare files
    if not os.path.exists('model_metrics.csv'):
        with open('model_metrics.csv', 'w') as file_object:
            file_object.write(
                "classifier,acc,auc_roc,log_loss,normalisation,balance_training,pca,svd\n"
            )


def progress_bar(current, total, bar_length=20):
    """ Prints progress bar

    Args:
        current (int): current progress
        total (int): total progress
        bar_length (int): length of progress bar
    """
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)
