""" Miscellaneous functions"""
import os


def cls():
    """Clear terminal"""
    # clear the terminal before running
    os.system("cls" if os.name == "nt" else "clear")


def print_title(title):
    """Print title

    Args:
        title (str): title
    """
    print("------------------------------------------------\n"
          f"{title:<10}\n"
          "------------------------------------------------")


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
