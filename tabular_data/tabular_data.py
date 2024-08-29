import itertools
import math
import os
import random
from distutils.util import strtobool
from os.path import exists
from turtle import title

import tensorflow as tf
from boto3 import client as boto3_client
from dotenv import load_dotenv
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers, models
import seaborn as sns
from utils.plot import *

from tabular_data.model import *


def plot_correlation_matrix(data, file_name="tabular_correlation_matrix.png"):
    """
    A function to calculate and plot
    correlation matrix of a DataFrame.
    """
    plt.rcParams.update({"font.size": 32})
    if "partial" in file_name:
        plt.rcParams.update({"font.size": 42})
        file_name = file_name.replace("partial", "partial_whole")
    # data = data.iloc[:, :5]
    # Create the matrix
    matrix = data.corr()
    # Create cmap
    cmap = sns.diverging_palette(
        250, 15, s=75, l=40, n=9, center="light", as_cmap=True)

    # Create a mask
    # mask = np.triu(np.ones_like(matrix, dtype=bool))

    # Make figsize bigger
    fig, ax = plt.subplots(figsize=(22, 22))

    # Plot the matrix
    _ = sns.heatmap(
        matrix,
        # mask=mask,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        cmap=cmap,
        ax=ax,
        cbar=False,
        # cbar_kws={"shrink": 0.70},
    )

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    _.figure.savefig(f"../plots/{file_name}")


def create_model(data):
    """Creates a model"""
    # NOTE: Create model
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(len(data.columns),)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # NOTE: Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def prepare_directories():
    """Creates necessary folders in preparation for data/models saved"""
    directories = ["plots", "models", "data", "optimal_parms"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if not os.path.isdir("plots/confusion_matrices"):
        os.mkdir("plots/confusion_matrices")

    # Prepare files
    if not exists('model_metrics.csv'):
        file_object = open('model_metrics.csv', 'w', encoding="utf-8")
        file_object.write("classifier,acc,auc_roc,log_loss,normalisation,balance_training,pca,svd\n")
        file_object.close()


def main():
    """Main"""
    print("[INFO] Tabular data classification")
    prepare_directories()

    print("[*]\tClient initialised")
    # tabular_data(CLIENT)
    data = pd.read_csv("../data/filtered_data.csv")
    # NOTE: Remove irrelevant columns
    del data['GROUP_ORIG1']
    del data['GROUP_ORIG2']
    # NOTE: Date/Time data is irrelevant
    del data['FSCAN_MONTHS']
    del data['DATE']
    del data['EXAMDATE']
    del data['PTDOB'] 
    # NOTE: Already a gender column
    del data['PTGENDER']
    del data['DATA_SOURCE']
    del data['SITEID']
    # NOTE: The following columns have the same value throughout
    del data['DXPARK']
    del data['DXNODEP']
    del data['VISCODE_Y']
    # NOTE: One outlier, just delete
    del data['DXOTHDEM']
    # NOTE: Unsure what these columns represent; likely irrelevant
    del data['SITEID_X']
    del data['SITEID_Y']

    print(f"[INFO] Data shape: {data.shape}")
    print(data.info())
    # NOTE: Discovered the following columns only have two unique values each; convert as such
    binary_cols = ["DXNORM", "DXMCI", "DXAD"]
    for col in binary_cols:
        # Convert first unique value to 1 and others to 0
        data[col] = data[col].apply(lambda x: 1 if x == 1 else 0)

    # NOTE: Collects all columns and casts values to integer datatypes
    int_cols = ["AGE", "MMSCORE"]
    int_cols.extend(binary_cols)
    for col in int_cols:
        # Convert to int
        data[col] = data[col].astype(int)

    print(data.describe())
    for col in data.columns:
        if col in ["PATIENT_ID", "MMSCORE", "AGE"]:
            continue
        # If col has more than 2 unique values, it is categorical
        if len(data[col].unique()) > 2:
            print("[INFO]\t{}".format(col))
            print(data[col].value_counts())

    # NOTE: TBD in GROUP column is an inconclusive diagnosis; remove
    data = data[data.GROUP != "TBD"]

    plot_correlation_matrix(data)

    # Train/Test split
    X = data.drop(["GROUP"], axis=1)
    y = data["GROUP"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=constants.TEST_SIZE, random_state=42)
    # Save data
    data.to_csv("../data/filtered_data_rep.csv", index=False)


if __name__ == "__main__":
    main()
