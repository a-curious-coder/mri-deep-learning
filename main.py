import visualisations as v

import boto3
import os
import math
import pandas as pd
import plotly
import random

from dotenv import load_dotenv
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import models
from tensorflow.keras import layers
from os.path import exists


def print_data_shape(data):
    print(f"This dataset has {data.shape[0]} rows and {data.shape[1]} columns")


def print_null_values(data):
    """ Prints the quantity of null/NaN values in columns containing null/NaN values

    Args:
        data (pd.DataFrame): Tabular MRI data
    """
    # Columns containing null/NaN values
    null_column_names = data.columns[data.isnull().any()].tolist()
    print(f"There are {len(null_column_names)} columns with null/NaN values")
    for column in null_column_names:
        print(f"{column}: {data[column].isna().sum()}")


def normalize_tabular_data(mri_data):
    """Normalizes the values of each column (Excluding columns unrelated to the MRI scan itself)

    Args:
        mri_data (pd.DataFrame): MRI Data

    Returns:
        pd.DataFrame: Normalized MRI data
    """
    avoid = ["Study", "SID", "total CNR", "Gender", "Research Group", "Age"]
    # apply normalization techniques
    for column in mri_data.columns:
        if column not in avoid:
            mri_data[column] = mri_data[column] / mri_data[column].abs().max()
    return mri_data


def clean_data(data):
    """Cleans MRI data, ready for deep learning model

    Args:
        data (pd.DataFrame): MRI data in tabular format
    """

    pass


def get_tabular_data(client):
    """Gets tabular MRI data from AWS

    Args:
        client (): AWS client

    Returns:
        pd.DataFrame: Tabular MRI data
    """
    response = client.get_object(Bucket='mri-deep-learning',
                                 Key='data/tabular/adni_ixi_rois_data_raw.csv')
    return pd.read_csv(response.get("Body"))


def linear_regression(X, y):
    """Linear Regression model trained given mri data; prints accuracy results based on test sets

    Args:
        X ([type]): MRI dataset
        y ([type]): MRI dataset labels
    """
    reg = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=4)
    reg.fit(x_train, y_train)
    print(f"[*]\t{reg.score(x_test, y_test)*100:.2f}%\tLinear Regression")


def linear_regression_pca(X, y):
    pca = PCA(n_components=190, whiten='True')
    pca_x = pca.fit(X).transform(X)
    reg = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(pca_x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=4)
    reg.fit(x_train, y_train)
    print(f"[*]\t{reg.score(x_test, y_test)*100:.2f}%\tLinear Regression PCA")


def linear_regression_svd(X, y):
    svd = TruncatedSVD(n_components=190)
    x = svd.fit(X).transform(X)
    reg = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=4)
    reg.fit(x_train, y_train)
    print(f"[*]\t{reg.score(x_test, y_test)*100:.2f}%\tLinear Regression SVD")


def random_forest_pca(X, y):
    pca = PCA(n_components=190, whiten='True')
    pca_x = pca.fit(X).transform(X)
    rf = RandomForestClassifier(n_estimators=50)
    x_train, x_test, y_train, y_test = train_test_split(pca_x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=4)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    labels = y_test.values
    count = 0
    for i in range(len(pred)):
        if pred[i] == labels[i]:
            count = count + 1
    print(f"[*]\t{count / float(len(pred))*100:.2f}%\tRandom Forest PCA")


def keras_network(data, test_size):
    # print(X.head())
    # x_train, x_test, y_train, y_test = train_test_split(X,
    #                                                     y,
    #                                                     test_size=test_size,
    #                                                     random_state=4)
    x_train, x_test, y_train, y_test = train_test_split_mri(data, test_size=test_size)
    # print(
    #     f"Train set size: {x_train.shape[0]}, test set size: {x_test.shape[0]}"
    # )
    epochs = 300
    size = math.floor((x_train.shape[0] / 10) * 9)
    rewrite_model = False
    if not exists(f"models/mri_model_{epochs}.tf") or rewrite_model:
        # Prepare network
        network = models.Sequential()
        network.add(layers.Dense(16, activation='relu', input_shape=(433, )))
        network.add(layers.Dense(16, activation='relu'))
        network.add(layers.Dense(1, activation='sigmoid'))

        network.compile(optimizer='rmsprop',
                        loss='binary_crossentropy',
                        metrics=['acc'])

        x_val = x_train[:size]
        partial_x_train = x_train[size:]
        y_val = y_train[:size]
        partial_y_train = y_train[size:]

        history = network.fit(partial_x_train,
                              partial_y_train,
                              epochs=epochs,
                              batch_size=512,
                              validation_data=(x_val, y_val))
        # Save mri network
        network.save(f"models/mri_model_{epochs}.tf")
        history_dict = history.history
        print(history_dict.keys())
        acc = history_dict['acc']
        val_acc = history_dict['val_acc']
        print(f"Accuracy: {history_dict['acc']}")
        print(f"Validation Accuracy: {history_dict['acc']}")
        v.plot_accuracy(acc[10:], val_acc[10:])
    else:
        network = models.load_model(f"models/mri_model_{epochs}.tf")
    # predictions = network.predict(x_test)

    predictions = network.evaluate(x_test, y_test, verbose = 0)
    print(f"[*]\t{predictions[1]*100:.2f}%\tKeras Neural Network")


def train_test_split_mri(data, test_size=0.2):
    """Splits the Tabular MRI data into train/test sets.
    Dataset contains multiple scans for some patients, we wouldn't want the same patient being included in both training and test sets.

    Args:
        x ([type]): [description]
        y ([type]): [description]
        test_size (float, optional): [description]. Defaults to 0.2.

    Returns:
        lists: Training and test splits
    """

    # TODO: Split dataset into a list of dataframes; one dataframe for each patient
    names = data['SID'].unique()
    # Shuffle names
    random.shuffle(names)
    # Split names array to train and test
    train_names, test_names = train_test_split(names, test_size = test_size)
    
    # Train
    x_train = data[data.SID.isin(train_names)]
    x_train = x_train.iloc[:, :x_train.shape[1] - 6]

    x_test = data[data.SID.isin(test_names)]
    x_test = x_test.iloc[:, :x_test.shape[1] - 6]

    y_train = data[data.SID.isin(train_names)]
    y_train = y_train['Research Group']

    y_test = data[data.SID.isin(test_names)]
    y_test = y_test['Research Group']
    # Test
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # TODO: Create training/test sets based on name order
    # Filter data to remove last 6 columns (irrelevant)
    # filtered_data = data.iloc[:, :data.shape[1] - 6]
    # # Checks if any values are NaN
    # nan_values = filtered_data.isna()
    # nan_columns = nan_values.any()
    # columns_with_nan = filtered_data.columns[nan_columns].tolist()
    
    # X = filtered_data.copy()
    # # Labels only
    # y = data['Research Group']
    y_train = y_train.replace("AD", 1)
    y_train = y_train.replace("CN", 0)
    y_test = y_test.replace("AD", 1)
    y_test = y_test.replace("CN", 0)
    # Split list of dataframes into training/test split and create training /test sets from them.
    return x_train, x_test, y_train, y_test


def prepare_directory():
    """Creates necessary folders in preparation for data/models saved
    """
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    if not os.path.isdir("models"):
        os.mkdir("models")


def main():
    # Prepares working directory to store deep learning models
    prepare_directory()
    # Loads access keys in from .env file
    load_dotenv()
    access_key = os.getenv("ACCESS_KEY")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY")
    test_size = float(os.getenv("TEST_SET_SIZE"))

    # Initialise AWS client to access Tabular Data
    client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_access_key)

    print("-------------------\nLoad Data\n-------------------")
    mri_data = get_tabular_data(client)
    print("-------------------\nExploratory Data Analysis\n-------------------")
    print_data_shape(mri_data)
    # TODO: Plotly Dashboard of sorts which will be saved as HTML
    print(f"There are {len(pd.unique(mri_data['SID']))} patients.")
    first_only = False
    if first_only == True:
        # Filters data to only include first row instance of each unique SID
        mri_data = mri_data.groupby('SID').first()
    print("-------------------\nPreprocessing Data\n-------------------")
    print("[*]\tStandardize / Normalize Data")
    normalized_mri_data = normalize_tabular_data(mri_data)
    # print_null_values(normalized_mri_data)
    # NaN values occur because there are a bunch of columns filled with zeros, thus can't be normalized
    # normalized_mri_data = normalized_mri_data.fillna(0)
    # Drop columns
    normalized_mri_data = normalized_mri_data.dropna(axis=1, how='all')

    train_test_split_mri(normalized_mri_data)

    # Models
    machine_learning = False
    deep_learning = True
    if machine_learning:
        print(
            "-------------------\nMachine Learning Models\n-------------------"
        )
        linear_regression(X, y)
        linear_regression_pca(X, y)
        linear_regression_svd(X, y)
        random_forest_pca(X, y)

    if deep_learning:
        print("-------------------\nDeep Learning Models\n-------------------")
        keras_network(normalized_mri_data, test_size)


if __name__ == "__main__":
    main()