import visualisations as v

import boto3
import os
import math
import pandas as pd
import plotly
import random
import tensorflow as tf

from dotenv import load_dotenv
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import models
from tensorflow.keras import layers
from os.path import exists
from scipy.stats import pearsonr


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
    x_train, x_test, y_train, y_test = train_test_split_with_labels(data, test_size=test_size)
    print("Accuracy\tNodes\tOptimizer\tLoss")
    settings = [['rmsprop', 'binary_crossentropy', 'accuracy'], ['adam', 'binary_crossentropy', 'accuracy']]
    layer_1_nodes = 32
    layer_2_nodes = 16
    for setting in settings:
        # print(type(x_train), type(x_test), type(y_train), type(y_test))
        epochs = 250
        # Validation set size
        val_size = math.floor((x_train.shape[0] / 10) * 9)
        rewrite_model = True
        # TODO: Loop over lists of settings and choose them based on evaulation accuracy on same data split
        if not exists(f"models/mri_model_{epochs}.tf") or rewrite_model:
            with tf.device('/cpu:0'):
                # Prepare network
                network = models.Sequential()
                network.add(layers.Dense(layer_1_nodes, activation='relu', input_shape=(x_train.shape[1], )))
                network.add(layers.Dense(layer_2_nodes, activation='relu'))
                network.add(layers.Dense(1, activation='sigmoid'))

                network.compile(optimizer=setting[0],
                                loss=setting[1],
                                metrics=[setting[2]])

                x_val = x_train[:val_size]
                partial_x_train = x_train[val_size:]
                y_val = y_train[:val_size]
                partial_y_train = y_train[val_size:]

                history = network.fit(partial_x_train,
                                    partial_y_train,
                                    epochs=epochs,
                                    batch_size=512,
                                    validation_data=(x_val, y_val),
                                    verbose = 0)
                # Save mri network
                network.save(f"models/mri_model_{epochs}.tf")
            history_dict = history.history
            # print(history_dict.keys())
            acc = history_dict['accuracy']
            val_acc = history_dict['val_accuracy']
            # print(f"Accuracy: {history_dict['acc']}")
            # print(f"Validation Accuracy: {history_dict['acc']}")
            v.plot_accuracy(acc[10:], val_acc[10:], f"mri_accuracy_{epochs}_{layer_1_nodes}_{layer_2_nodes}_{setting[0]}")
        else:
            network = models.load_model(f"models/mri_model_{epochs}.tf")
        predictions = network.evaluate(x_test, y_test, verbose = 0)
        print(f"[*]\t{predictions[1]*100:.2f}%\t({layer_1_nodes}, {layer_2_nodes})\t{setting[0]}\t{setting[1]}")
        # predictions = network.predict(x_test)


def split_data_train_test(data, test_size = 0.2):
    """Splits data into train and test dataframes

    Args:
        data (pd.DataFrame): tabular MRI data

    Returns:
        pd.DataFrame: Train and Test sets
    """
    names = data['SID'].unique()
    random.shuffle(names)
    size = math.floor(data.shape[0]*test_size)
    X = data.copy()
    y = data['Research Group'].replace("AD", 1).replace("CN", 0)
    return X, y


def train_test_split_with_labels(data, test_size=0.2):
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
    train_names, test_names = train_test_split(names, test_size=test_size)

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

    # Split list of dataframes into training/test split and create training /test sets from them.
    return x_train, x_test, y_train, y_test
    

def train_test_split_with_labels2(X, y, test_size=0.2):
    """Splits the Tabular MRI data into train/test sets.
    Dataset contains multiple scans for some patients, we wouldn't want the same patient being included in both training and test sets.

    Args:
        x (pd.DataFrame): data bar labels
        y (pd.DataFrame): labels
        test_size (float, optional): [description]. Defaults to 0.2.

    Returns:
        lists: Training and test splits
    """

    # TODO: Split dataset into a list of dataframes; one dataframe for each patient
    names = X['SID'].unique()
    # Shuffle names
    random.shuffle(names)
    # Split names array to train and test
    train_names, test_names = train_test_split(names, test_size=test_size)

    x_train = X[data.SID.isin(train_names)]
    x_train = x_train.iloc[:, :x_train.shape[1] - 6]

    x_test = X[X.SID.isin(test_names)]
    x_test = x_test.iloc[:, :x_test.shape[1] - 6]

    y_train = X[X.SID.isin(train_names)]
    y_train = y_train['Research Group']

    y_test = X[X.SID.isin(test_names)]
    y_test = y_test['Research Group']
    # Test
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Prepares working directory to store deep learning models
    prepare_directory()
    # Loads access keys in from .env file
    load_dotenv()
    # Load in environment variables
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
    normalized_mri_data = normalized_mri_data.replace("AD", 1).replace("CN", 0)
    # Drop columns filled with na/0 values
    normalized_mri_data = normalized_mri_data.dropna(axis=1, how='all')

    # Models
    machine_learning = False
    deep_learning = True
    if machine_learning:
        print(
            "-------------------\nMachine Learning Models\n-------------------"
        )
        X, y = split_data_train_test(normalised_mri_data, test_size = 0.2)
        linear_regression(X, y)
        linear_regression_pca(X, y)
        linear_regression_svd(X, y)
        random_forest_pca(X, y)

    if deep_learning:
        print("-------------------\nDeep Learning Models\n-------------------")
        keras_network(normalized_mri_data, test_size)

    print("done")


if __name__ == "__main__":
    main()