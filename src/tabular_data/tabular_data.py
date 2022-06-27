import itertools
import math
import os
import random
from distutils.util import strtobool
from os.path import exists

import tensorflow as tf
from boto3 import client as boto3_client
from dotenv import load_dotenv
from pandas import read_csv as pd_read_csv
from pandas import unique as pd_unique
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers, models

from plot import *
from src.tabular_data.model import *


def tabular_data(client):
    """Handles tabular MRI data

    Args:
        client (botocore.client.S3): API client to access tabular data
    """
    print_title("Load Data")
    mri_data = get_tabular_data(client)
    print_title("Exploratory Data Analysis")
    dprint(mri_data.shape)
    # TODO: Plotly Dashboard of sorts which will be saved as HTML
    print(f"There are {len(pd_unique(mri_data['SID']))} patients.")
    first_only = False
    if first_only:
        # Filters data to only include first row instance of each unique SID
        mri_data = mri_data.groupby('SID').first()

    print_title("Preprocessing Data")
    mri_data = mandatory_preprocessing(mri_data)

    print("[*]\tStandardize / Normalize Data")
    if NORMALISATION:
        mri_data = normalize_tabular_data(mri_data)

    # Drop columns filled with na/0 values
    mri_data = mri_data.dropna(axis=1, how='all')
    # mri_data = mri_data[mri_data['Gender'] == 1]
    if ML:
        if TEST_ALL_CONFIGS:
            all_configurations(mri_data)
        else:
            test_models(mri_data)

    if DL:
        print_title("Deep Learning Models")
        keras_network(mri_data, TEST_SIZE)


def get_tabular_data(client):
    """Gets tabular MRI data from AWS

    Args:
        client (): AWS client

    Returns:
        pd.DataFrame: Tabular MRI data
    """
    response = client.get_object(Bucket='mri-deep-learning',
                                 Key='data/tabular/adni_ixi_rois_data_raw.csv')
    return pd_read_csv(response.get("Body"))


def mandatory_preprocessing(data):
    # Drop study and sid strings
    data.drop(["Study", "SID"], axis=1, inplace=True)

    # Encode labels and gender
    data['Research Group'] = data['Research Group'].replace("AD", 1).replace(
        "CN", 0)
    data['Gender'] = data['Gender'].replace("F", 0).replace("M", 1)
    return data


def normalize_tabular_data(mri_data):
    """Normalizes the values of each column (Excluding columns unrelated to the MRI scan itself)

    Args:
        mri_data (pd.DataFrame): MRI Data

    Returns:
        pd.DataFrame: Normalized MRI data
    """
    avoid = ["Gender", "Research Group", "Age"]
    # apply normalization techniques
    for column in mri_data.columns:
        if column not in avoid:
            mri_data[column] = mri_data[column] / mri_data[column].abs().max()
    return mri_data


def test_model():
    pass


def test_models(mri_data, preprocessing_mod=None):
    """Tests machine learning classification model

    Args:
        mri_data (_type_): _description_
        preprocessing_mod (_type_, optional): _description_. Defaults to None.
    """

    if preprocessing_mod is not None:
        global PREPROCESSING
        PREPROCESSING = preprocessing_mod

    # Data/Label split
    X, y = split_data_train_test(mri_data, test_size=TEST_SIZE)

    # Dimensionality reduction
    if DR:
        if PCA:
            pca = PCA(n_components=190, whiten='True')
            X = pca.fit(X).transform(X)
        elif SVD:
            svd = TruncatedSVD(n_components=190)
            X = svd.fit(X).transform(X)

    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # all_configurations()
    models = [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        SVC(C=1, gamma=0.01, kernel='rbf'),
        GaussianNB(),
        MultinomialNB()
    ]

    model_names = [
        "Random Forest", "Decision Tree", "Support Vector Machine",
        "Naive Bayes", "MN Naive Bayes"
    ]

    # all model parameter grids
    model_parameter_grids = get_model_parameter_grids(model_names)

    # Models
    print_title("Machine Learning Models")
    for i, classifier in enumerate(models):
        name = model_names[i]
        parms = model_parameter_grids[i]
        model = Model(PREPROCESSING, x_train, x_test, y_train, y_test,
                      classifier, name)

        model.initialise_optimal_parameters(parms)
        # Train model and predict labels
        model.train_predict()

        # Save results
        model.save_metrics()

        if BALANCE_TRAINING:
            model.balance_dataset()

        # Plot Confusion matrix
        model.plot_cm()


def all_configurations(mri_data):
    """Iterates through all permutations of possible pre-processing settings for each model

    Args:
        mri_data (pd.DataFrame): Tabular MRI data
    """
    settings_perms = list(itertools.product([False, True], repeat=3))
    dprint(f"[*]\t{len(settings_perms)} tests")
    for settings in settings_perms:
        test_models(mri_data, settings)


def split_data_train_test(data, test_size=0.2):
    """Splits data into train and test dataframes

    Args:
        data (pd.DataFrame): tabular MRI data

    Returns:
        pd.DataFrame: Train and Test sets
    """
    X = data.drop(['Research Group'], axis=1)
    y = data['Research Group'].tolist()
    return X, y


def get_model_parameter_grids(model_names):
    grids = []
    # Random Forest parameter grid
    rf_param_grid = {
        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    # SVM parameter grid
    svm_param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    }

    # Decision Tree parameter grid
    dt_param_grid = {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    # Naive Bayes parameter grid
    nb_param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }

    # MN Naive Bayes parameter grid
    mn_nb_param_grid = {
        'alpha': [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    for model_name in model_names:
        print(model_name)
        if model_name == 'Random Forest':
            grids.append(rf_param_grid)
        elif model_name == 'Support Vector Machine':
            grids.append(svm_param_grid)
        elif model_name == 'Decision Tree':
            grids.append(dt_param_grid)
        elif model_name == 'Naive Bayes':
            grids.append(nb_param_grid)
        elif model_name == 'MN Naive Bayes':
            grids.append(mn_nb_param_grid)
    return grids


# Misc functions

def cls():
    """Clear terminal"""
    # clear the terminal before running
    os.system("cls" if os.name == "nt" else "clear")


def dprint(text: str):
    """Prints text during verbose mode

    Args:
        text (str): text
    """
    if VERBOSE:
        print(text)


def print_title(title):
    """Print title

    Args:
        title (str): title
    """
    dprint("------------------------------------------------"
           f"\n\t{title}\n"
           "------------------------------------------------")


def initialise_settings():
    """Loads environment variables to settings"""
    global TEST_SIZE
    global RANDOM_STATE
    global DR
    global PCA
    global SVD
    global BALANCE_TRAINING
    global NORMALISATION
    global ML
    global DL
    global CLIENT
    global MRI_IMAGE_DIR
    global VERBOSE
    global PREPROCESSING
    global TEST_ALL_CONFIGS
    # Loads access keys in from .env file
    load_dotenv()

    MRI_IMAGE_DIR = os.getenv("MRI_IMAGES_DIRECTORY")

    # Load in environment variables
    VERBOSE = strtobool(os.getenv("VERBOSE"))

    # Normalisation of data
    NORMALISATION = strtobool(os.getenv("NORMALISATION"))

    # Dimensionality Reduction
    DR = strtobool(os.getenv("DIMENSIONALITY_REDUCTION"))
    PCA = strtobool(os.getenv("PCA"))
    SVD = strtobool(os.getenv("SVD"))

    # Data Balancing
    BALANCE_TRAINING = strtobool(os.getenv("BALANCE_TRAINING"))

    # Classification Settings
    TEST_SIZE = float(os.getenv("TEST_SIZE"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

    # Classification methods
    ML = strtobool(os.getenv("MACHINE_LEARNING"))
    DL = strtobool(os.getenv("DEEP_LEARNING"))

    # All preprocessing settings as booleans
    PREPROCESSING = [NORMALISATION, BALANCE_TRAINING, DR]
    TEST_ALL_CONFIGS = strtobool(os.getenv("TEST_ALL_CONFIGS"))

    # Access to AWS for data
    access_key = os.getenv("ACCESS_KEY")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY")
    # Initialise AWS CLIENT to access Tabular Data
    CLIENT = boto3_client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_access_key)


def prepare_directory():
    """Creates necessary folders in preparation for data/models saved"""
    directories = ["plots", "models", "data"]
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    if not os.path.isdir("plots/confusion_matrices"):
        os.mkdir("plots/confusion_matrices")
    if not os.path.isdir("models"):
        os.mkdir("models")
    if not os.path.isdir("optimal_parms"):
        os.mkdir("optimal_parms")
    # Prepare files
    if not exists('model_metrics.csv'):
        file_object = open('model_metrics.csv', 'w')
        file_object.write(
            "classifier,acc,auc_roc,log_loss,normalisation,balance_training,pca,svd\n"
        )
        # Close the file
        file_object.close()

# Unused


def keras_network(data, test_size):
    x_train, x_test, y_train, y_test = train_test_split_with_labels(
        data, test_size=test_size)
    print(x_train.head())
    print(type(x_train))
    input()
    print("Accuracy\tNodes\tOptimizer\tLoss")
    settings = [['rmsprop', 'binary_crossentropy', 'accuracy'],
                ['adam', 'binary_crossentropy', 'accuracy']]
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
                network.add(
                    layers.Dense(layer_1_nodes,
                                 activation='relu',
                                 input_shape=(x_train.shape[1], )))
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
                                      verbose=0)
                # Save mri network
                network.save(f"models/mri_model_{epochs}.tf")
            history_dict = history.history
            # print(history_dict.keys())
            acc = history_dict['accuracy']
            val_acc = history_dict['val_accuracy']
            # print(f"Accuracy: {history_dict['acc']}")
            # print(f"Validation Accuracy: {history_dict['acc']}")
            plot_accuracy(
                acc[10:], val_acc[10:],
                f"mri_accuracy_{epochs}_{layer_1_nodes}_{layer_2_nodes}_{setting[0]}"
            )
        else:
            network = models.load_model(f"models/mri_model_{epochs}.tf")
        predictions = network.evaluate(x_test, y_test, verbose=0)
        print(
            f"[*]\t{predictions[1]*100:.2f}%\t({layer_1_nodes}, {layer_2_nodes})\t{setting[0]}\t{setting[1]}"
        )
        # predictions = network.predict(x_test)


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


def main():
    """Main"""
    cls()
    initialise_settings()
    print_title("Alzheimer's Classification Project (Tabular Data)")
    prepare_directory()

    print("[*]\tClient initialised")
    tabular_data(CLIENT)


if __name__ == "__main__":
    main()
