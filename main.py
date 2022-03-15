import glob
import math
import os
import random
import shutil
from distutils.util import strtobool
from os.path import exists
from tabnanny import verbose

import boto3
import nibabel as nib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import tensorflow as tf
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from visualisations import *
from model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    avoid = ["Gender", "Research Group", "Age"]
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


def random_forest(x_train, x_test, y_train, y_test):
    # Initialise classifier
    rf = RandomForestClassifier(n_estimators=50)

    # Train random forest model
    rf.fit(x_train, y_train)

    # Predict labels
    pred = rf.predict(x_test)

    # Return actual and predicted labels
    return y_test, pred


def keras_network(data, test_size):
    x_train, x_test, y_train, y_test = train_test_split_with_labels(
        data, test_size=test_size)
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


def tabular_data(client):
    """Handles tabular MRI data

    Args:
        client (botocore.client.S3): API client to access tabular data
    """
    printt("Load Data")
    mri_data = get_tabular_data(client)
    printt("Exploratory Data Analysis")
    dprint(mri_data.shape)
    # TODO: Plotly Dashboard of sorts which will be saved as HTML
    print(f"There are {len(pd.unique(mri_data['SID']))} patients.")
    first_only = False
    if first_only:
        # Filters data to only include first row instance of each unique SID
        mri_data = mri_data.groupby('SID').first()

    printt("Preprocessing Data")
    df = mri_data.select_dtypes(exclude=['float64', 'int'])
    mri_data.drop(["Study", "SID", "total CNR"], axis=1, inplace=True)
    print("[*]\tStandardize / Normalize Data")
    normalized_mri_data = normalize_tabular_data(mri_data)

    # Encode labels and gender
    normalized_mri_data = normalized_mri_data.replace("AD", 1).replace("CN", 0)
    normalized_mri_data['Gender'] = normalized_mri_data['Gender'].replace(
        "F", 0).replace("M", 1)
    # normalized_mri_data = normalized_mri_data[normalized_mri_data['Gender'] == 1]
    # Drop columns filled with na/0 values
    normalized_mri_data = normalized_mri_data.dropna(axis=1, how='all')

    # Data/Label split
    X, y = split_data_train_test(normalized_mri_data, test_size=TEST_SIZE)

    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)
    # Dimensionality
    if PCA:
        pca = PCA(n_components=190, whiten='True')
        X = pca.fit(X).transform(X)
    elif SVD:
        svd = TruncatedSVD(n_components=190)
        X = svd.fit(X).transform(X)

    models = [RandomForestClassifier(), DecisionTreeClassifier(),
              SVC(kernel="rbf")]
    model_names = ["Random Forest", "Decision Tree", "Support Vector Machine"]

    # Models
    printt("Machine Learning Models")
    if machine_learning:
        for name, classifier in zip(model_names, models):
            model = Model(x_train, x_test, y_train, y_test,
                          classifier, name)

            model.train_predict()
            # Display results
            model.print_metrics()

            model.balance_dataset()
            # Plot Confusion matrix
            model.plot_cm()

    if deep_learning:
        printt("Deep Learning Models")
        keras_network(normalized_mri_data, TEST_SIZE)


def filter_original_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filters full data-set to records we want

    Args:
        data (pd.DataFrame): full data-set

    Returns:
        pd.DataFrame: filtered data-set
    """
    # Filter data to AIBL project
    data = data[data['PROJECT'] == "AIBL"]
    # Filter data to scan number 1
    data = data[data['SCAN_NUM'] == 1]
    return data


def image_data_eda(data: pd.DataFrame):
    """Exploratory Data Analysis on dataframe

    Args:
        data (pd.DataFrame): mri data
    """
    dprint(data.info())
    dprint(data.describe())
    data.describe().to_csv("dataset-description.csv", index=True)


def load_mri_scans(dirs: list) -> list:
    """Loads in mri scans

    Parameters
    ----------
    dirs : list
        Directories

    Returns
    -------
    list
        MRI Scans
    """
    images = []
    # Load each patient's image to list
    for folder in dirs:
        for file in os.listdir(mri_image_dir + folder):
            if file.endswith(".nii"):
                images.append(
                    nib.load(mri_image_dir + "/" + folder + "/" + file))
    return images


def image_data(client):
    """Image data classification

    Args:
        client (botocore.client.S3): API client to access image data
    """

    if not exists("refined_data.csv"):
        dprint("[*]\tRefining big data-frame to SCAN_NUM: 1, PROJECT: AIBL")
        # Load in original data
        data = pd.read_csv("adni_all_aibl_all_oasis_all_ixi_all.csv")
        # Filter data by scan number and study
        data = filter_original_data(data)
        # Perform eda
        image_data_eda(data)
        # Remove rows/columns with null values
        data = handle_null_values(data)

        # NOTE: EDA says max age is 1072, remove ages more than 125
        # Remove rows where age is less than 125
        data = data[data['AGE'] < 125]

        # Remove irrelevant columns to this data
        del data['PROJECT']
        del data['SCAN_NUM']

        # Extract patient ids from each directory path
        patients = [txt.split("\\")[-1] for txt in data['Path']]
        # Drop path column
        del data['Path']
        # Replace path values with patient ids
        data['PATIENT_ID'] = patients

        data.to_csv("refined_data.csv", index=False)

    # If we haven't made an image details file
    if not exists('image_details.csv'):
        dprint("[*]\tGenerating details associated with image")
        data = pd.read_csv("refined_data.csv")

        # Get all folder/patient names in current directory
        dirs = [item for item in os.listdir(
            mri_image_dir) if os.path.isdir(mri_image_dir + item)]
        # Using patient names from directory's folder names, filter dataframe
        patients = data[data['PATIENT_ID'].isin(dirs)]
        # Get classification results for each patient in dataframe
        classifications = [label for label in patients['GROUP']]

        # Load in MRI Scans
        # NOTE: Multiple frames per mri scan (Typically 256)
        images = load_mri_scans(dirs)

        # Image shapes to evidence image is loaded
        image_shapes = [image.shape for image in images]

        # new dataframe for images to corresponding labels
        image_details = pd.DataFrame(
            {"name": dirs, "image": image_shapes, "classification": classifications})

        # Save image details dataframe to file
        image_details.to_csv('image_details.csv', index=False)

    image_details = pd.read_csv('image_details.csv')
    print(image_details)


def handle_null_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handles null values from data

    Args:
        data (pd.DataFrame): original data

    Returns:
        pd.DataFrame: identifies and removes null/nan values
    """
    null_val_per_col = data.isnull().sum().to_frame(
        name='counts').query('counts > 0')
    print(null_val_per_col)
    # NOTE: Null value quantity is same as max number of rows
    # NOTE: Thus, delete whole columns instead
    # Get column names
    columns = null_val_per_col.index.tolist()
    # Drop columns with null values
    data.drop(columns, axis=1, inplace=True)
    return data


def prepare_directory():
    """Creates necessary folders in preparation for data/models saved"""
    if not os.path.isdir("plots"):
        os.mkdir("plots")
    if not os.path.isdir("plots/confusion_matrices"):
        os.mkdir("plots/confusion_matrices")
    if not os.path.isdir("models"):
        os.mkdir("models")


def cls():
    """Clear terminal"""
    # clear the terminal before running
    os.system("cls" if os.name == "nt" else "clear")


def dprint(text: str):
    """Prints text during verbose mode

    Args:
        text (str): text
    """
    if verbose:
        print(text)


def printt(title):
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
    global PCA
    global SVD

    global machine_learning
    global deep_learning
    global tabular
    global image
    global client
    global mri_image_dir
    global verbose

    # Loads access keys in from .env file
    load_dotenv()

    # Load in environment variables
    verbose = strtobool(os.getenv("VERBOSE"))
    mri_image_dir = os.getenv("MRI_IMAGES_DIRECTORY")

    TEST_SIZE = float(os.getenv("TEST_SIZE"))
    RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
    machine_learning = strtobool(os.getenv("MACHINE_LEARNING"))
    deep_learning = strtobool(os.getenv("DEEP_LEARNING"))

    # Dimensionality Reduction
    PCA = strtobool(os.getenv("PCA"))
    SVD = strtobool(os.getenv("SVD"))
    tabular = strtobool(os.getenv("TABULAR_DATA"))
    image = strtobool(os.getenv("IMAGE_DATA"))

    access_key = os.getenv("ACCESS_KEY")
    secret_access_key = os.getenv("SECRET_ACCESS_KEY")
    # Initialise AWS client to access Tabular Data
    client = boto3.client('s3',
                          aws_access_key_id=access_key,
                          aws_secret_access_key=secret_access_key)


def main():
    """Main"""
    cls()
    printt("Alzheimer's Classification Project")
    initialise_settings()
    prepare_directory()

    print("[*]\tClient initialised")
    if tabular:
        tabular_data(client)

    if image:
        image_data(client)


if __name__ == "__main__":
    main()
