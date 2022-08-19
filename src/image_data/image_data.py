""" Process, transform, and load image data """
import csv
import hashlib
import os
import random
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.regularizers import l2 as l2_regularizer
from tensorflow.keras import datasets

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

from tensorflow.keras import callbacks

from plot import *
import sys

MRI_IMAGE_DIR = "../data/mri_images"
EDA = False

def filter_data(data, scan_num=None, project=None):
    """Filters full data-set to records we want

    Args:
        data (pd.DataFrame): full data-set
    """
    # If the filtered data is not already available, run the data filtering
    print(
        f"[INFO]  Filtering tabular data to SCAN_NUM: {scan_num}, PROJECT: {project}"
    )
    # Filter data by scan number and study
    # if PROJECT is not none
    if project is not None:
        data = data[data['PROJECT'] == project]
    if scan_num is not None:
        data = data[data['SCAN_NUM'] == scan_num]

    # Remove rows/columns with null values
    null_val_per_col = data.isnull().sum().to_frame(
        name='counts').query('counts > 0')
    # Drop columns with null values
    columns = null_val_per_col.index.tolist()
    data.drop(columns, axis=1, inplace=True)

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
    
    # Move patient_id to the front of the dataframe
    data = data[['PATIENT_ID'] + data.columns.tolist()[1:-1]]
    # Sort data by patient id
    data.sort_values(by=['PATIENT_ID'], inplace=True)

    return data


def link_tabular_to_image_data(data=None):
    """ Assigns the corresponding labels from the tabular data to the MRI scan file names
    Args:
        data (pd.DataFrame): tabular mri data
    Returns:
        pd.DataFrame: mri data with labels
    """

    # Collect all the MRI scan file names
    patient_ids = [
        patient_id for patient_id in os.listdir(MRI_IMAGE_DIR)
        if os.path.isdir(MRI_IMAGE_DIR + "/" + patient_id)
    ]

    info = []
    batchmode = False
    if batchmode:
        # Create batches of patient_ids
        batches = [
            patient_ids[i:i + 10] for i in range(0, len(patient_ids), 10)
        ]
        print(f"{len(patient_ids)} scans")
        for batch, patient_ids in enumerate(batches):
            print(f"[*]\tLoading MRI scans Batch {batch + 1}")
            # Load MRI scans to memory
            images = get_mri_scans(patient_ids)

            # Image resolutions to evidence image is loaded
            # image_shapes = [image.GetSize() for image in images]
            image_shapes = [image.shape for image in images]

            # Using patient names from directory's folder names, filter dataframe
            patients = data[data['PATIENT_ID'].isin(patient_ids)]

            # Get classification results for each patient in dataframe
            classifications = list(patients['GROUP'])
            # Get genders of each patient
            genders = list(patients['GENDER'])
            # Bring data together into single dataframe
            temp = pd.DataFrame({
                "NAME": patient_ids,
                "SHAPE": image_shapes,
                "GENDER": genders,
                "DIAGNOSIS": classifications
            })
            info.append(temp)
        final = pd.concat(info, axis=0, ignore_index=False)
    else:
        print(f"[*]\tLoading all MRI scans")
        # Load MRI scans to memory
        images = get_mri_scans(patient_ids)

        # Image resolutions to evidence image is loaded
        image_shapes = [image.shape for image in images]
        # Using patient names from directory's folder names, filter dataframe
        patients = data[data['PATIENT_ID'].isin(patient_ids)]
        # Get classification results for each patient in dataframe
        classifications = list(patients['GROUP'])
        # Get genders of each patient
        genders = list(patients['GENDER'])

        # Bring data together into single dataframe
        final = pd.DataFrame({
            "NAME": patient_ids,
            "SHAPE": image_shapes,
            "GENDER": genders,
            "DIAGNOSIS": classifications
        })
    # Save dataframe to file
    final.to_csv('../data/image_details.csv', index=False)

    return final


def image_data_eda(data):
    """Exploratory Data Analysis on dataframe

    Args:
        data (pd.DataFrame): mri data
    """
    if EDA:
        print(data.info())
        print(data.describe())
        data.describe().to_csv("../data/dataset-description.csv", index=True)


def reset_random_seeds(seed):
    """ Resets the random seed for tensorflow and numpy
    Args:
        seed (int): seed for random number generator
    
    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_mri_scan(patient_id, data_dir=None):
    """Loads in MRI scan

    Args:
        patient_id (str): patient id
        data_dir (str): directory to load data from
    Returns:
        np.array: MRI scan
    """

    mri_scan = None
    if data_dir is not None:
        MRI_IMAGE_DIR = data_dir
    # print(f"[INFO] Loading MRI scan for patient {patient_id} ")
    files = os.listdir(MRI_IMAGE_DIR + "/" + patient_id)

    # Collect all .nii files
    files = [file for file in files if file.endswith(".nii")]

    # If no mri scan file found
    if len(files) == 0:
        print(f"[!]\tNo MRI scan file found for patient: {patient_id}")

    # If a single mri scan file is found
    if len(files) == 1:
        for file in files:
            # If file is an MRI scan (With .nii extension)
            if file.endswith(".nii"):
                mri_scan = nib.load(MRI_IMAGE_DIR + "/" + patient_id + "/" +
                                    file)

    # If multiple mri scan files are found
    if len(files) > 1:
        print(f"[!]\tMultiple MRI scan files found for patient: {patient_id}")
        for file in files:
            print(file[:-10])

    return mri_scan


def get_mri_data(patient_id, data_dir=None, verbose=False):
    """ Loads in MRI scan data

    Args:
        patient_id (str): patient ID
    Returns:
        numpy.ndarray: MRI scan data
    """
    mri_scan = None
    try:
        # Load MRI scan
        mri_scan = get_mri_scan(patient_id, data_dir)
        # Get MRI scan data
        mri_scan = mri_scan.get_fdata()
    except Exception as err:
        if verbose:
            print(f"{patient_id:<10} {err}")
        mri_scan = None
        if verbose:
            print(f"[!]\tNo MRI scan file found for patient: {patient_id}")
        # ! If invalid_files.csv doesn't exist, create it
        if not os.path.isfile("../data/invalid_files.csv"):
            with open("../data/invalid_files.csv", "w",
                      encoding="utf-8") as file:
                file.write("patient_id\n")
            # Close file
            file.close()
        
        # ! If patient_id is not in invalid_files.csv, add it
        if patient_id not in open("../data/invalid_files.csv",
                                  "r",
                                  encoding="utf-8").read():
            with open("../data/invalid_files.csv", "a",
                      encoding="utf-8") as file:
                file.write(patient_id + "\n")
            # Close file
            file.close()
        return mri_scan
    
    # If mri scan is 4D, remove 4th dimension as it's useless
    if len(mri_scan.shape) == 4:
        mri_scan = mri_scan[:, :, :, 0]

    return mri_scan


def get_mri_scans(patient_ids: list):
    """Loads in multiple mri scans

    Args:
        patient_ids (list): list of patient IDs
    Returns:
        list: list of mri scans
    """
    return [
        get_mri_scan(patient_id, MRI_IMAGE_DIR) for patient_id in patient_ids
    ]


def get_mri_scans_data(patient_ids: list):
    """Loads in mri scans data

    Parameters
    ----------
    patient_ids : list
        Directories

    Returns
    -------
    list
        MRI Scans
    """
    return [
        get_mri_data(patient_id, MRI_IMAGE_DIR) for patient_id in patient_ids
    ]


def create_model():
    """Creates a CNN model
    Returns:
        model (keras.models.Sequential): CNN model
    """
    print('[INFO] Creating model')
    size = image_size[0]

    model = models.Sequential()
    model.add(layers.Conv2D(100, (3, 3),
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=l2_regularizer(l=0.01),
                kernel_constraint=unit_norm(),
                input_shape=(size, size, 3))
            )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(70, (3, 3), activation='relu', kernel_regularizer=l2_regularizer(l=0.01)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(50, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    metrics=["acc"])
    # model.compile(loss="binary_crossentropy",
    #                 optimizer=optimizers.RMSprop(learning_rate=1e-4),
    #                 metrics=["acc"])

    return model


def train_model(model, train_data, train_labels, val_data, val_labels, epochs, batch_size, guid):
    """ Trains a CNN model
    
    Args:
        model (keras.models.Sequential): CNN model
        train_data (list): Training data
        train_labels (list): Training labels
        val_data (list): Validation images
        val_labels (list): Validation labels
        epochs (int): Number of epochs to train for
        batch_size (int): Batch size
        guid (str): Unique identifier for the model
    
    Returns:
        history (keras.callbacks.History): Training history
    """

    earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                        mode ="min", patience = 5,
                                        restore_best_weights = True)

    print(f"[INFO] Training CNN model using {slice_mode} slices")

    history = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}

    for epoch in range(epochs):
        start = time.time()
        one_history = model.fit(train_data,
                                train_labels,
                                epochs=1,
                                batch_size=batch_size,
                                validation_data =(val_images, val_labels),
                                callbacks=[earlystopping],
                                verbose=0)

        history["acc"].append(one_history.history["acc"][0])
        history["val_acc"].append(one_history.history["val_acc"][0])
        history["loss"].append(one_history.history["loss"][0])
        history["val_loss"].append(one_history.history["val_loss"][0])

        # the exact output you're looking for:
        print(
            f"[INFO] Epoch {epoch+1}/{epochs}\t{time.time() - start:.2f} seconds",
            end="\r")
    print(
        f"[INFO] Epoch {epochs}/{epochs}\t{time.time() - start:.2f} seconds"
    )

    # Save training history to csv
    with open(f"../data/history/cnn_{guid}.csv", "w", encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["acc", "val_acc", "loss", "val_loss"])
        for i in range(epochs):
            writer.writerow([
                history["acc"][i], history["val_acc"][i],
                history["loss"][i], history["val_loss"][i]
            ])
    plot_history(history, guid)
    return model


def train_and_test(train_data, val_data, test_data, train_labels, val_labels, test_labels):
    """ Trains and tests a CNN on the data

    Args:
        train_data (list): training data
        test_data (list): testing data
        train_labels (list): training labels
        test_labels (list): testing labels

    Returns:
        None
    """

    val_size = 0.2
    test_size = round(len(test_data)/(len(train_data) + len(test_data)), 1)
    height = image_size[0]
    width = image_size[1]

    # Compute the mean and the variance of the training data for normalization.
    accs = []
    f1 = []
    precision = []
    recall = []

    # Ensure the same random is generated each time
    random.seed(42)
    seeds = random.sample(range(1, 20), 5)
    seeds.sort()
    retrain = False
    print(f"{'Acc':<6} {'Loss':<6} {'seed':<6}")
    avg_guid = []
    for seed in seeds:
        reset_random_seeds(seed)

        epochs = 20
        batch_size = 32
        # GUID, seed, epochs, batch_size, slice_mode, image_size, acc, loss
        guid = f"{seed}_{epochs}_{batch_size}_{slice_mode}_{image_size[0]}"
        # Create hash of guid 8 characters long
        guid = hashlib.md5(guid.encode()).hexdigest()[:8]
        # For avg results
        avg_guid.append(guid)
        # Load training_log.csv into pandas dataframe
        train_log = pd.read_csv("../models/training_log.csv")
        
        # Declare blank model
        model = None

        # Check if model exists in models folder
        # TODO: fix name
        if os.path.isfile(f"../models/{guid}.h5") and not retrain:
            model = models.load_model(f"../models/{guid}.h5")

        if model is None:
            # ! Create train/validation test split
            # ! 1 Create model
            model = create_model()
            # ! 2 Train model
            model = train_model(model, train_data, train_labels, val_data, val_labels, epochs, batch_size, guid)
            # Save trained model
            model.save(f"../models/cnn_{guid}.h5")

        # ! 3 Evaluate model 
        score = model.evaluate(test_data, test_labels, verbose=0)

        loss = f"{score[0]*100:.2f}"
        acc = f"{score[1]*100:.2f}"
        print(f'{acc:<6} {loss:<6} {seed:<6}')

        # Predict labels for test data
        test_predictions = model.predict(test_data)
        test_label = utils.to_categorical(test_labels, 2)

        true_label = np.argmax(test_label, axis=1)
        # predicted_label = np.argmax(test_predictions, axis=1)
        # Round test predictions to nearest integer
        predicted_label = np.round(test_predictions)
        
        class_report = classification_report(true_label,
                                             predicted_label,
                                             output_dict=True,
                                             zero_division=0)

        # if guid is not in guid column, write to training_log.csv
        if guid not in train_log["guid"].values:
            # Append guid to csv file with stats
            with open("../models/training_log.csv", "a", encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                # If guid exists in csv, skip writing to csv
                writer.writerow([guid,
                                    seed,
                                    epochs,
                                    batch_size,
                                    slice_mode,
                                    height,
                                    width,
                                    acc,
                                    class_report["macro avg"]["precision"],
                                    class_report["macro avg"]["recall"],
                                    class_report["macro avg"]["f1-score"],
                                    loss,
                                    test_size,
                                    val_size,
                                    False
                                ])

    # Get accs from training_log.csv
    df = pd.read_csv("../models/training_log.csv")
    # Filter by guid
    df = df[df["guid"].isin(avg_guid)]
    # Get average acc
    accs = df["acc"].tolist()
    precision = df["precision"].tolist()
    recall = df["recall"].tolist()
    f1 = df["f1"].tolist()
    
    # Calculate statistics
    avg_acc = f"{np.array(accs).mean():.2f}"
    avg_precision = f"{np.array(precision).mean():.2f}"
    avg_recall = f"{np.array(recall).mean():.2f}"
    avg_f1 = f"{np.array(f1).mean():.2f}"

    std_acc = f"{np.array(accs).std():.2f}"
    std_precision = f"{np.array(precision).std():.2f}"
    std_recall = f"{np.array(recall).std():.2f}"
    std_f1 = f"{np.array(f1).std():.2f}"
    
    # Create hash of avg guid 8 characters long
    avg_guid = hashlib.md5(avg_guid[0].encode()).hexdigest()[:8]

    with open("../models/training_log.csv", "a", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([avg_guid,
                            seed,
                            epochs,
                            batch_size,
                            slice_mode,
                            height,
                            width,
                            avg_acc,
                            avg_precision,
                            avg_recall,
                            avg_f1,
                            "NULL",
                            test_size,
                            val_size,
                            True
                        ])
    
    # Print statistics
    print(f"{'Type':<10} {'Metric':<10} {'Standard Deviation':<10}")
    print(f"{'Average':<10}{'Accuracy':<10} {avg_acc:<10}")
    print(f"{'Average':<10}{'Precision':<10} {avg_precision:<10}")
    print(f"{'Average':<10}{'Recall':<10} {avg_recall:<10}")
    print(f"{'Average':<10}{'F1':<10} {avg_f1:<10}")
    print(f"{'STD':<10} {'Accuracy':<10} {std_acc:<10}")
    print(f"{'STD':<10} {'Precision':<10} {std_precision:<10}")
    print(f"{'STD':<10} {'Recall':<10} {std_recall:<10}")
    print(f"{'STD':<10} {'F1':<10} {std_f1:<10}")


def train_and_test_pretrained(train_data, test_data, train_labels,
                              test_labels, slice_mode="center"):
    """
    Train and test the pretrained ResNET50

    Args:
        train_data (list): training images
        test_data (list): testing images
        train_labels (list): training labels
        test_labels (list): testing labels
    """
    print("[INFO]  Training and testing pretrained ResNet50")
    # Load pretrained model
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    # Ensure the pre-trained model itself isn't retrained
    conv_base.summary()
    conv_base.trainable = True
    for layer in conv_base.layers:
        if 'block5_conv' in layer.name:
            layer.trainable = True
            continue
        layer.trainable = False

    # Incorporate pretrained model
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.001),
                  metrics=['acc'])
    history = model.fit(train_data,
                        train_labels,
                        epochs=15,
                        batch_size=32,
                        validation_split=0.1,
                        verbose=1)
    plot_history(history, "pretrained")
    score = model.evaluate(test_data, test_labels, verbose=0)
    print(f'{score[1]:.4f} {score[0]:.4f} ')

    # Append results to results file
    with open("../data/results/results.csv", "a", encoding="utf-8") as file:
        file.write(f"pretrained,{score[1]},{score[0]},{slice_mode},{image_size}\n")

    model.save('../models/pretrained_vgg16.h5')


def train_and_test_lstm_model():
    """
    Train and test the LSTM model
    """
    print("[INFO]  Training and testing LSTM model")
    # Load data
    train_data, test_data, train_labels, test_labels = load_data()
    # Create LSTM model and train
    pass


def process_labels(labels):
    """ Prepares labels for
        Binary classification
    Args:
        labels (list): labels

    Returns:
        list: labels
    """
    # Replace labels in series with categorical labels
    labels = pd.Series(labels)
    # Replace AD and MCI with 1
    labels[labels == "AD"] = 1
    labels[labels == "MCI"] = 1
    # Replace NL with 0
    labels[labels == "NL"] = 0
    # Print counts
    print(labels.value_counts())
    return np.array(labels)


def prepare_files():
    # If results.csv doesn't exist
    if not os.path.exists("../data/results/results.csv"):
        # Create results.csv
        with open("../data/results/results.csv", "w",
                  encoding="utf-8") as file:
            file.write("model,acc,loss,slice_mode,image_size\n")
            
    if not os.path.exists("../models/training_log.csv"):
        with open("../models/training_log.csv", "w",
                  encoding="utf-8") as file:
            file.write("guid,seed,epochs,batch_size,slice_mode,height,width,acc,loss,test_size,val_size,individual\n")


def image_data_classification(im, sl):
    """Image data classification"""
    global MRI_IMAGE_DIR
    global image_size
    global slice_mode
    MRI_IMAGE_DIR = "../data/mri_images"
    image_size = im
    slice_mode = sl
    prepare_files()
    print("")
    print(f"[INFO] Image data classification")
    # if ../data/slice_mode doesn't exist, create it
    if not os.path.exists(f"../data/{slice_mode}"):
        os.makedirs(f"../data/{slice_mode}")

    # ! If clinical_data does not exist for these settings, create it
    if not os.path.isfile(f'../data/{slice_mode}/clinical_data_{image_size[0]}.csv'):
        # ! Load clinical data associated with mri scans
        # Load in mri data schema
        clinical_data = pd.read_csv("../data/tabular_data.csv",
                                    low_memory=False)
        # NOTE: Filters data for the first scan of each patient for AIBL project
        clinical_data = filter_data(clinical_data, scan_num=1, project="AIBL")
        # Save data to file
        clinical_data.to_csv("../data/filtered_data.csv", index=False)
        # Create a tabular representation of the classification for each image in the data
        clinical_data = link_tabular_to_image_data(clinical_data)
        # If any mri scans are corrupted, remove them from the tabular data
        if os.path.exists("../data/invalid_files.csv"):
            # Load invalid_files.csv
            invalid_files = pd.read_csv("../data/invalid_files.csv")
            # Remove rows in data where patient_id is in invalid_files
            clinical_data = clinical_data[~clinical_data["NAME"].isin(invalid_files["patient_id"])]

        # ! Extract labels from clinical data
        # Collect indices where DIAGNOSIS is TBD (aka "No diagnosis")
        to_be_classified = clinical_data[clinical_data["DIAGNOSIS"] == "TBD"].index
        # Remove rows where DIAGNOSIS is TBD
        clinical_data = clinical_data.drop(to_be_classified)
        # Get labels
        labels = clinical_data['DIAGNOSIS']
        # Prepares labels for binary classification according to mode 1
        labels = np.asarray(process_labels(labels).tolist())

        # ! Create a training and test set of images corresponding to clinical data
        file_name = f"../data/dataset/all_{slice_mode}_slices_{image_size[0]}.npy"
        images = np.load(file_name, allow_pickle=True)
        
        print(f"[INFO]  {len(images)} images loaded paired with {len(labels)} labels")
        print(f"\t\tSlice: {slice_mode}")
        print(f"\t\tSize: {image_size}")

        # Remove mri images that have invalid labels
        images = np.delete(images, to_be_classified, axis=0)

        # ! Generate list of hashes for each image as to link images to corresponding data
        hashes = [
            hashlib.sha256(images[i].tobytes()).hexdigest()
            for i in range(len(images))
        ]
        # append hashes as new column to clinical_data
        clinical_data["IMAGE_HASH"] = hashes
        clinical_data.to_csv(f'../data/{slice_mode}/clinical_data_{image_size[0]}.csv', index=False)

    # ! Load clinical and image data
    # Load clinical data from file
    clinical_data = pd.read_csv(f'../data/{slice_mode}/clinical_data_{image_size[0]}.csv', low_memory=False)
    # get image labels as np array
    labels = np.asarray(clinical_data['DIAGNOSIS'].tolist())

    # Load all images
    images = np.load(f"../data/dataset/all_{slice_mode}_slices_{image_size[0]}.npy", allow_pickle=True)

    # Generate list of hashes for each image as to link images to corresponding data
    hashes = [
        hashlib.sha256(images[i].tobytes()).hexdigest()
        for i in range(len(images))
    ]

    invalid_hashes = []
    for i, image_hash in enumerate(hashes):
        if image_hash not in clinical_data["IMAGE_HASH"].tolist():
            invalid_hashes.append(i)
    # Remove invalid images from images
    images = np.delete(images, invalid_hashes, axis=0)

    # Perform stratified train/test split on images and labels
    train_data, test_data, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Split train into train and val
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_data, train_labels, test_size = 0.2, random_state=42, stratify=train_labels
    )

    # Augment training data
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )
    # Fit datagen to training data
    datagen.fit(train_data)

    print(f"[INFO]  {len(train_data)} images in training set")
    print(f"\tTrain NC: {train_labels.tolist().count(0):<3}")
    print(f"\tTrain AD: {train_labels.tolist().count(1):<3}")
    print(f"[INFO]  {len(test_data)} images in testing set")
    print(f"\tTest NC: {test_labels.tolist().count(0):<3}")
    print(f"\tTest AD: {test_labels.tolist().count(1):<3}")

    # ! Train and test own CNN model
    train_and_test(train_data, val_data, test_data, train_labels, val_labels, test_labels)
    # ! Train and test pre-trained CNN model (ResNET50)
    # train_and_test_pretrained(train_data, test_data, train_labels, test_labels)
    # ! Train and test model with self-attention layer
    # train_and_test_attention(train_data, test_data, train_labels, test_labels)
    
    # ! Plot entire mri scan
    # patient_id = str(
    #     clinical_data.loc[clinical_data["IMAGE_HASH"] == train_data_hashes[0],
    #                       "NAME"].values[0])
    # plot_mri_slices(train_data[0], train_labels[0], patient_id, slice_mode)


if __name__ == "__main__":
    try:
        image_data_classification()
    except KeyboardInterrupt as keyboard_interrupt:
        print("[EXIT] User escaped")
