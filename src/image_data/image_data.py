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
# Import resnet50
from tensorflow.keras.applications.resnet50 import ResNet50
# Import VGG16
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import to_categorical
from tensorflow.keras.utils import to_categorical

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


def tabularise_image_data(data=None):
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


def show_slices(slices, name):
    """ Function to display row of image slices

    Args:
        slices (np.array): image slices
        name (str): name of the image
    """
    filename = name.replace(".nii", "") + ".png"
    # if file exists
    if os.path.isfile(f"../data/images/{filename}"):
        return

    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        # remove axis
        plt.axis('off')
        # Remove padding
        plt.tight_layout()
    # Save figure
    plt.savefig("../data/images/" + filename)


def get_mri_scan(patient_id, data_dir=None):
    """Loads in MRI scan

    Args:
        patient_id (str): patient id
        data_dir (str): directory to load data from
    Returns:
        np.array: MRI scan
    """

    mri_scan = None
    global MRI_IMAGE_DIR
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


def get_mri_data(patient_id, data_dir=None):
    """ Loads in MRI scan data

    Args:
        patient_id (str): patient ID
    Returns:
        numpy.ndarray: MRI scan data
    """
    mri_scan = None
    try:
        # Load MRI scan's data
        mri_scan = get_mri_scan(patient_id, data_dir).get_fdata()
    except OSError:
        print(f"[!]\tMRI scan file corrupt for patient: {patient_id}")
        # If invalid_files.csv doesn't exist, create it
        if not os.path.isfile("../data/invalid_files.csv"):
            with open("../data/invalid_files.csv", "w",
                      encoding="utf-8") as file:
                file.write("patient_id\n")
            # Close file
            file.close()
        # If patient_id is not in invalid_files.csv, add it
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


def create_cnn():
    """Create CNN model"""
    print("[INFO] Creating CNN model")
    # Create CNN model
    model = models.Sequential()
    model.add(
        layers.Conv2D(32, (3, 3),
                      activation="relu",
                      input_shape=(image_size[0], image_size[1], 3)))
    # NOTE: layers.MaxPooling2D is used to reduce the size of the image
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # NOTE: layers.Conv2D is used to add more layers/filters for each 3x3 segment of the image to the CNN
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # NOTE: layers.Flatten is used to convert the image to a 1D array
    model.add(layers.Flatten())
    # NOTE: layers.Dense is used to add more layers to the CNN
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer="optimizers.Adam",
                  metrics=["accuracy"])
    return model


def create_cnn2():
    """ Own CNN model """
    model = models.Sequential()
    model.add(
        layers.Conv2D(100, (3, 3), activation='relu', input_shape=(72, 72, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(70, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(50, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(optimizers.Adam(learning_rate=0.001),
                  "sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])

    model.summary()
    return model


def train_cnn(data, labels):
    """Trains a CNN on the data

    Parameters
    ----------
    data : list
        Data to train on
    """
    if len(data) == 0:
        print("[!]\tNo data to train on")
        return
    epochs = 20
    # If trained model already exists
    if os.path.isfile(f"../data/models/cnn_{epochs}.h5"):
        print("[!]\tTrained model already exists")
        return
    print(f"[INFO] {len(data)} brain scans to train on")
    # Create CNN
    cnn = create_cnn2()
    print("[INFO] Training CNN")
    # Train CNN
    history = cnn.fit(data, labels, epochs=epochs)

    # Plot history
    # plot_history(history, epochs)
    # Save CNN
    cnn.save(f"../data/models/cnn_{epochs}.h5")


def test_cnn(data, labels):
    """Tests a CNN on the data

    Parameters
    ----------
    data : list
        Data to test on
    """
    if len(data) == 0:
        print("[!]\tNo data to test on")
        return

    print(f"[INFO] {len(data)} brain scans to test on")
    # Create CNN
    cnn = create_cnn2()
    # Load CNN
    cnn.load_weights("../data/models/cnn_20.h5")
    # Test CNN
    results = cnn.evaluate(data, labels)
    print(type(results))


def create_train_get_model(train_data, train_labels, epochs, batch_size, guid):
    """Trains a CNN on the data

    Args:
        train_data (list): Data to train on
        train_labels (list): Labels to train on
        epochs (int): Number of epochs to train for
        batch_size (int): Batch size
        guid (str): Guid of the patient

    Returns:
        model: Trained model
    """
    size = image_size[0]
    # If model already exsizets
    if not os.path.isfile(f"../models/cnn_{guid}.h5"):
        model = models.Sequential()
        model.add(
            layers.Conv2D(100, (3, 3),
                          activation='relu',
                          input_shape=(size, size, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(70, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(50, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        # Add dense layer for binary classification
        model.add(layers.Dense(1, activation="sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer=optimizers.Adam(learning_rate=0.001),
                      metrics=["acc"])

        print("[INFO] Training own CNN")
        history = {"acc": [], "val_acc": [], "loss": [], "val_loss": []}
        for epoch in range(epochs):
            start = time.time()
            one_history = model.fit(train_data,
                                    train_labels,
                                    epochs=1,
                                    batch_size=batch_size,
                                    validation_split=0.1,
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
        # save history to csv
        with open(f"../data/history/cnn_{guid}.csv", "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["acc", "val_acc", "loss", "val_loss"])
            for i in range(epochs):
                writer.writerow([
                    history["acc"][i], history["val_acc"][i],
                    history["loss"][i], history["val_loss"][i]
                ])
        
        # Plot history stats to see if model is overfitting
        plot_history(history, guid)
        print(f"[INFO] Saving model to ../models/epoch_{epochs}/cnn_{guid}.h5")
        # Save model to models folder
        model.save(f"../models/cnn_{guid}.h5")
    else:
        # print(f"[!]\tModel already exists")
        model = models.load_model(f"../models/cnn_{guid}.h5")
    return model


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_and_test(train_data, test_data, train_labels, test_labels):
    """ Trains and tests a CNN on the data

    Args:
        train_data (list): training data
        test_data (list): testing data
        train_labels (list): training labels
        test_labels (list): testing labels

    Returns:
        None
    """
    # Compute the mean and the variance of the training data for normalization.

    accs = []
    f1 = []
    precision = []
    recall = []

    # Ensure the same random is generated each time
    random.seed(42)
    seeds = random.sample(range(1, 20), 5)
    seeds.sort()

    print(f"{'Acc':<6} {'Loss':<6} {'seed':<6}")
    for seed in seeds:
        reset_random_seeds(seed)

        epochs = 10
        batch_size = 32
        guid = f"{seed}_{epochs}_{batch_size}_{image_size[0]}"

        # Create, train and get model
        model = create_train_get_model(train_data, train_labels, epochs,
                                       batch_size, guid)
        # Evaluate model
        score = model.evaluate(test_data, test_labels, verbose=0)

        loss = f"{score[0]*100:.2f}"
        acc = f"{score[1]*100:.2f}"
        print(f'{acc:<6} {loss:<6} {seed:<6}')

        accs.append(score[1])

        # Predict labels for test data
        test_predictions = model.predict(test_data)
        test_label = utils.to_categorical(test_labels, 2)

        true_label = np.argmax(test_label, axis=1)
        predicted_label = np.argmax(test_predictions, axis=1)

        class_report = classification_report(true_label,
                                             predicted_label,
                                             output_dict=True,
                                             zero_division=0)
        precision.append(class_report["macro avg"]["precision"])
        recall.append(class_report["macro avg"]["recall"])
        f1.append(class_report["macro avg"]["f1-score"])

    # Calculate statistics
    avg_acc = f"{np.array(accs).mean() * 100:.2f}"
    avg_precision = f"{np.array(precision).mean() * 100:.2f}"
    avg_recall = f"{np.array(recall).mean() * 100:.2f}"
    avg_f1 = f"{np.array(f1).mean() * 100:.2f}"

    std_acc = f"{np.array(accs).std() * 100:.2f}"
    std_precision = f"{np.array(precision).std() * 100:.2f}"
    std_recall = f"{np.array(recall).std() * 100:.2f}"
    std_f1 = f"{np.array(f1).std() * 100:.2f}"

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
                              test_labels):
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
    with open("../data/results/results.txt", "a", encoding="utf-8") as file:
        file.write(f"pretrained,{score[1]},{score[0]}\n")

    model.save('../models/pretrained_vgg16.h5')


def train_and_test_lstm_model():
    """
    Train and test the LSTM model
    """
    print("[INFO]  Training and testing LSTM model")
    # Load data
    train_data, test_data, train_labels, test_labels = load_data()
    # Create LSTM model and train


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


def image_data_classification():
    """Image data classification"""
    global image_size
    global slice_mode
    global MRI_IMAGE_DIR
    image_size = (72, 72)
    # "center", "average_center", "area"
    slice_mode = "average_center"
    MRI_IMAGE_DIR = "../data/mri_images"
    print(f"[INFO] Image data classification\n{image_size}\t{slice_mode}")
    # ! Prepare data
    start = time.time()
    from image_data.image_prepare import prepare_images
    print(
        f"[INFO] Prepare images file imported in {time.time() - start:.2f} seconds"
    )
    prepare_images(image_size, slice_mode)
    # if ../data/{slice_mode} doesn't exist, create it
    if not os.path.exists(f"../data/{slice_mode}"):
        os.makedirs(f"../data/{slice_mode}")

    # If clinical_data.csv does not exist
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
        clinical_data = tabularise_image_data(clinical_data)
        # If any mri scans are corrupted, remove them from the tabular data
        if os.path.exists("../data/invalid_files.csv"):
            # Load invalid_files.csv
            invalid_files = pd.read_csv("../data/invalid_files.csv")
            # Remove rows in data where patient_id is in invalid_files
            clinical_data = clinical_data[~clinical_data["NAME"].
                                          isin(invalid_files["patient_id"])]

        # ! Extract labels from clinical data
        # Collect indices where DIAGNOSIS is TBD (aka "No diagnosis")
        to_be_classified = clinical_data[clinical_data["DIAGNOSIS"] ==
                                         "TBD"].index
        # Remove rows where DIAGNOSIS is TBD
        clinical_data = clinical_data.drop(to_be_classified)
        # Get labels
        labels = clinical_data['DIAGNOSIS']
        # Prepares labels for binary classification according to mode 1
        labels = np.asarray(process_labels(labels).tolist())

        # ! Create a training and test set of images corresponding to clinical data
        # Load all mri slices (The order should correspond to the order of clinical data)
        images = np.load(f"../data/dataset/all_{slice_mode}_slices_{image_size[0]}.npy",
                         allow_pickle=True)

        # Remove mri images that have invalid labels
        images = np.delete(images, to_be_classified, axis=0)

        # ! Generate list of hashes for each image as to link images to corresponding data
        hashes = [
            hashlib.sha256(images[i].tobytes()).hexdigest()
            for i in range(len(images))
        ]
        # append hashes as new column to clinical_data
        clinical_data["IMAGE_HASH"] = hashes
        clinical_data.to_csv(f'../data/{slice_mode}/clinical_data_{image_size[0]}.csv',
                             index=False)

    # ! Load clinical and image data
    # Load clinical data from file
    clinical_data = pd.read_csv(f'../data/{slice_mode}/clinical_data_{image_size[0]}.csv',
                                low_memory=False)
    # get image labels as np array
    labels = np.asarray(clinical_data['DIAGNOSIS'].tolist())
    # Load all images
    images = np.load(f"../data/dataset/all_{slice_mode}_slices_{image_size[0]}.npy",
                     allow_pickle=True)

    # Generate list of hashes for each image as to link images to corresponding data
    hashes = [
        hashlib.sha256(images[i].tobytes()).hexdigest()
        for i in range(len(images))
    ]

    invalid_hashes = []
    for i, hash in enumerate(hashes):
        if hash not in clinical_data["IMAGE_HASH"].tolist():
            invalid_hashes.append(i)
    # Remove invalid images from images
    images = np.delete(images, invalid_hashes, axis=0)

    # Perform stratified train/test split on images and labels
    train_data, test_data, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.1, random_state=42, stratify=labels)

    print(f"[INFO] {train_data.shape[0]} training samples")
    print(f"[INFO] {test_data.shape[0]} test samples")

    # If results.txt doesn't exist
    if not os.path.exists("../data/results/results.txt"):
        # Create results.txt
        with open("../data/results/results.txt", "w",
                  encoding="utf-8") as file:
            file.write("model,acc,loss")

    # ! Train and test own CNN model
    train_and_test(train_data, test_data, train_labels, test_labels)
    # ! Train and test pre-trained CNN model (ResNET50)
    # train_and_test_pretrained(train_data, test_data, train_labels, test_labels)
    # ! Train and test model with self-attention layer
    # train_and_test_attention(train_data, test_data, train_labels, test_labels)

    train_data_hashes = [
        hashlib.sha256(train_data[i].tobytes()).hexdigest()
        for i in range(len(train_data))
    ]
    # Get patient_id as string for first element in train_data_hashes from clinical data
    patient_id = str(
        clinical_data.loc[clinical_data["IMAGE_HASH"] == train_data_hashes[0],
                          "NAME"].values[0])
    # ! Plot entire mri scan
    plot_mri_slices(train_data[0], train_labels[0], patient_id)


if __name__ == "__main__":
    try:
        image_data_classification()
    except KeyboardInterrupt as keyboard_interrupt:
        print("[EXIT] User escaped")
