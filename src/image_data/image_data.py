""" Process, transform, and load image data """
import glob
import os
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow
from skimage.transform import resize
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import utils

from misc import progress_bar
from plot import *

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


def get_center_slices(mri_scan):
    """Returns the center slices of the scan

    Args:
        scan (np.array): scan to be processed
    Returns:
        list: center slices of the scan
    """
    # Store mri_scan dimensions to individual variables
    n_i, n_j, n_k = mri_scan.shape

    # Calculate center frames for each scan
    center_i = (n_i - 1) // 2
    center_j = (n_j - 1) // 2
    center_k = (n_k - 1) // 2

    slice_0 = mri_scan[center_i, :, :]
    slice_1 = mri_scan[:, center_j, :]
    slice_2 = mri_scan[:, :, center_k]

    return [slice_0, slice_1, slice_2]


def resize_slices(slice_list, new_shape=(72, 72)):
    """ Resize the slices to the new shape

    Args:
        slice_list (list): list of slices
        new_shape (tuple): new shape of the image

    Returns:
        list: list of resized slices
    """
    # Resizes images for the purpose of being more visible in the plot
    im1 = resize(slice_list[0], new_shape, order=1, preserve_range=True)
    im2 = resize(slice_list[1], new_shape, order=1, preserve_range=True)
    im3 = resize(slice_list[2], new_shape, order=1, preserve_range=True)
    return [im1, im2, im3]


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

    Parameters
    ----------
    patient_id : str
        Patient ID

    Returns
    -------
    numpy.ndarray
        MRI scan
    """
    global MRI_IMAGE_DIR
    if data_dir is not None:
        MRI_IMAGE_DIR = data_dir
    # print(f"[INFO] Loading MRI scan for patient {patient_id} ")
    # ! Have assumed each patient folder has only one MRI scan file
    files = os.listdir(MRI_IMAGE_DIR + "/" + patient_id)
    # Remove files that begin with .
    files = [file for file in files if file.endswith(".nii")]
    if len(files) > 1:
        print(f"[!]\tMultiple MRI scan files found for patient: {patient_id}")
        for file in files:
            print(file[:-10])
        return
    if len(files) == 0:
        print(f"[!]\tNo MRI scan file found for patient: {patient_id}")
        return

    for file in files:
        # If file is an MRI scan (With .nii extension)
        if file.endswith(".nii"):
            # print(MRI_IMAGE_DIR + "/" + patient_id + "/" + file)
            # return sitk.ReadImage(MRI_IMAGE_DIR + "/" + patient_id + "/" + file)
            return nib.load(MRI_IMAGE_DIR + "/" + patient_id + "/" + file)


def get_mri_data(patient_id, data_dir=None):
    """ Loads in MRI scan data

    Args:
        patient_id (str): patient ID
    Returns:
        numpy.ndarray: MRI scan data
    """
    # Load MRI scan
    mri_scan = get_mri_scan(patient_id, data_dir).get_fdata()

    # If mri scan is 4D, remove 4th dimension as it's useless
    if len(mri_scan.shape) == 4:
        mri_scan = mri_scan[:, :, :, 0]

    return mri_scan


def get_mri_scans(patient_ids: list):
    """Loads in mri scans

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
        get_mri_data(patient_id, MRI_IMAGE_DIR)
        for patient_id in patient_ids
    ]


def resize_images(images, shape):
    """Resize images to a given shape

    Parameters
    ----------
    images : list
        Images to be resized
        shape : tuple of int
            Shape to resize to
    """
    # Resize images to a given shape
    return [resize(image, shape) for image in images]


def create_cnn():
    """Create CNN model"""
    print("[INFO] Creating CNN model")
    # Create CNN model
    model = models.Sequential()
    model.add(
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(72, 72, 3)))
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


def plot_history(history, guid):
    """Plot training history

    Args:
        history (keras.callbacks.History): training history
        epochs (int): number of epochs

    Returns:
        None
    """
    # Plot training history
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, history.params["epochs"]+1)

    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.savefig(f"../data/images/{guid}_acc.png")

    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(f"../data/images/{guid}_loss.png")


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


def train_and_test(X_train, X_test, y_train, y_test):
    """ Trains and tests a CNN on the data

    Args:
        X_train (list): training data
        X_test (list): testing data
        y_train (list): training labels
        y_test (list): testing labels

    Returns:
        None
    """
    # Compute the mean and the variance of the training data for normalization.
    import random

    acc = []
    f1 = []
    precision = []
    recall = []
    seeds = random.sample(range(1, 20), 5)
    for seed in seeds:
        # Reset seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        tensorflow.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = models.Sequential()
        # NOTE: layers.Conv2D is used to add more layers/filters for each 3x3 segment of the image to the CNN
        model.add(
            layers.Conv2D(100, (3, 3),
                          activation='relu',
                          input_shape=(150, 150, 3)))
        # NOTE: layers.MaxPooling2D is used to extract features and reduce the size of the image
        model.add(layers.MaxPooling2D((2, 2)))
        # NOTE: layers.Dropout is used to prevent overfitting by randomly dropping out a percentage of neurons
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
        # model.compile(optimizers.Adam(learning_rate=0.001),
        #               "sparse_categorical_crossentropy",
        #               metrics=["sparse_categorical_accuracy"])

        model.summary()
        epochs = 2
        batch_size = 32
        history = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            verbose=1)
        print(history.history.keys())
        guid = f"{seed}_{epochs}_{batch_size}"
        # Plot history stats to see if model is overfitting
        plot_history(history, guid)
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        acc.append(score[1])

        test_predictions = model.predict(X_test)
        test_label = utils.to_categorical(y_test, 3)

        true_label = np.argmax(test_label, axis=1)

        predicted_label = np.argmax(test_predictions, axis=1)

        class_report = classification_report(true_label,
                                             predicted_label,
                                             output_dict=True)
        precision.append(class_report["macro avg"]["precision"])
        recall.append(class_report["macro avg"]["recall"])
        f1.append(class_report["macro avg"]["f1-score"])

    # Calculate statistics
    avg_acc = f"{np.array(acc).mean() * 100:.2f}"
    avg_precision = f"{np.array(precision).mean() * 100:.2f}"
    avg_recall = f"{np.array(recall).mean() * 100:.2f}"
    avg_f1 = f"{np.array(f1).mean() * 100:.2f}"

    std_acc = f"{np.array(acc).std() * 100:.2f}"
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


def normalise_data(data):
    """ Normalises the data

    Args:
        data (list): List of data to normalise
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def prepare_labels(labels, mode=0):
    """ Prepares labels for
        Multiclass or binary classification
    Args:
        labels (list): labels
        mode (int): translate labels according to mode

    Returns:
        list: labels
    """
    if mode == 0:
        # Convert "NL" to 0
        labels = [0 if label == "NL" else label for label in labels]
        # Convert "MCI" to 1
        labels = [1 if label == "MCI" else label for label in labels]
        # Convert "AD" to 2
        labels = [2 if label == "AD" else label for label in labels]
    if mode == 1:
        # Convert 0 to "NL" else 1
        labels = ["NL" if label == 0 else 1 for label in labels]
    return np.array(labels)


def plot_image(image, label):
    """ Plots an image and its label

    Args:
        image (np array): image
        label (str): label
    """
    # Plot each channel separately
    fig, axs = plt.subplots(1, 3, figsize=(15, 15))
    axs[0].imshow(image[:, :, 0], cmap="gray")
    axs[0].set_title("Axial")
    axs[1].imshow(image[:, :, 1], cmap="gray")
    axs[1].set_title("Coronal")
    axs[2].imshow(image[:, :, 2], cmap="gray")
    axs[2].set_title("Saggital")
    # remove axis
    for ax in axs:
        ax.axis("off")
    # Tight layout
    fig.tight_layout()
    # Sup title
    fig.suptitle("Alzheimer's" if label == 1 else "Non-Alzheimer's")
    # plt.show()


def image_data_classification():
    """Image data classification"""
    global MRI_IMAGE_DIR
    MRI_IMAGE_DIR = "../data/mri_images"
    print("[INFO] Image data classification")
    # if not os.path.exists("../data/image_details.csv"):
    # Load in mri data schema
    data = pd.read_csv("../data/tabular_data.csv", low_memory=False)
    # NOTE: Filters data for the first scan of each patient for AIBL project
    data = filter_data(data, scan_num=1, project="AIBL")
    # Save data to file
    data.to_csv("../data/filtered_data.csv", index=False)

    # Create a tabular representation of the classification for each image in the data
    data = tabularise_image_data(data)

    labels = data['DIAGNOSIS'].tolist()
    labels = prepare_labels(labels, mode=1)

    # If dataset folder doesn't exist, create it
    if not os.path.exists("../data/dataset"):
        os.mkdir("../data/dataset")
    # Load all.npy file
    dataset = np.load("../data/all_slices.npy", allow_pickle=True)

    # Print length of dataset
    print(f"[INFO] Length of dataset: {len(dataset)}")
    # split dataset into train/test
    print(f"[INFO] dataset shape: {dataset.shape}")
    print("[INFO] Splitting dataset into train/test")
    train_data = dataset[:int(len(dataset) * 0.8), :, :, :]
    test_data = dataset[:int(len(dataset) * 0.2), :, :, :]
    train_labels = labels[:int(len(dataset) * 0.8)]
    test_labels = labels[:int(len(dataset) * 0.2)]
    # Plot image from train_data
    # plot_image(train_data[0], train_labels[0])
    train_and_test(train_data, test_data, train_labels, test_labels)
    # Train CNN on dataset
    # train_cnn(train_data, train_labels)
    # Test CNN on dataset
    # test_cnn(test_data, test_labels)
    # Train CNN to identify best frame(s) of MRI scans
    # train_cnn(data)
    # get_best_mri_frame()
    # ! Compare MRI images from each diagnosis
    # compare_mri_images(data)
    # Save the same slice of each patient's MRI scan to file
    # for index, row in data.iterrows():
    #     progress_bar(index, data.shape[0])
    #     patient_id = row['NAME']
    #     patient_diagnosis = row['DIAGNOSIS']
    #     image = get_mri_scan(patient_id)
    #     # strip_skull_from_mri(image)
    #     # Load in image
    #     slices = sitk.GetArrayFromImage(image)
    #     single_slice = slices[128]
    #     # ! Figure out which slice is most appropriate per patient
    #     im = Image.fromarray(single_slice)
    #     im = im.convert("RGB")

    #     im.save(
    #         f"plots/{row['GENDER']}/{patient_diagnosis}/{patient_id}.jpg")
    # plotted = plot_mri_slice(patient_id, patient_diagnosis, slices[128], directory=f"plots/{row['GENDER']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as keyboard_interrupt:
        print("[EXIT] User escaped")
