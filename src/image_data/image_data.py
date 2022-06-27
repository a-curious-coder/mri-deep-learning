""" Process, transform, and load image data """
from plot import *
from misc import progress_bar
from tensorflow.keras import layers, models
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import glob
import os

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import skimage.segmentation as seg
import tensorflow
from tensorflow.keras.utils import to_categorical


MRI_IMAGE_DIR = "../data/mri_images"
EDA = False


def handle_null_values(data):
    """Handleks null values from data

    Args:
        data (pd.DataFrame): original data

    Returns:
        pd.DataFrame: identifies and removes null/nan values
    """
    null_val_per_col = data.isnull().sum().to_frame(
        name='counts').query('counts > 0')
    # print(null_val_per_col)
    # NOTE: Null value quantity is same as max number of rows
    # NOTE: Thus, delete whole columns instead
    # Get column names
    columns = null_val_per_col.index.tolist()
    # Drop columns with null values
    data.drop(columns, axis=1, inplace=True)
    return data


def filter_data(data, scan_num=None, project=None):
    """Filters full data-set to records we want

    Args:
        data (pd.DataFrame): full data-set
    """
    # If the filtered data is not already available, run the data filtering
    print(
        f"[*]\tRefining big data-frame to SCAN_NUM: {scan_num}, PROJECT: {project}")
    if not os.path.exists("../data/filtered_data.csv"):
        # Filter data by scan number and study
        # if PROJECT is not none
        if project is not None:
            data = data[data['PROJECT'] == project]
        if scan_num is not None:
            data = data[data['SCAN_NUM'] == scan_num]
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
        # Save filtered data
        data.to_csv("../data/filtered_data.csv", index=False)
    else:
        print("[*]\tLoading filtered data-frame from file")
        data = pd.read_csv("../data/filtered_data.csv", low_memory=False)
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
    # Create batches of patient_ids
    batches = [patient_ids[i:i + 10] for i in range(0, len(patient_ids), 10)]
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
        temp = pd.DataFrame({"NAME": patient_ids, "SHAPE": image_shapes,
                            "GENDER": genders, "DIAGNOSIS": classifications})
        info.append(temp)
    final = pd.concat(info, axis=0, ignore_index=False)
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


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    """ Show the image presented in the parameters

    Args:
        image (np.array): image to be shown
        nrows (int): number of rows in the image
        ncols (int): number of columns in the image
        cmap (str): colour map

    Returns:
        None
    """
    fig, ax = plt.subplots(nrows - nrows, ncols - ncols, figsize=(9, 9))
    ax.imshow(image, cmap - ' gray ')
    ax.axis('off')
    return fig, ax


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


def select_frame(patient_id: str):
    """ Selects the frame of the MRI scan
    Args:
        image (np.array): MRI scan
    """
    # TODO: Train machine/deep learning model to perform better frame selection
    #       versus choosing the center frame from each angle of the scan

    mri_scan = get_mri_scan_data(patient_id)

    # Extract slices from each angle
    center_slices = get_center_slices(mri_scan)

    # Resize the slices for the plots/images
    im1, im2, im3 = resize_slices(center_slices, (250, 250))

    show_slices([im1, im2, im3], patient_id)

    # NOTE: Formats scan images for CNN
    # im = np.array([im1,im2,im3]).T


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


def get_mri_scan(patient_id):
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
    # print(f"[INFO] Loading MRI scan for patient {patient_id} ")
    # ! Have assumed each patient folder has only one MRI scan file
    files = os.listdir(MRI_IMAGE_DIR + "/" + patient_id)
    # Remove files that begin with .
    files = [file for file in files if not file.startswith(".")]
    if len(files) > 1:
        print(f"[!]\tMultiple MRI scan files found for patient: {patient_id}")
        for file in files:
            print(file[:-10])
        return

    for file in files:
        # If file is an MRI scan (With .nii extension)
        if file.endswith(".nii"):
            # print(MRI_IMAGE_DIR + "/" + patient_id + "/" + file)
            # return sitk.ReadImage(MRI_IMAGE_DIR + "/" + patient_id + "/" + file)
            return nib.load(MRI_IMAGE_DIR + "/" + patient_id + "/" + file)


def get_mri_scan_data(patient_id):
    """ Loads in MRI scan data

    Args:
        patient_id (str): patient ID
    Returns:
        numpy.ndarray: MRI scan data
    """
    # Load MRI scan
    mri_scan = get_mri_scan(patient_id).get_fdata()

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
    return [get_mri_scan(patient_id) for patient_id in patient_ids]


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
    return [get_mri_scan_data(patient_id) for patient_id in patient_ids]


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
    model.add(layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(72, 72, 3)))
    # NOTE: MaxPooling2D is used to reduce the size of the image
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # NOTE: Conv2D is used to add more layers/filters for each 3x3 segment of the image to the CNN
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # NOTE: Flatten is used to convert the image to a 1D array
    model.add(layers.Flatten())
    # NOTE: Dense is used to add more layers to the CNN
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def plot_history(history, epochs):
    """Plot training history"""
    # Print history
    input(history.history.keys())
    # Plot training history
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["loss"], label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    # Save figure
    plt.savefig(f"../data/images/history_{epochs}.png")


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
    cnn = create_cnn()
    print("[INFO] Training CNN")
    # Train CNN
    history = cnn.fit(data, labels, epochs=epochs)

    # Plot history
    plot_history(history, epochs)
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
    cnn = create_cnn()
    # Load CNN
    cnn.load_weights("../data/models/cnn_10.h5")
    # Test CNN
    results = cnn.evaluate(data, labels)
    print(f"[INFO] Test results: {results}")


def generate_dataset(patient_ids):
    """ Generates dataset for CNN

    Args:
        patient_ids (list): patient IDs
    """
    batch = "all"
    # if ../data/dataset/all.npy doesn't exist
    if not os.path.exists(f"../data/dataset/{batch}.npy"):
        print("[INFO] Generating image dataset")
        # Create final data-set for classification
        # Split patient_ids into batches of 10
        batches = np.array_split(patient_ids, len(patient_ids)//10)
        for batch, patient_ids in enumerate(batches):
            print(f"[{batch}] Saving  mri scan data")
            if os.path.exists(f"../data/dataset/{batch}_batch_data.npy"):
                continue
            final = []
            all_mri_center_slices = []
            # print(f"[INFO] Getting all {len(patient_ids)} mri scans worth of data")
            mri_scans_data = get_mri_scans_data(patient_ids)

            # print("[INFO] Extracting center slices of each mri scan angle")
            for mri_scan in mri_scans_data:
                all_mri_center_slices.append(get_center_slices(mri_scan))

            # print("[INFO] Concatenating all center slices for each scan")
            for i, center_slices in enumerate(all_mri_center_slices):
                print(f"{i/len(all_mri_center_slices)*100}%", end="\r")
                # Resizing each center slice to 72/72
                # TODO: Determine an optimal image size
                # NOTE: Could it be plausible to suggest a size closest to native scan resolution is best?
                #   Maintain as much quality?
                im1, im2, im3 = resize_slices(center_slices, (72, 72))
                # Convert these image slices of scan to concatenated np array for CNN
                all_angles = np.array([im1, im2, im3]).T
                # print(type(all_angles))
                final.append(all_angles)
            # Save final data-set to file
            np.save(
                f"../data/dataset/{batch}_batch_data.npy", final, allow_pickle=True)

        npfiles = glob.glob("../data/dataset/*.npy")
        npfiles.sort()
        # Merge all .npy files into one file
        all_arrays = []
        for npfile in npfiles:
            if "all" in npfile:
                continue
            all_arrays.append(np.load(npfile, allow_pickle=True))
        # Flatten 2d array
        all_arrays = [item for sublist in all_arrays for item in sublist]
        # Print length of all_arrays
        print(f"[INFO] Length of all_arrays: {len(all_arrays)}")
        np.save("../data/dataset/all.npy", all_arrays)
# Unused


def unused_passholder():
    """ Function to passholder images"""
    def get_n_mri_scans(n):
        """ Loads a set number of mri scans from the data directory

        Parameters
        ----------  n : int
            Number of scans to load

        Returns
        -------
        list
            MRI scans
        """
        # Load image_details
        image_details = pd.read_csv(
            '../data/image_details.csv', low_memory=False)
        # image_details equals where diagnosis is not MCI
        image_details = image_details[image_details['diagnosis'] != 'MCI']

        # Get first n names from image_details
        patient_ids = image_details['name'].head(n).tolist()
        patient_scans = [get_mri_scan(patient_id)
                         for patient_id in patient_ids]
        # Get patient id classification from image_details
        patient_diagnosis = image_details.loc[image_details['name'].isin(
            patient_ids), 'diagnosis']

        return zip(patient_ids, patient_diagnosis, patient_scans)

    def train_cnn(data):
        """Trains CNN to classify best frame of MRI scan for classification model

        Parameters
        ----------
        data : list
            List of tuples containing patient_id, diagnosis, and MRI scan
        """

        # Read in all MRI scans using patient IDs from data
        mri_scans = get_mri_scans(data['PATIENT_ID'])
        # Remove last 64 frames from each MRI scan
        mri_scans = [(patient_id, diagnosis, scan[:-64])
                     for patient_id, diagnosis, scan in mri_scans]
        # Remove first 64 frames from each MRI scan
        mri_scans = [(patient_id, diagnosis, scan[64:])
                     for patient_id, diagnosis, scan in mri_scans]
        # Detect which frame has largest surface area
        mri_scans = [(patient_id, diagnosis, scan[np.argmax(scan.sum(axis=1))])
                     for patient_id, diagnosis, scan in mri_scans]

    def find_bounding_box(image):
        """Finds bounding box of image

        Parameters
        ----------
        image : numpy.ndarray
            Image to find bounding box of

        Returns
        -------
        numpy.ndarray
            Bounding box of image
        """
        # Find bounding box of image
        bounding_box = np.zeros(image.shape)
        bounding_box[image > 0] = 1
        # Find bounding box of image
        bounding_box = np.argwhere(bounding_box)
        # Find bounding box of image
        bounding_box = np.min(bounding_box, axis=0)
        # Find bounding box of image
        bounding_box = np.max(bounding_box, axis=0)
        # Return bounding box
        return bounding_box

    def strip_skull_from_mri(image):
        """Strips skull from MRI scan
        TODO: Incomplete

        Parameters
        ----------
        image : numpy.ndarray
        """
        image_path = "plots/F/AD/S232906.png"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert to float
        image = image.astype(float)
        # Rescale gray scale between 0-255
        image = (image - image.min()) / (image.max() - image.min())
        # Convert to uint8
        image = (image * 255).astype(np.uint8)
        # show_image("gray", image, "gray")
        # to binary using Otsu's method
        # Threshold the image
        ret, thresh_custom = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ShowImage ( " Applying Otsu ' , thresh_custom , ' gray " )
        points = circle_points(200, [128, 128], 35)[: -1]
        # fig , ax image_show ( gray )
        # ax.plot ( points [ :, 0 ] , points [ :, 1 ] , ' --r ' , Lw = 3 )
        snake = seg.active_contour(thresh_custom, points)
        fig, ax = image_show(thresh_custom)
        # ax.plot ( points [ :, 0 ] , points [ :, 1 ] , ' --r ' , Lw = 3 )
        ax.plot(snake[:, 0], snake[:, 1], ' b ', lw=3)
        # OTSU THRESHOLDING
        _, binarized = cv2.threshold(
            filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(binarized.shape, ' is Otsu thresholding value ')
        # ShowImage ( ' Brain with Skull XXX ' , binarized , ' gray ' )
        plt.imshow(binarized)

    def circle_points(resolution, center, radius):
        """Generate points which define a circle on an image.Centre refers to the centre of the circle"""
        radians = np.linspace(0, 2*np.pi, resolution)
        c = center[1] + radius*np.cos(radians)  # polar co - ordinates
        r = center[0] + radius*np.sin(radians)
        return np.array([c, r]) . T

    def compare_mri_images(image_details):
        """ Compares images from different scans """
        details = []
        images = []

        print("Diagnosis\tFrequency")
        # For each unique diagnosis
        for diagnosis in image_details['diagnosis'].unique():
            temp = image_details[image_details['diagnosis'] == diagnosis]
            print(f"{diagnosis}\t{len(temp)}")
            # Get patient id
            patient_id = temp.head(1)['name'].values[0]
            # Append patient's details
            details.append(temp.head(1))
            # Load MRI scan from patient_id
            image = get_mri_scan(patient_id)
            # Rearrange 3D array
            # slices = sitk.GetArrayFromImage(image)
            # plot_mri_image(patient_id, patient_diagnosis, slices)
            # Append rearranged image format to images
            images.append(slices)

    def show_image(title, img, ctype):
        plt.figure(figsize=(10, 10))
        if ctype == 'bgr':
            b, g, r = cv2.split(img)
            # get b , g , r
            rgb_img = cv2.merge([r, g, b])
            # switch it to rgb
            plt.imshow(rgb_img)
        elif ctype == "hsv":
            rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            plt.imshow(rgb)
        elif ctype == "gray":
            plt.imshow(img, cmap="gray")
        elif ctype == "rgb":
            plt.imshow(img)
        else:
            raise Exception("Unknown colour")
        plt.axis('off')
        plt.title(title)
        plt.show()

    pass


def main():
    """Image data classification"""
    # TODO: Skull stripping
    # TODO: Effective frame selection

    try:
        print("🅸 🅼 🅰 🅶 🅴 🅼 🅾 🅳 🅴 🅻")

        # If ..data/image_details.csv exists, load it
        if not os.path.exists("../data/image_details.csv"):
            # Load in mri data schema
            data = pd.read_csv(
                "../data/adni_all_aibl_all_oasis_all_ixi_all.csv", low_memory=False)

            # NOTE: Filters data for the first scan of each patient for AIBL project
            data = filter_data(data, scan_num=1, project="AIBL")

            # Create a tabular representation of the classification for each image in the data
            data = tabularise_image_data(data)
        else:
            data = pd.read_csv("../data/image_details.csv")

        # Count quantity for each unique scan resolution in dataset
        data_shape = data['SHAPE'].value_counts()
        print(
            f"[INFO] {len(data)} total scans\n\tScans per resolution\n{data_shape}")
        labels = data['DIAGNOSIS'].tolist()
        # Convert "NL" to 0
        labels = [0 if label == "NL" else label for label in labels]
        # Convert "MCI" to 1
        labels = [1 if label == "MCI" else label for label in labels]
        # Convert "AD" to 2
        labels = [2 if label == "AD" else label for label in labels]
        labels = np.array(labels)
        # Get all patient_ids from data
        patient_ids = data['NAME'].unique()

        # Generate dataset
        generate_dataset(patient_ids)
        # Load all.npy file
        dataset = np.load("../data/dataset/all.npy", allow_pickle=True)

        # Delete all other .npy files
        # for npfile in glob.glob("../data/dataset/*.npy"):
        #     if npfile != "../data/dataset/all.npy":
        #         os.remove(npfile)
        # split dataset into train/test
        print(f"[INFO] dataset shape: {dataset.shape}")
        print("[INFO] Splitting dataset into train/test")
        train_data = dataset[:int(len(dataset) * 0.8), :, :, :]
        test_data = dataset[:int(len(dataset) * 0.2), :, :, :]
        train_labels = labels[:int(len(dataset) * 0.8)]
        test_labels = labels[:int(len(dataset) * 0.2)]
        print(type(labels), type(labels[0]))
        # Train CNN on dataset
        train_cnn(train_data, train_labels)
        # Test CNN on dataset
        test_cnn(test_data, test_labels)
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
    except KeyboardInterrupt as keyboard_interrupt:
        print("[EXIT] User escaped")


if __name__ == "__main__":
    main()
