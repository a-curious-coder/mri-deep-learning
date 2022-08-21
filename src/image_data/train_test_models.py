""" Process, transform, and load image data """
import csv
import hashlib
import multiprocessing
import os
import random
import time
import math
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models, optimizers, utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.regularizers import l2 as l2_regularizer
from tensorflow.keras import datasets

from image_data.prepare_data import prepare_images
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

from tensorflow.keras import callbacks

from plot import *
import sys

MRI_IMAGE_DIR = "../data/mri_images"
EDA = False


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


def reshuffle_test_train_sets(seed):
    """ Reshuffles the test and training sets in storage
    Args:
        seed (int): seed for random number generator
    """
    print(f"[INFO] Reshuffling test and training sets with seed {seed}")
    train_dir = f"../data/dataset/{slice_mode}_{image_size[0]}/train"
    test_dir = f"../data/dataset/{slice_mode}_{image_size[0]}/test"
    # Redistribute test and training sets for each label AD and NL
    for label in ["AD", "NL"]:
        # Get list of files in test and training directories
        train_files = os.listdir(os.path.join(train_dir, label))
        test_files = os.listdir(os.path.join(test_dir, label))

        # Dictionary of dirs for each file
        file_dict = {f: os.path.join(train_dir, label, f) for f in train_files}
        test_dict = {f: os.path.join(test_dir, label, f) for f in test_files}
        # Merge dicts
        file_dict.update(test_dict)

        all_files = list(file_dict.keys())
        random.seed(seed)
        random.shuffle(all_files)

        # Save all_files to text file
        with open(f"../data/dataset/{slice_mode}_{image_size[0]}/{seed}_files.csv", "w", encoding = 'utf-8') as f:
            # Write headers
            f.write("set,file\n")
            for file_name in all_files:
                dataset = "train" if file_name in train_files else "test"
                f.write(f"{dataset},{file_dict[file_name]}\n")

        # Split files into test and train sets
        test_files = all_files[:len(test_files)]
        train_files = all_files[len(test_files):]

        # Move test files to label directory in test folder
        for test_file in test_files:
            shutil.move(file_dict[test_file], os.path.join(test_dir, label, test_file))

        # Move train files to label directory in train folder
        for train_file in train_files:
            shutil.move(file_dict[train_file], os.path.join(train_dir, label, train_file))

    print(f"[INFO] Reshuffling complete (seed : {seed}) ")


def get_dataset_generators(train_dir, test_dir, seed, batch_size):
    """ Loads the training and testing datasets
    Args:
        train_dir (str): path to training images
        test_dir (str): path to testing images
    """
    reshuffle_test_train_sets(seed)

    datagen_args = dict(
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        rescale=1./255,
        validation_split = 0.2
    )

    # Image augmentation on training set
    train_datagen = ImageDataGenerator(**datagen_args)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size= image_size,
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = True,
        seed = seed,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size= image_size,
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = True,
        seed = seed,
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size= image_size,
        batch_size = batch_size,
        class_mode = 'binary'
    )

    # NOTE: Specifying subset validation ensures no data is augmented for this set
    print(f"Number of samples in training set: {train_generator.samples}")
    print(f"Number of samples in validation set: {validation_generator.samples}")
    print(f"Number of samples in test set: {test_generator.samples}")
    print(f"Number of classes in training set: {train_generator.class_indices}")
    # Print distribution of classes in training set
    print(f"Number of samples per class in training set: {int(train_generator.samples / len(train_generator.class_indices))}")

    return train_generator, validation_generator, test_generator


def create_model():
    """Creates an original/own Convolutional Neural Network model
    Returns:
        model (keras.models.Sequential): CNN model
    """
    print('[INFO] Creating model')
    size = image_size[0]

    model = models.Sequential()
    # Convolutional layer 1
    model.add(
        layers.Conv2D(100, (3, 3),
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=l2_regularizer(l=0.01),
            kernel_constraint=unit_norm(),
            input_shape=(size, size, 3)
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    # Convolutional layer 2
    model.add(
        layers.Conv2D(70, (3, 3),
            activation='relu',
            kernel_regularizer=l2_regularizer(l=0.01)
        )
    )
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Final convolutional layer
    model.add(
        layers.Conv2D(50, (3, 3), 
            activation='relu', 
            kernel_regularizer=l2_regularizer(l=0.01)
        )
    )
    model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    metrics=["acc"])

    return model


def train_model(model, train_generator, validation_generator, epochs, guid):
    """ Trains CNN model
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
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=f"../models/{guid}_best.h5",
            monitor="val_loss",
            mode = 'min',
            save_best_only=True,
            verbose = 0
        ),
        callbacks.EarlyStopping(
            monitor ="val_loss",
            min_delta=0.4,
            mode='auto',
            patience = 5,
            restore_best_weights = True,
            baseline = None,
            verbose = 1,
        ),
        callbacks.ReduceLROnPlateau(monitor = "val_loss",
                                    mode = "min",
                                    factor = 0.1,
                                    patience = 2,
        ),
        callbacks.TensorBoard(
                            log_dir="../models/logs/{}".format(guid),
                            histogram_freq=1
        )
    ]


    print(f"[INFO] Training CNN model using {slice_mode} slices")

    # for epoch in range(epochs):
    # start = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=callbacks_list
    )

    history = history.history
    # Save training history to csv
    with open(f"../data/history/cnn_{guid}.csv", "w", encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["acc", "val_acc", "loss", "val_loss"])
        for i in range(len(history["acc"])):
            writer.writerow([
                history["acc"][i], history["val_acc"][i],
                history["loss"][i], history["val_loss"][i]
            ])
    plot_history(history, guid)
    return model


def train_and_test(train_dir, test_dir, epochs, batch_size):
    """ Trains and tests a CNN on the data

    Args:
        train_data (list): training data
        test_data (list): testing data
        train_labels (list): training labels
        test_labels (list): testing labels

    Returns:
        None
    """

    # TODO: Get these values from source / more reliable way
    val_size = 0.2
    test_size = 0.1
    # test_size = round(len(test_data)/(len(train_data) + len(test_data)), 1)
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
        train_gen, validation_gen, test_gen = get_dataset_generators(train_dir, test_dir, seed, batch_size)

        reset_random_seeds(seed)

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
        if os.path.isfile(f"../models/cnn_{guid}.h5") and not retrain:
            print("[INFO] Loading model from storage")
            model = models.load_model(f"../models/cnn_{guid}.h5")

        if model is None:
            # ! Create train/validation test split
            # ! 1 Create model
            model = create_model()
            # ! 2 Train model
            model = train_model(model, train_gen, validation_gen, epochs, guid)
            # Save trained model
            model.save(f"../models/cnn_{guid}.h5")

        # ! 3 Evaluate model 
        score = model.evaluate(test_gen, verbose=0)

        loss = f"{score[0]*100:.2f}"
        acc = f"{score[1]*100:.2f}"
        print(f'{acc:<6} {loss:<6} {seed:<6}')

        # 
        # Predict labels for test data
        test_predictions = model.predict(test_gen)
        test_label = utils.to_categorical(test_gen.classes, 2)

        true_label = np.argmax(test_label, axis=1)
        # predicted_label = np.argmax(test_predictions, axis=1)
        # Round test predictions to nearest integer
        predicted_label = np.round(test_predictions)
        
        class_report = classification_report(
                                            true_label,
                                            predicted_label,
                                            output_dict=True,
                                            zero_division=0
                                            )
        
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


def prepare_files():
    """ Prepare files for training and testing """
    # If results.csv doesn't exist
    if not os.path.exists("../data/results/results.csv"):
        # Create results.csv
        with open("../data/results/results.csv", "w", encoding="utf-8") as file:
            file.write("model,acc,loss,slice_mode,image_size\n")

    if not os.path.exists("../models/training_log.csv"):
        with open("../models/training_log.csv", "w", encoding="utf-8") as file:
            file.write("guid,seed,epochs,batch_size,slice_mode,height,width,acc,loss,test_size,val_size,individual\n")



def image_data_classification(im, sl):
    """Image data classification"""
    global MRI_IMAGE_DIR
    global image_size
    global slice_mode
    MRI_IMAGE_DIR = "../data/mri_images"

    image_size = im
    slice_mode = sl
    epochs = 50
    batch_size = 32
    
    prepare_files()
    print("")
    print(f"[INFO] Image data classification")

    # Train directory
    train_dir = f'../data/dataset/{slice_mode}_{image_size[0]}/train'
    test_dir = f'../data/dataset/{slice_mode}_{image_size[0]}/test'
    
    
    # class_mapping = {v:k for k,v in train_generator.class_indices.items()}
    # show_grid(x, 1, 2,label_list=y, show_labels=True,figsize=(20,10),class_mapping = class_mapping)


    # ! Train and test own CNN model
    train_and_test(train_dir, test_dir, epochs, batch_size)
    # ! Train and test pre-trained CNN model (ResNET50)
    # train_and_test_pretrained(train_data, test_data, train_labels, test_labels)
    # ! Train and test model with self-attention layer
    # train_and_test_attention(train_data, test_data, train_labels, test_labels)
    
    # ! Plot entire mri scan
    # patient_id = str(
    #     clinical_data.loc[clinical_data["IMAGE_HASH"] == train_data_hashes[0],
    #                       "NAME"].values[0])
    # plot_mri_slices(train_data[0], train_labels[0], patient_id, slice_mode)
