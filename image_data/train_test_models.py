""" Process, transform, and load image data """
import csv
import os
import random
import shutil
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow
from tensorflow.keras import Input
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D, GlobalAveragePooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2 as l2_regularizer
from tensorflow.keras.utils import to_categorical

import image_data.constants as constants
from utils.plot import generate_caption, plot_cm, plot_history

from tensorflow.keras.utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import balanced_batch_generator

class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling"""
    def __init__(self, x, y, datagen, batch_size=32):
        self.datagen = datagen
        self.batch_size = min(batch_size, x.shape[0])
        datagen.fit(x)
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)
        self._shape = (self.steps_per_epoch * batch_size, *x.shape[1:])
        
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()


def image_data_eda(data):
    """Exploratory Data Analysis on dataframe

    Args:
        data (pd.DataFrame): mri data
    """
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
    tensorflow.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def reset_train_test_sets(labels):
    """ Gets training and test """
    n_files = 0
    for label in labels:
        # Get list of files in test and training directories
        train_files = os.listdir(os.path.join(constants.TRAIN_DIR, label))
        test_files = os.listdir(os.path.join(constants.TEST_DIR, label))

        # Dictionary of dirs for each file
        file_dict = {f: os.path.join(constants.TRAIN_DIR, label, f) for f in train_files}
        test_dict = {f: os.path.join(constants.TEST_DIR, label, f) for f in test_files}
        # Dictionary containing directories for each data sample with corresponding labels
        file_dict.update(test_dict)
        
        # consistent order of files
        all_files = list(file_dict.keys())
        all_files.sort()
        print(f"Number of files in {label} directory: {len(all_files)}")
        n_files += len(all_files)

        # Split data samples into training and testing sets based on split ratio
        train_files = all_files[:int(len(all_files) * (1 - constants.TEST_SIZE))]
        test_files = all_files[int(len(all_files) * (1 - constants.TEST_SIZE)):]

        # Move files to correct directories
        for f in train_files:
            shutil.move(file_dict[f], os.path.join(constants.TRAIN_DIR, label, f))
        
        for f in test_files:
            shutil.move(file_dict[f], os.path.join(constants.TEST_DIR, label, f))
    print(f"Total number of data samples: {n_files}")


def save_training_history(history, guid):
    """ Saves the training history to a csv file
    Args:
        history (dict): training history
        guid (str): guid for the model
    """
    if constants.CLASSIFICATION == "binary":
        val_acc = str(round(np.mean(history['val_acc'])*100, 2)).replace('.', ',')
        # Get average val_acc
        guid = str(val_acc) + "_" + guid
        # Save training history to csv
        with open(f"../data/history/{constants.CLASSIFICATION}/{guid}.csv", "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["acc", "val_acc", "loss", "val_loss"])
            for i in range(len(history["acc"])):
                writer.writerow([
                    history["acc"][i], history["val_acc"][i],
                    history["loss"][i], history["val_loss"][i]
                ])
    elif constants.CLASSIFICATION == "multiclass":
        val_acc = str(round(np.mean(history['val_sparse_categorical_accuracy'])*100, 2)).replace('.', ',')
        # Get average val_acc
        guid = str(val_acc) + "_" + guid
        # Save training history to csv
        with open(f"../data/history/{constants.CLASSIFICATION}/{guid}.csv", "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["acc", "val_acc", "loss", "val_loss"])
            for i in range(len(history["sparse_categorical_accuracy"])):
                writer.writerow([
                    history["sparse_categorical_accuracy"][i], history["val_sparse_categorical_accuracy"][i],
                    history["loss"][i], history["val_loss"][i]
                ])


def count_dataset_files():
    """ Counts the number of files in the dataset"""
    file_dir = f"../data/dataset/{constants.CLASSIFICATION}/{constants.SLICE_MODE}"
    # Count png files in all subdirectories of file_dir
    png_count = 0
    for _, _, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".png"):
                png_count += 1
    print(f"[INFO] {png_count} png files in {file_dir}")


def reshuffle_test_train_kfold(fold, k):
    """ Reshuffle images in test/train directories based on fold/k folds"""
    labels = ["AD", "NL"] if constants.CLASSIFICATION == "binary" else ["AD", "MCI", "NL"]
    reset_train_test_sets(labels)
    # Collect files for each label from train and test directories
    for label in labels:
        # Get list of files in test and training directories
        train_files = os.listdir(os.path.join(constants.TRAIN_DIR, label))
        test_files = os.listdir(os.path.join(constants.TEST_DIR, label))
        # sort files
        train_files.sort()
        test_files.sort()
        # Dictionary of dirs for each file
        file_dict = {f: os.path.join(constants.TRAIN_DIR, label, f) for f in train_files}
        test_dict = {f: os.path.join(constants.TEST_DIR, label, f) for f in test_files}
        # Merge dicts
        file_dict.update(test_dict)

        all_files = list(file_dict.keys())
        # Split all files into k folds
        folds = np.array_split(all_files, k)
        # Get test fold
        test_fold = folds[fold]
        # Get training folds
        train_folds = [folds[i] for i in range(k) if i != fold]
        # Flatten list of lists
        train_fold = [item for sublist in train_folds for item in sublist]
        # Move files to correct directories
        for train_file in train_fold:
            shutil.move(file_dict[train_file], os.path.join(constants.TRAIN_DIR, label, train_file))
        for test_file in test_fold:
            shutil.move(file_dict[test_file], os.path.join(constants.TEST_DIR, label, test_file))
    count_dataset_files()


def reshuffle_test_train_sets(seed):
    """ Reshuffles the test and training sets in storage
    Args:
        seed (int): seed for random number generator
    """
    print(f"[INFO] Reshuffling test and training sets with seed {seed}")
    labels = ["AD", "NL"] if constants.CLASSIFICATION == "binary" else ["AD", "MCI", "NL"]
    reset_train_test_sets(labels)
    # Redistribute test and training sets for each label AD and NL
    for label in labels:
        # Get list of files in test and training directories
        train_files = os.listdir(os.path.join(constants.TRAIN_DIR, label))
        test_files = os.listdir(os.path.join(constants.TEST_DIR, label))
        # Sorts files consistently so they're shuffled the same way based on the seed every time
        train_files.sort()
        test_files.sort()

        # Dictionary of dirs for each file
        file_dict = {f: os.path.join(constants.TRAIN_DIR, label, f) for f in train_files}
        test_dict = {f: os.path.join(constants.TEST_DIR, label, f) for f in test_files}
        # Merge dicts
        file_dict.update(test_dict)

        all_files = list(file_dict.keys())
        random.seed(seed)
        random.shuffle(all_files)

        # Save all_files to text file
        with open(f"../data/dataset/{constants.CLASSIFICATION}/{constants.SLICE_MODE}/{seed}_files.csv", "w", encoding = 'utf-8') as f:
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
            shutil.move(file_dict[test_file], os.path.join(constants.TEST_DIR, label, test_file))

        # Move train files to label directory in train folder
        for train_file in train_files:
            shutil.move(file_dict[train_file], os.path.join(constants.TRAIN_DIR, label, train_file))
    
    count_dataset_files()


def print_generator_statistics(train_generator, val_generator, test_generator):
    """ Prints statistics about the generators
    Args:
        train_generator (generator): training generator
        val_generator (generator): validation generator
        test_generator (generator): test generator
    """
    print(f"Classes in dataset: {train_generator.class_indices}")
    train_distrib = np.bincount(train_generator.classes)
    val_distrib = np.bincount(val_generator.classes)
    test_distrib = np.bincount(test_generator.classes)
    if constants.CLASSIFICATION == "binary":
        print(f"{'Dataset':<10}{'AD':<10}{'NL':<10}")
        print(f"{'Training':<10}{train_distrib[0]:<10}{train_distrib[1]:<10} = {sum(train_distrib)} data samples")
        print(f"{'Val':<10}{val_distrib[0]:<10}{val_distrib[1]:<10} = {sum(val_distrib)} data samples")
        print(f"{'Test':<10}{test_distrib[0]:<10}{test_distrib[1]:<10} = {sum(test_distrib)} data samples")
        # Calculate totals for each label
        ad_total = train_distrib[0] + val_distrib[0] + test_distrib[0]
        nl_total = train_distrib[1] + val_distrib[1] + test_distrib[1]
        print(f"{'Total':<10}{ad_total:<10}{nl_total:<10} = {ad_total + nl_total} data samples")
    else:
        print(f"{'Dataset':<10}{'AD':<10}{'MCI':<10}{'NL':<10}")
        print(f"{'Train':<10}{train_distrib[0]:<10}{train_distrib[1]:<10}{train_distrib[2]:<10} = {sum(train_distrib)} data samples")
        print(f"{'Val':<10}{val_distrib[0]:<10}{val_distrib[1]:<10}{val_distrib[2]:<10} = {sum(val_distrib)} data samples")
        print(f"{'Test':<10}{test_distrib[0]:<10}{test_distrib[1]:<10}{test_distrib[2]:<10} = {sum(test_distrib)} data samples")
        # Calculate totals for each label
        ad_total = sum([train_distrib[0], val_distrib[0], test_distrib[0]])
        mci_total = sum([train_distrib[1], val_distrib[1], test_distrib[1]])
        nl_total = sum([train_distrib[2], val_distrib[2], test_distrib[2]])
        total_sum = ad_total + mci_total + nl_total
        print(f"{'Total':<10}{ad_total:<10}{mci_total:<10}{nl_total:<10} = {total_sum} data samples")
    return


def get_dataset_generators(seed = None, k = None, fold = None):
    """ Loads the training and testing datasets
    Args:
        constants.TRAIN_DIR (str): path to training images
        constants.TEST_DIR (str): path to testing images
    """
    if seed is not None:
        reshuffle_test_train_sets(seed)
    elif k is not None and fold is not None:
        reshuffle_test_train_kfold(fold, k)
    else:
        print("[INFO] No reshuffling of test and training sets")
    
    if constants.AUGMENTATION:
        datagen_args = dict(
            # rotation_range = 5,
            shear_range = 0.02,
            zoom_range = 0.05,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            rescale=1./255,
            validation_split = constants.VAL_SIZE
        )
    else:
        datagen_args = dict(
            rescale=1./255,
            validation_split = constants.VAL_SIZE
        )

    # Image augmentation on training set
    train_datagen = ImageDataGenerator(**datagen_args)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    # Dictates the label split mode
    class_mode = "binary" if constants.CLASSIFICATION == "binary" else "sparse"

    train_generator = train_datagen.flow_from_directory(
        constants.TRAIN_DIR,
        target_size= constants.IMAGE_SIZE,
        batch_size = constants.BATCH_SIZE,
        class_mode = class_mode,
        shuffle = True,
        interpolation='lanczos',
        seed = seed,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        constants.TRAIN_DIR,
        target_size= constants.IMAGE_SIZE,
        batch_size = constants.BATCH_SIZE,
        class_mode = class_mode,
        shuffle = True,
        interpolation='lanczos',
        seed = seed,
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        constants.TEST_DIR,
        target_size= constants.IMAGE_SIZE,
        batch_size = constants.BATCH_SIZE,
        class_mode = class_mode,
        shuffle = False
    )

    print_generator_statistics(train_generator, validation_generator, test_generator)
    return train_generator, validation_generator, test_generator


def create_model():
    """Creates an original/own Convolutional Neural Network model
    Returns:
        model (keras.Sequential): CNN model
    """
    print('[INFO] Creating model')
    size = constants.IMAGE_SIZE[0]

    model = Sequential()
    # Convolutional layer 1
    model.add(
        Conv2D(100, (3, 3),
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=l2_regularizer(l=0.01),
            kernel_constraint=unit_norm(),
            input_shape=(size, size, 3)
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.5))

    # Convolutional layer 2
    model.add(
        Conv2D(70, (3, 3),
            activation='relu',
            kernel_regularizer=l2_regularizer(l=0.01)
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))

    # Final convolutional layer
    model.add(
        Conv2D(50, (3, 3),
            activation='relu'
        )
    )
    model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    
    if constants.CLASSIFICATION == 'binary':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["acc"]
        )
        print("[INFO] Compiled!")
    elif constants.CLASSIFICATION == 'multiclass':
        model.add(Dense(3, activation='softmax'))
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"]
        )
        print("[INFO] Compiled!")

    model.summary()
    return model


def create_pretrained_model():
    """
    Train and test the pretrained DenseNet121

    Args:
        constants.TRAIN_DIR (str): Path to training directory
        constants.TEST_DIR (str): Path to testing directory
        seed (int): Random seed for reproducibility
    """
    print("[INFO]  Loading pretrained model")
    
    # Create pretrained model InceptionResNetV2
    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(constants.IMAGE_SIZE[0], constants.IMAGE_SIZE[1], 3)
    )

    # Freeze all layers in the pretrained model
    for layer in base_model.layers:
        layer.trainable = False

    # Add new layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    if constants.CLASSIFICATION == "binary":
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["acc"]
        )
    elif constants.CLASSIFICATION == "multiclass":
        predictions = Dense(3, activation='softmax')(x)

        # Create new model
        model = Model(inputs=base_model.input, outputs=predictions)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"]
        )
    print("[INFO] Compiled!")

    return model


def create_lstm_model():
    """
    Train and test the LSTM model
    """
    # TODO: Finish this function
    print("[INFO]  Creating LSTM model")


def train_model(model, train_generator, validation_generator, guid, model_type):
    """ Train own CNN model
    Args:
        model (keras.Sequential): CNN model
        train_generator (keras.preprocessing.image.ImageDataGenerator): training data generator
        validation_generator (keras.preprocessing.image.ImageDataGenerator): validation data generator
        epochs (int): Number of epochs to train for
        guid (str): Unique identifier for the model
    Returns:
        model (keras.Sequential): trained CNN model
        actual_epochs (int): Number of epochs model was actually trained for
    """
    # Initialise training callbacks
    callbacks_list = [
        ModelCheckpoint(
            filepath=f"../models/{guid}_best.h5",
            monitor="val_loss",
            mode = 'min',
            save_best_only=True,
            save_weights_only=True
        ),
        EarlyStopping(
            monitor ="val_loss",
            min_delta=0.001,
            mode='min',
            patience = 5,
            restore_best_weights = True,
            baseline = None,
            verbose = 1,
        ),
        ReduceLROnPlateau(
            monitor = "val_loss",
            mode = "auto",
            factor = 0.3,
            patience = 2,
        ),
        TensorBoard(
            log_dir=f"../logs/{guid}"
        )
    ]
    
    class_weight = {0: 0.73, 1: 0.27} if constants.CLASSIFICATION == 'binary' else {0: 0.36, 1: 0.36, 2: 0.28}

    print(f"[INFO] Training CNN model using {constants.SLICE_MODE} slices")
    history = model.fit(
        train_generator,
        class_weight=class_weight,
        steps_per_epoch=train_generator.samples // constants.BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // constants.BATCH_SIZE,
        epochs=constants.EPOCHS,
        callbacks=callbacks_list,
        verbose=1,
    )

    history = history.history
    # Save training history
    save_training_history(history, guid)

    acc_metric = "acc" if constants.CLASSIFICATION == "binary" else "sparse_categorical_accuracy"
    # Plot training history
    actual_epochs = len(history[acc_metric])
    settings = {'epochs': constants.EPOCHS, 'epochs_executed': actual_epochs, 'batch_size': constants.BATCH_SIZE}
    plot_history(history, guid, model_type, constants.CLASSIFICATION, generate_caption(settings))

    # Save trained model
    model.save(f"../models/{constants.CLASSIFICATION}/{guid}.h5")

    # Remove any model file with "best" in the name
    for file in os.listdir("../models"):
        if "best" in file:
            os.remove(f"../models/{file}")
    
    return model, actual_epochs


def train_and_test(model_type):
    """ Trains and tests a CNN on the data

    Args:
        train_data (list): training data
        test_data (list): testing data
        train_labels (list): training labels
        test_labels (list): testing labels

    Returns:
        None
    """
    guids = []
    seeds = [1, 4, 8, 9, 19]

    for fold, seed in enumerate(seeds):
        reset_random_seeds(seed)

        # Get training / testing data
        # train_gen, validation_gen, test_gen = get_dataset_generators(seed = seed)
        train_gen, validation_gen, test_gen = get_dataset_generators(fold = fold, k = len(seeds))

        guid = f"{constants.EPOCHS}_{constants.BATCH_SIZE}_{constants.SLICE_MODE}_{constants.IMAGE_SIZE[0]}_{seed}"
        guids.append(guid)
        
        # Declare blank model
        model = None

        # Get all models from models folder
        models = [f for f in os.listdir(f"../models/{constants.CLASSIFICATION}") if f.endswith(".h5")]
        # Filter models where guid in model
        models = [f for f in models if guid in f]

        # If only one model is found and we don't want to retrain a model with the same settings
        if len(models) == 1 and not constants.RETRAIN:
            # Load model from models folder
            model = models.load_model(models[0])
            print(f"[INFO] Model {models[0]} loaded")

        if model is None:
            # ! 1 Create model
            if model_type == "own_models":
                model = create_model()
            elif model_type == "transfer_models":
                model = create_pretrained_model()
            else:
                print("[ERROR] Invalid model type")
                return
            # ! 2 Train model
            model, actual_epochs = train_model(model, train_gen, validation_gen, guid, model_type)

        score = model.evaluate(test_gen, verbose=0)

        loss = f"{score[0]*100:.2f}"
        acc = f"{score[1]*100:.2f}"
        print(f'{acc:<6} {loss:<6} {seed:<6}')

        # Predict labels for test data
        test_predictions = model.predict(test_gen)

        test_label = to_categorical(test_gen.classes, len(test_gen.class_indices))

        actual = np.argmax(test_label, axis=1)

        # Round test predictions to nearest integer and select max value
        pred = np.argmax(test_predictions, axis=1)
        
        class_report = classification_report(
            actual,
            pred,
            output_dict=True
        )

        cm = confusion_matrix(actual, pred)
        plot_cm(cm, list(test_gen.class_indices.keys()), f"Confusion Matrix : {constants.CLASSIFICATION}", guid, constants.CLASSIFICATION)

        # Load training_log.csv into pandas dataframe        
        train_log = pd.read_csv("../models/training_log.csv")
        # Filter model_type 
        train_log = train_log[train_log["model_type"] == model_type]
        # ! Append results to training_log csv
        # Append guid to csv file with stats
        with open("../models/training_log.csv", "a", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # If guid exists in csv, skip writing to csv
            writer.writerow([constants.CLASSIFICATION,
                                    model_type,
                                    guid,
                                    seed,
                                    constants.EPOCHS,
                                    actual_epochs,
                                    constants.BATCH_SIZE,
                                    constants.SLICE_MODE,
                                    constants.IMAGE_SIZE[0],
                                    constants.IMAGE_SIZE[1],
                                    acc,
                                    class_report["macro avg"]["precision"],
                                    class_report["macro avg"]["recall"],
                                    class_report["macro avg"]["f1-score"],
                                    loss,
                                    constants.TEST_SIZE,
                                    constants.VAL_SIZE,
                                    False,
                                    int(time.time()),
                                    constants.AUGMENTATION
                                ])

    # Get accs from training_log.csv
    train_log = pd.read_csv("../models/training_log.csv")
    # Filter by guid
    train_log = train_log[train_log["guid"].isin(guids)]
    
    # Get statistics
    accs = train_log["acc"].tolist()
    precision = train_log["precision"].tolist()
    recall = train_log["recall"].tolist()
    f1 = train_log["f1"].tolist()
    
    # Calculate averages of statistics
    avg_acc = f"{np.array(accs).mean():.2f}"
    avg_precision = f"{np.array(precision).mean():.2f}"
    avg_recall = f"{np.array(recall).mean():.2f}"
    avg_f1 = f"{np.array(f1).mean():.2f}"

    # Calculate standard deviations of statistics
    std_acc = f"{np.array(accs).std():.2f}"
    std_precision = f"{np.array(precision).std():.2f}"
    std_recall = f"{np.array(recall).std():.2f}"
    std_f1 = f"{np.array(f1).std():.2f}"

    # ! Record average stats for the trained models
    with open("../models/training_log.csv", "a", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([constants.CLASSIFICATION,
                            model_type,
                            f"{model_type}_cv_{constants.CV}",
                            "",
                            constants.EPOCHS,
                            "",
                            constants.BATCH_SIZE,
                            constants.SLICE_MODE,
                            constants.IMAGE_SIZE[0],
                            constants.IMAGE_SIZE[1],
                            avg_acc,
                            avg_precision,
                            avg_recall,
                            avg_f1,
                            "",
                            constants.TEST_SIZE,
                            constants.VAL_SIZE,
                            True,
                            int(time.time()),
                            constants.AUGMENTATION
                        ])

    # Print statistics
    print(f"{'Type':<10} {'Metric':<10} {'Value':<10}")
    print(f"{'Average':<10}{'Accuracy':<10} {avg_acc:<10}")
    print(f"{'Average':<10}{'Precision':<10} {avg_precision:<10}")
    print(f"{'Average':<10}{'Recall':<10} {avg_recall:<10}")
    print(f"{'Average':<10}{'F1':<10} {avg_f1:<10}")
    print(f"{'STD':<10} {'Accuracy':<10} {std_acc:<10}")
    print(f"{'STD':<10} {'Precision':<10} {std_precision:<10}")
    print(f"{'STD':<10} {'Recall':<10} {std_recall:<10}")
    print(f"{'STD':<10} {'F1':<10} {std_f1:<10}")


def prepare_log_files():
    """ Prepare files for training and testing """
    if not os.path.exists(f"../data/history/{constants.CLASSIFICATION}"):
        os.makedirs(f"../data/history/{constants.CLASSIFICATION}")

    # If training_log.csv exists, delete it
    if not os.path.exists("../models/training_log.csv"):
        print("[INFO]  Creating training_log.csv")
        training_log_headers = "classification,model_type,guid,seed,epochs,actual_epochs,batch_size,constants.SLICE_MODE,height,width,acc,precision,recall,f1,loss,constants.TEST_SIZE,constants.VAL_SIZE,individual,time"
        with open("../models/training_log.csv", "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(training_log_headers.split(","))


def main(image_size = None, slice_mode = None, batch_size = None, epochs = None, test_size = None, val_size = None, cv = None):
    """Image data classification"""
    prepare_log_files()
    print("[INFO] Image data classification")

    # ! Train and test own CNN model
    train_and_test("transfer_models")
    # ! Train and test pre-trained CNN model (ResNET50)
    # train_and_test_pretrained(constants.TRAIN_DIR, constants.TEST_DIR, 42)
    # ! Train and test model with self-attention layer
    # train_and_test_attention(train_data, test_data, train_labels, test_labels)s


if __name__ == "__main__":
    main()