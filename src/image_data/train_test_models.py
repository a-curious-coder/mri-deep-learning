""" Process, transform, and load image data """
import csv
import os
import random
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import callbacks, layers, models, optimizers, utils
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2 as l2_regularizer
from plot import plot_history, generate_caption


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
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def reshuffle_test_train_sets(seed):
    """ Reshuffles the test and training sets in storage
    Args:
        seed (int): seed for random number generator
    """
    print(f"[INFO] Reshuffling test and training sets with seed {seed}")
    train_dir = f"../data/dataset/{SLICE_MODE}_{IMAGE_SIZE[0]}/train"
    test_dir = f"../data/dataset/{SLICE_MODE}_{IMAGE_SIZE[0]}/test"
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
        with open(f"../data/dataset/{SLICE_MODE}_{IMAGE_SIZE[0]}/{seed}_files.csv", "w", encoding = 'utf-8') as f:
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
    
    file_dir = f"../data/dataset/{SLICE_MODE}_{IMAGE_SIZE[0]}"
    # Count png files in all subdirectories of file_dir
    png_count = 0
    for _, _, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".png"):
                png_count += 1
    print(f"[INFO] {png_count} png files in {file_dir}")
    print(f"[INFO] Reshuffling complete (seed : {seed}) ")


def get_dataset_generators(train_dir, test_dir, seed):
    """ Loads the training and testing datasets
    Args:
        train_dir (str): path to training images
        test_dir (str): path to testing images
    """
    reshuffle_test_train_sets(seed)

    datagen_args = dict(
        rotation_range = 5,
        shear_range = 0.02,
        zoom_range = 0.05,
        samplewise_center = True,
        samplewise_std_normalization = True,
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
        target_size= IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = 'binary',
        shuffle = True,
        seed = seed,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size= IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = 'binary',
        shuffle = True,
        seed = seed,
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size= IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        class_mode = 'binary',
        shuffle = False,
        seed = seed
    )

    print(f"Classes in dataset: {train_generator.class_indices}")
    # Print distribution of classes in training set
    print(f"Number of samples per class in training set: {int(train_generator.samples / len(train_generator.class_indices))}")


    # class_mapping = {v:k for k,v in train_generator.class_indices.items()}
    # show_grid(x, 1, 2,label_list=y, show_labels=True,figsize=(20,10),class_mapping = class_mapping)

    return train_generator, validation_generator, test_generator


def save_training_history(history, guid):
    """ Saves the training history to a csv file
    Args:
        history (dict): training history
        guid (str): guid for the model
    """
    history = history.history
    val_acc = str(round(np.mean(history['val_acc'])*100, 2)).replace('.', ',')
    # Get average val_acc
    guid = str(val_acc) + "_" + guid
    # Save training history to csv
    with open(f"../data/history/{guid}.csv", "w", encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["acc", "val_acc", "loss", "val_loss"])
        for i in range(len(history["acc"])):
            writer.writerow([
                history["acc"][i], history["val_acc"][i],
                history["loss"][i], history["val_loss"][i]
            ])


def create_model():
    """Creates an original/own Convolutional Neural Network model
    Returns:
        model (keras.models.Sequential): CNN model
    """
    print('[INFO] Creating model')
    size = IMAGE_SIZE[0]

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
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Dropout(0.5))

    # Convolutional layer 2
    model.add(
        layers.Conv2D(70, (3, 3),
            activation='relu',
            kernel_regularizer=l2_regularizer(l=0.01)
        )
    )
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Dropout(0.3))

    # Final convolutional layer
    model.add(
        layers.Conv2D(50, (3, 3),
            activation='relu'
        )
    )
    model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["acc"]
    )

    return model


def train_model(model, train_generator, validation_generator, guid):
    """ Train own CNN model
    Args:
        model (keras.models.Sequential): CNN model
        train_generator (keras.preprocessing.image.ImageDataGenerator): training data generator
        validation_generator (keras.preprocessing.image.ImageDataGenerator): validation data generator
        epochs (int): Number of epochs to train for
        guid (str): Unique identifier for the model
    Returns:
        model (keras.models.Sequential): trained CNN model
        actual_epochs (int): Number of epochs model was actually trained for
    """
    # Initialise training callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=f"../models/{guid}_best.h5",
            monitor="val_loss",
            mode = 'min',
            save_best_only=True,
            save_weights_only=True
        ),
        callbacks.EarlyStopping(
            monitor ="val_loss",
            min_delta=0.001,
            mode='min',
            patience = 5,
            restore_best_weights = True,
            baseline = None,
            verbose = 1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor = "val_loss",
            mode = "auto",
            factor = 0.3,
            patience = 2,
        ),
        callbacks.TensorBoard(
            log_dir=f"../logs/{guid}"
        )
    ]

    print(f"[INFO] Training CNN model using {SLICE_MODE} slices")

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1,
    )

    # Save training history
    save_training_history(history, guid)

    # Plot training history
    actual_epochs = len(history["acc"])
    settings = {'epochs': EPOCHS, 'epochs_executed': actual_epochs, 'batch_size': BATCH_SIZE}
    plot_history(history, guid, "own_models", generate_caption(settings))

    # Save trained model
    model.save(f"../models/{guid}.h5")

    # Remove any model file with "best" in the name
    for file in os.listdir(f"../models"):
        if "best" in file:
            os.remove(f"../models/{file}")
    
    return model, actual_epochs


def train_and_test(train_dir, test_dir):
    """ Trains and tests a CNN on the data

    Args:
        train_data (list): training data
        test_data (list): testing data
        train_labels (list): training labels
        test_labels (list): testing labels

    Returns:
        None
    """
    cv = 5
    
    height = IMAGE_SIZE[0]
    width = IMAGE_SIZE[1]

    # Compute the mean and the variance of the training data for normalization.
    accs = []
    f1 = []
    precision = []
    recall = []

    # Ensure the same random is generated each time
    random.seed(42)
    seeds = random.sample(range(1, 20), cv)
    seeds.sort()
    guids = []
    for seed in seeds:
        reset_random_seeds(seed)
        # get the training and testing data
        train_gen, validation_gen, test_gen = get_dataset_generators(train_dir, test_dir, seed)

        guid = f"{EPOCHS}_{BATCH_SIZE}_{SLICE_MODE}_{IMAGE_SIZE[0]}_{seed}"
        guids.append(guid)
        # Load training_log.csv into pandas dataframe
        train_log = pd.read_csv("../models/training_log.csv")
        
        # Declare blank model
        model = None

        # Get all models from models folder
        models = [f for f in os.listdir("../models") if f.endswith(".h5")]
        # Filter models where guid in model
        models = [f for f in models if guid in f]
        if len(models) == 1 and not RETRAIN:
            # Load model from models folder
            model = models.load_model(models[0])
            print(f"[INFO] Model {models[0]} loaded")

        if model is None:
            # ! 1 Create model
            model = create_model()
            # ! 2 Train model
            model, actual_epochs = train_model(model, train_gen, validation_gen, guid)

        score = model.evaluate(test_gen, verbose=0)

        loss = f"{score[0]*100:.2f}"
        acc = f"{score[1]*100:.2f}"
        print(f'{acc:<6} {loss:<6} {seed:<6}')

        # Predict labels for test data
        test_predictions = model.predict(test_gen)
        test_label = utils.to_categorical(test_gen.classes, 2)

        true_label = np.argmax(test_label, axis=1)

        # Round test predictions to nearest integer
        predicted_label = np.round(test_predictions)

        class_report = classification_report(
            true_label,
            predicted_label,
            output_dict=True,
            zero_division=0
        )
        
        # ! Append results to training_log csv
        # if guid is not in guid column, write to training_log.csv
        if guid not in train_log["guid"].values:
            # Append guid to csv file with stats
            with open("../models/training_log.csv", "a", encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                # If guid exists in csv, skip writing to csv
                writer.writerow([guid,
                                    seed,
                                    EPOCHS,
                                    actual_epochs,
                                    BATCH_SIZE,
                                    SLICE_MODE,
                                    height,
                                    width,
                                    acc,
                                    class_report["macro avg"]["precision"],
                                    class_report["macro avg"]["recall"],
                                    class_report["macro avg"]["f1-score"],
                                    loss,
                                    TEST_SIZE,
                                    VAL_SIZE,
                                    False
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
            writer.writerow([f"own_model_cv_{cv}",
                            "",
                            EPOCHS,
                            "",
                            BATCH_SIZE,
                            SLICE_MODE,
                            height,
                            width,
                            avg_acc,
                            avg_precision,
                            avg_recall,
                            avg_f1,
                            "",
                            TEST_SIZE,
                            VAL_SIZE,
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


def create_pretrained_model(train_dir, test_dir, seed):
    """
    Train and test the pretrained DenseNet121

    Args:
        train_dir (str): Path to training directory
        test_dir (str): Path to testing directory
        seed (int): Random seed for reproducibility
    """
    print("[INFO]  Loading DenseNet121")
    # Load pretrained model
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    x = base_model.output

    x = GlobalAveragePooling2D()(x)
    predictions = layers.Dense(1, activation="sigmoid")(x)
    # Stitch model together
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=["acc"]
    )

    return model


def create_lstm_model():
    """
    Train and test the LSTM model
    """
    # TODO: Finish this function
    print("[INFO]  Training and testing LSTM model")
    # Load data
    train_data, test_data, train_labels, test_labels = load_data()
    # Create LSTM model and train
    pass


def prepare_log_files():
    """ Prepare files for training and testing """
    # If training_log.csv exists, delete it
    if not os.path.exists("../models/training_log.csv"):
        print("[INFO]  Creating training_log.csv")
        training_log_headers = "guid,seed,epochs,actual_epochs,batch_size,SLICE_MODE,height,width,acc,precision,recall,f1,loss,TEST_SIZE,VAL_SIZE,individual"
        with open("../models/training_log.csv", "w", encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(training_log_headers.split(","))


def image_data_classification(im, sm, ts = 0.1, vs = 0.2):
    """Image data classification"""
    global IMAGE_SIZE
    global SLICE_MODE
    global TEST_SIZE
    global VAL_SIZE
    global EPOCHS
    global BATCH_SIZE
    global RETRAIN

    IMAGE_SIZE = im
    SLICE_MODE = sm
    TEST_SIZE = ts
    VAL_SIZE = vs
    EPOCHS = 50
    BATCH_SIZE = 32
    RETRAIN = False

    prepare_log_files()
    print("")
    print("[INFO] Image data classification")

    # Train directory
    train_dir = f'../data/dataset/{SLICE_MODE}_{IMAGE_SIZE[0]}/train'
    test_dir = f'../data/dataset/{SLICE_MODE}_{IMAGE_SIZE[0]}/test'

    # ! Train and test own CNN model
    train_and_test(train_dir, test_dir)
    # ! Train and test pre-trained CNN model (ResNET50)
    # train_and_test_pretrained(train_dir, test_dir, 42)
    # ! Train and test model with self-attention layer
    # train_and_test_attention(train_data, test_data, train_labels, test_labels)