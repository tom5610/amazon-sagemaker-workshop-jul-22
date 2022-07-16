"""CNN-based image classification on SageMaker with TensorFlow and Keras

REFERENCE SOLUTION IMPLEMENTATION

(Complete me with help from Local Notebook.ipynb, and the NLP example's src/main.py!)
"""

# Dependencies:
import argparse
import os

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

def parse_args():
    """Acquire hyperparameters and directory locations passed by SageMaker"""
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    return parser.parse_known_args()

# TODO: Other function definitions, if you'd like to break up your code into functions?

def load_data(args):
    labels = sorted(os.listdir(args.train))
    n_labels = len(labels)

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    print("Loading label ", end="")
    for ix_label in range(n_labels):
        label_str = labels[ix_label]
        print(f"{label_str}...", end="")
        trainfiles = filter(
            lambda s: s.endswith(".png"),
            os.listdir(os.path.join(args.train, label_str))
        )    

        for filename in trainfiles:
            # Can't just use tf.keras.preprocessing.image.load_img(), because it doesn't close its file
            # handles! So get "Too many open files" error... Grr
            with open(os.path.join(args.train, label_str, filename), "rb") as imgfile:
                x_train.append(
                    # Squeeze (drop) that extra channel dimension, to be consistent with prev format:
                    np.squeeze(tf.keras.preprocessing.image.img_to_array(
                        Image.open(imgfile)
                    ))
                )
                y_train.append(ix_label)

        # Repeat for test data:
        testfiles = filter(
            lambda s: s.endswith(".png"),
            os.listdir(os.path.join(args.test, label_str))
        )

        for filename in testfiles:
            with open(os.path.join(args.test, label_str, filename), "rb") as imgfile:
                x_test.append(
                    np.squeeze(tf.keras.preprocessing.image.img_to_array(
                        Image.open(imgfile)
                    ))
                )
                y_test.append(ix_label)
    print()


    print("Shuffling trainset...")
    train_shuffled = [(x_train[ix], y_train[ix]) for ix in range(len(y_train))]
    np.random.shuffle(train_shuffled)

    x_train = np.array([datum[0] for datum in train_shuffled])
    y_train = np.array([datum[1] for datum in train_shuffled])
    train_shuffled = None

    print("Shuffling testset...")
    test_shuffled = [(x_test[ix], y_test[ix]) for ix in range(len(y_test))]
    np.random.shuffle(test_shuffled)

    x_test = np.array([datum[0] for datum in test_shuffled])
    y_test = np.array([datum[1] for datum in test_shuffled])
    test_shuffled = None

    print("Done!")

    print(f"training data set shape: {x_train.shape}")
    print(K.image_data_format())

    if 'channels_last' != K.image_data_format():
        print("use 'channels_last' data format...")
        K.set_image_data_format('channels_last')

    # convert dataset matrix to be float32 and normalize them by 255
    # x_train = x_train.astype("float32")
    # x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0

    input_shape = x_train.shape[1:]

    print("x_train shape:", x_train.shape)
    print("input_shape:", input_shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, n_labels)
    y_test = tf.keras.utils.to_categorical(y_test, n_labels)

    print("n_labels:", n_labels)
    print("y_train shape:", y_train.shape)

    return x_train, y_train, x_test, y_test, input_shape, n_labels

def build_model(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(n_labels, activation='softmax'))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )

    return model

# Training script:
if __name__ == "__main__":
    # Load arguments from CLI / environment variables:
    args, _ = parse_args()
    print(args)

    print("Loading dataset...")
    x_train, y_train, x_test, y_test, input_shape, n_labels = load_data(args)

    print("Building a model...")
    model = build_model(input_shape, n_labels)

    # Fit the Keras model
    print("Fitting the data...")
    model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        verbose=2, 
        validation_data=(x_test, y_test)
    )

    # Evaluate model quality and log metrics
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")

    # Save outputs (trained model) to specified folder?
    model.save(os.path.join(args.model_dir, "model/1"))
