"""CNN-based image classification on SageMaker with TensorFlow and Keras
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

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
CHANNELS = 3
INPUT_SHAPE = (32, 32, CHANNELS)
NUM_LABELS = 10

def parse_args():
    """Acquire hyperparameters and directory locations passed by SageMaker"""
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    return parser.parse_known_args()

def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)
    
    image = tf.image.decode_png(example['image'], channels=CHANNELS)
    image = tf.cast(image, tf.float32) / 255.0
    
    class_label = tf.cast(example['class'], tf.int32)
    
    return image, class_label


def get_batched_dataset(filenames):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) 
    dataset = dataset.prefetch(AUTO) #

    return dataset
  
def load_data(args):
    labels = sorted(os.listdir(args.train))
    n_labels = len(labels)
    training_filenames = tf.io.gfile.glob(args.train + '/*.tfrec')
    print("training dataset:", training_filenames)
    training_dataset = get_batched_dataset(training_filenames)
    validation_filenames = tf.io.gfile.glob(args.test + '/*.tfrec')
    validation_dataset = get_batched_dataset(validation_filenames)
    print("validation dataset:", validation_filenames)
    return training_dataset, validation_dataset
    
    
def build_model(input_shape, n_labels):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_labels, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )    
    return model
    
    
# Training script:
if __name__ == "__main__":
    # Load arguments from CLI / environment variables:
    args, _ = parse_args()
    print(args)

    print("Loading dataset...")
    training_dataset, validation_dataset = load_data(args)

    print("Building a model...")
    model = build_model(INPUT_SHAPE, NUM_LABELS)

    # Fit the Keras model
    print("Fitting the data...")
    model.fit(
        training_dataset,
        epochs=args.epochs,
        validation_data=validation_dataset,
        verbose=2
    )

    # Evaluate model quality and log metrics
    score = model.evaluate(validation_dataset)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")

    # Save outputs (trained model) to specified folder?
    model.save(os.path.join(args.model_dir, "model/1"))
    