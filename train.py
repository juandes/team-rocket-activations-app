import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 64
EPOCHS = 25
IMG_HEIGHT = 150
IMG_WIDTH = 150


def create_data_generators():
    """Create the data generators used for training.

    Returns:
        tf.keras.preprocessing.image.ImageDataGenerator: Two ImageDataGenerator;
        the training and validation dataset.
    """
    image_gen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )

    # James is label 0
    train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory='data/train/',
                                                         shuffle=True,
                                                         target_size=(
                                                             IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='categorical',
                                                         subset='training')

    val_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                       directory='data/train/',
                                                       shuffle=True,
                                                       target_size=(
                                                           IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='categorical',
                                                       subset='validation')

    return train_data_gen, val_data_gen


def train(train_data_gen, val_data_gen):
    # Set up TensorBoard directory and callback
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "/tmp/tensorboard/{}".format(ts)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.fit(
        train_data_gen,
        steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_data_gen,
        callbacks=[tensorboard_callback]
    )

    print(model.summary())
    model.save('models/{}'.format(ts))


if __name__ == "__main__":
    train_data_gen, val_data_gen = create_data_generators()
    train(train_data_gen, val_data_gen)
