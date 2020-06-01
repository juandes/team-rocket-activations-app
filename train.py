import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.layers import (Conv2D, Dense, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 64
EPOCHS = 20
IMG_HEIGHT = 150
IMG_WIDTH = 150

ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

log_dir = "/tmp/tensorboard/{}".format(ts)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)

image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)


train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory='data_temp/',
                                                     shuffle=True,
                                                     target_size=(
                                                               IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary',
                                                     subset='training')

val_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                   directory='data_temp/',
                                                   shuffle=True,
                                                   target_size=(
                                                             IMG_HEIGHT, IMG_WIDTH),
                                                   class_mode='binary',
                                                   subset='validation')

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
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    callbacks=[tensorboard_callback]
)

# Visualize activations
output_layers = [layer.output for layer in model.layers[:6]]
activation_model = tf.keras.Model(inputs=model.input, outputs=output_layers)
activations = activation_model.predict(val_data_gen)

# each index of activations represents the activation tensor at that particular layer
first_layer_activations = activations[0]
plt.matshow(first_layer_activations[1, :, :, 4])
plt.show()

second_layer_activations = activations[1]
plt.matshow(second_layer_activations[1, :, :, 4], cmap='viridis')
plt.show()

third_layer_activations = activations[2]
plt.matshow(third_layer_activations[1, :, :, 4], cmap='viridis')
plt.show()

fourth_layer_activations = activations[3]
plt.matshow(fourth_layer_activations[1, :, :, 4], cmap='viridis')
plt.show()

fifth_layer_activations = activations[4]
plt.matshow(fifth_layer_activations[1, :, :, 4], cmap='viridis')
plt.show()

sixth_layer_activations = activations[5]
plt.matshow(sixth_layer_activations[1, :, :, 4], cmap='viridis')
plt.show()

print(model.summary())
model.save('model/{}'.format(ts))
