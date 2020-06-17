import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

model = tf.keras.models.load_model('models/20200602-142312')


BATCH_SIZE = 64
IMG_HEIGHT = 150
IMG_WIDTH = 150

IMAGE_NUMBER = 0
FILTER_NUMBER = 0

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
                                                     directory='data/train/',
                                                     shuffle=True,
                                                     target_size=(
                                                               IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='categorical',
                                                     subset='training')

# Visualize activations
output_layers = [layer.output for layer in model.layers[:6]]
activation_model = tf.keras.Model(inputs=model.input, outputs=output_layers)
activations = activation_model.predict(train_data_gen)

# each index of activations represents the activation tensor at that particular layer
first_layer_activations = activations[0]
plt.matshow(first_layer_activations[IMAGE_NUMBER, :, :, FILTER_NUMBER])
plt.show()

second_layer_activations = activations[1]
plt.matshow(
    second_layer_activations[IMAGE_NUMBER, :, :, FILTER_NUMBER], cmap='viridis')
plt.show()

third_layer_activations = activations[2]
plt.matshow(third_layer_activations[IMAGE_NUMBER,
                                    :, :, FILTER_NUMBER], cmap='viridis')
plt.show()

fourth_layer_activations = activations[3]
plt.matshow(
    fourth_layer_activations[IMAGE_NUMBER, :, :, FILTER_NUMBER], cmap='viridis')
plt.show()

fifth_layer_activations = activations[4]
plt.matshow(fifth_layer_activations[IMAGE_NUMBER,
                                    :, :, FILTER_NUMBER], cmap='viridis')
plt.show()

sixth_layer_activations = activations[5]
plt.matshow(sixth_layer_activations[IMAGE_NUMBER,
                                    :, :, FILTER_NUMBER], cmap='viridis')
plt.show()
