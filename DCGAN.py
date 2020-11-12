import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Reshape, \
    BatchNormalization, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from IPython import display
from LogoDCGAN.images import load_images, generate_and_save_images, show

# downscale the images
IMAGE_DIM = 64
# size of noise vector
NOISE_DIM = 100
#  batch_size
BATCH_SIZE = 50
# data path
DIRPATH = '/Users/trishathakur/Downloads/LLD_favicons_clean_png'
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)
# DIRPATH = 'all/Train'


def build_discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), (2, 2), padding='same',
                     input_shape=[IMAGE_DIM, IMAGE_DIM, 3]))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (5, 5), (2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_generator_model():
    model = Sequential()
    model.add(Dense(16 * 16 * 128, input_shape=[NOISE_DIM]))
    model.add(Reshape([16, 16, 128]))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (5, 5), (2, 2), padding='same',
                              activation='selu'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(3, (5, 5), (2, 2), padding='same',
                              activation='tanh'))
    return model


def build_dcgan():
    discriminator = build_discriminator_model()
    discriminator.compile(loss='binary_crossentropy', optimizer='rmsprop')
    discriminator.trainable = False
    generator = build_generator_model()
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return gan


def train_dcgan(gan, dataset, batch_size, num_features, seed, epochs=5):
    generator, discriminator = gan.layers
    for epoch in tqdm(range(epochs)):
        print("Epochs {}/{}".format(epoch + 1, epochs))
        for X_batch in dataset:
            noise = tf.random.normal(shape=[batch_size, num_features])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
            display.clear_output(wait=True)
            # generate_and_save_images(generator, epoch + 1, seed)
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


if __name__ == '__main__':
    print('Tensorflow version:', tf.__version__)

    gan = build_dcgan()
    generator, discriminator = gan.layers
    noise = tf.random.normal(shape=[1, NOISE_DIM])
    generated_image = generator(noise, training=False)
    print(generated_image.shape)
    # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
    # plt.show()

    decision = discriminator(generated_image)
    print(decision)

    seed = tf.random.normal(shape=[BATCH_SIZE, 100])
    x_train = load_images(DIRPATH, 1000)
    x_train = x_train.reshape(-1, 64, 64, 3) * 2. -1.
    x_train = tf.cast(x_train, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(1)

    train_dcgan(gan, dataset, BATCH_SIZE, NOISE_DIM, seed, epochs=100)

    noise = tf.random.normal(shape=[BATCH_SIZE, NOISE_DIM])
    generated_images = generator(noise)
    show(generated_images, 8)
