import numpy as np
import os
# using OpenCV to load images
import cv2 as cv
import matplotlib.pyplot as plt

# downscale the images
SPATIAL_DIM = 64


def load_images(dirpath: str, size: int):
    image_list = np.empty(shape=(size, SPATIAL_DIM, SPATIAL_DIM, 3))
    images = os.listdir(f'{dirpath}')
    for i in range(size):
        image_name = images[i]
        path = f'{dirpath}/{image_name}'

        # Load and Normalize the image
        # img = normalize(load(path))
        img = cv.imread(path)
        # Skip if there were image loading errors
        if img is None:
            continue
        img = cv.resize(img, (SPATIAL_DIM, SPATIAL_DIM))
        # reverses the order of elements based on the axis
        img = np.flip(img, axis=2)
        img = img.astype(np.float32)/ 127.5 - 1.0
        image_list[i] = img

    return image_list


def generate_and_save_images(model, epoch, test_input):
    """
    Generate and save images
    """

    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[0] * 0.5 + 0.5, cmap='binary')
        plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


## Source https://www.tensorflow.org/tutorials/generative/dcgan#create_a_gif
def show(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image * 0.5 + 0.5, cmap="binary")
        plt.axis("off")
    plt.savefig('generated_images.png')
    plt.show()


if __name__ == '__main__':
    # dirpath = 'all/Train'
    dirpath = '/Users/trishathakur/Downloads/LLD_favicons_clean_png'
    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)
    images = load_images(dirpath, 10)
    print(images.shape)
    plt.imshow(images[0] * 0.5 + 0.5)
    plt.show()

