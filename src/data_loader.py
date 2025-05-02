import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_mnist_data():
    # Load raw data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test

def pre_process_data(x_train, y_train, x_test, y_test):
    # Normalize pixel values to [0, 1] and cast data type
    x_train=x_train.astype('float32')/255.0
    x_test=x_test.astype('float32')/255.0

    #Reshaping of images
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    #One hot encoding
    y_train = to_categorical(y_train, num_classes=10)
    y_test=to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test

def pre_process_real_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
    img = cv2.resize(img, (28, 28))                    # Resize to 28x28
    img = img.astype('float32') / 255.0                # Normalize to [0,1]
    img = np.expand_dims(img, axis=-1)                 # Add channel dimension (28,28,1)
    img = np.expand_dims(img, axis=0)                  # Add batch dimension (1,28,28,1)
    return img
