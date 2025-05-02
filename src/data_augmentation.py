from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

def get_data_generator():
    return ImageDataGenerator(
        rotation_range=10,        # Rotate images by up to 10 degrees
        zoom_range=0.1,           # Zoom in/out by 10%
        width_shift_range=0.1,    # Shift horizontally by 10%
        height_shift_range=0.1,   # Shift vertically by 10%
        brightness_range=[0.8,1.2], # Slight brightness variation
    )

def add_noise_and_blur(x_test):
    x_test_noisy = []
    for img in x_test:
        img = img.squeeze()  # Remove channel dimension (28,28)

        # Add Gaussian noise
        noise = np.random.normal(0, 0.2, img.shape)
        noisy_img = img + noise

        # Clip to [0,1]
        noisy_img = np.clip(noisy_img, 0., 1.)

        # Slight blur (using OpenCV or scipy)
        blurred_img = cv2.GaussianBlur(noisy_img, (3,3), 0)

        # Expand dims back to (28,28,1)
        blurred_img = np.expand_dims(blurred_img, axis=-1)
        x_test_noisy.append(blurred_img)

    return np.array(x_test_noisy)
