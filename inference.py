from tensorflow.keras.models import load_model
from src.data_loader import pre_process_real_image
import numpy as np
import os
path='data'
model=load_model('models/non_augmented/mnist_cnn_adam_10.h5')
pics = [f for f in os.listdir(path) if f.startswith('IMG_')]
for p in pics:
    x_real=pre_process_real_image(os.path.join(path, p))
    prediction=model.predict(x_real)
    pred_class=np.argmax(prediction)
    print(f"predicted digit :{pred_class} for {p}")