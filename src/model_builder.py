from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from src.data_augmentation import get_data_generator
import os
def build_model(input_shape):
    model = Sequential()

    # First Convolution Block
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Optional

    # Second Convolution Block
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Optional

    # Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Optional
    model.add(Dense(10, activation='softmax'))  # 10 classes for digits 0-9
    return model

def compile_model(model, optimizer='adam'):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=None, callbacks=None):
    model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.1, callbacks=callbacks)
    return model

def train_with_augmentation(model, x_train, y_train, x_val, y_val, epochs, batch_size, callbacks):

    datagen = get_data_generator()
    datagen.fit(x_train)

    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=callbacks
    )

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)