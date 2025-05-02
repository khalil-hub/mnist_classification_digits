from src.data_loader import load_mnist_data, pre_process_data
from src.train_utils import get_tensorboard_callback
from src.model_builder import build_model, compile_model, train_model, save_model, train_with_augmentation
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    combinations = [('adam', 10), ('adam', 5), ('sgd', 5), ('sgd', 10)]

    x_train, y_train, x_test, y_test = load_mnist_data()
    x_train, y_train, x_test, y_test = pre_process_data(x_train, y_train, x_test, y_test)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

    for optimizer, epochs in combinations:
        for augmentation in [False, True]:  # Loop over normal + augmented training
            # Create a new fresh model every time
            model = build_model((28, 28, 1))
            model = compile_model(model, optimizer=optimizer)

            # Choose training method
            if augmentation:
                train_with_augmentation(
                    model, 
                    x_train, y_train, x_val, y_val, 
                    epochs=epochs, 
                    batch_size=128, 
                    callbacks=[get_tensorboard_callback(f'logs/fit/{optimizer}_{epochs}_augmented')]
                )
                save_model(model, path=f"models/augmented/mnist_cnn_{optimizer}_{epochs}_augmented.h5")
            else:
                model = train_model(
                    model, 
                    x_train, y_train, 
                    epochs=epochs, 
                    callbacks=[get_tensorboard_callback(f'logs/fit/{optimizer}_{epochs}')]
                )
                save_model(model, path=f"models/non_augmented/mnist_cnn_{optimizer}_{epochs}.h5")
