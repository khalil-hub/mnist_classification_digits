from src.eval_utils import  save_wrong_predictions, evaluate_models, plot_precision_recall_curve, ROC_AUC_curve
from tensorflow.keras.models import load_model
from src.data_loader import load_mnist_data, pre_process_data
from src.data_augmentation import add_noise_and_blur
import numpy as np
model_path='models/non_augmented/mnist_cnn_adam_10.h5'
for distortion in (0, 1):
    if distortion==0:
        x_train, y_train, x_test, y_test=load_mnist_data()
        x_train, y_train, x_test, y_test=pre_process_data(x_train, y_train, x_test, y_test)

        evaluate_models(x_test, y_test, models_dir='models', save_dir='predictions/non_distorted')

        best_model=load_model(model_path)
        y_true=np.argmax(y_test, axis=1)
        y_pred_probs=best_model.predict(x_test)
        y_pred=np.argmax(y_pred_probs, axis=1)

        plot_precision_recall_curve(y_test, y_pred_probs)
        ROC_AUC_curve(y_test, y_pred_probs)
        save_wrong_predictions(x_test, y_true, y_pred, save_dir="predictions/wrong_preds/non_distorted", max_to_save=None)
    else:
        x_train, y_train, x_test, y_test=load_mnist_data()
        x_train, y_train, x_test, y_test=pre_process_data(x_train, y_train, x_test, y_test)
        x_test=add_noise_and_blur(x_test)

        evaluate_models(x_test, y_test, models_dir='models', save_dir='predictions/distorted')

        best_model=load_model(model_path)
        y_true=np.argmax(y_test, axis=1)
        y_pred_probs=best_model.predict(x_test)
        y_pred=np.argmax(y_pred_probs, axis=1)

        plot_precision_recall_curve(y_test, y_pred_probs)
        ROC_AUC_curve(y_test, y_pred_probs)
        save_wrong_predictions(x_test, y_true, y_pred, save_dir="predictions/wrong_preds/distorted", max_to_save=None)