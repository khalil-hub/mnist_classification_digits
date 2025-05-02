import numpy as np
import os
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_curve, roc_curve, auc

def evaluate_models(x_test, y_test, models_dir, save_dir=None):

    os.makedirs(save_dir, exist_ok=True)
    summary = []

    model_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.h5'):
                model_files.append(os.path.join(root, file))

    y_true = np.argmax(y_test, axis=1)

    for model_path in model_files:
        model = load_model(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]  
        print(f"Loaded model: {model_name}")

        y_pred_prob = model.predict(x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Save predictions and confusion matrix
        np.save(os.path.join(save_dir, f'y_pred_{model_name}.npy'), y_pred)
        np.save(os.path.join(save_dir, f'confusion_matrix_{model_name}.npy'), confusion_matrix(y_true, y_pred))
        
        # Save classification report
        with open(os.path.join(save_dir, f'metrics_{model_name}.json'), "w") as f:
            json.dump(classification_report(y_true, y_pred, output_dict=True), f, indent=4)

        # Add to summary
        report = classification_report(y_true, y_pred, output_dict=True)
        summary.append({
            "model": model_name,
            "accuracy": np.mean(y_pred == y_true),
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro": report["macro avg"]["recall"],
            "F1_score": report["macro avg"]["f1-score"]
        })

    # Save summary
    pd.DataFrame(summary).to_csv(os.path.join(save_dir, "summary.csv"), index=False)
    print(" Evaluation complete. Results saved to predictions/")

def ROC_AUC_curve(y_test, y_pred_prob):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(10):  # For each class
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(12,8))
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label=f"Digit {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line (random)
    plt.title('ROC Curves for Each Class')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_probs):
    for i in range (10):
        precision, recall, thresholds=precision_recall_curve(
            y_test[:, i],
            y_pred_probs[:, i]
        )
        plt.plot(recall, precision, marker='.', label=f'Class {i}')
        
    plt.title(f'precision-recall curve for all classes')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid(True)
    plt.legend()
    plt.show()

def save_wrong_predictions (x_test, y_true, y_pred, save_dir="predictions/wrong_preds", max_to_save=None):
    """save wrong predictions as images for error analysis """
    wrong_indices=np.where(y_pred!=y_true)[0]
    os.makedirs(save_dir, exist_ok=True)
    for idx, i in enumerate(wrong_indices[:max_to_save]):
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_true[i]}, Predicted: {y_pred[i]}")
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"wrong_{idx}_true_{y_true[i]}_pred_{y_pred[i]}.png"))
        plt.close()

    print(f"Saved {len(wrong_indices[:max_to_save])} wrong predictions to {save_dir}")
