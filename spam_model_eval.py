import numpy as np
import os
import torch
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix,
                             roc_curve,
                             auc)


def evaluate(y_true, y_pred, run_name="spam_classifier"):
    """
    Evaluate model performance and optionally log metrics to Weights & Biases if WANDB_PROJECT is set

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        run_name: Name for the wandb run (only used if wandb logging is enabled)
    """
    use_wandb = bool(os.getenv('WANDB_PROJECT'))
    if use_wandb:
        import wandb
        import seaborn as sns
        import matplotlib.pyplot as plt
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
        else:
            gpu_name = "CPU"
            gpu_memory = 0

        wandb.init(project=os.getenv('WANDB_PROJECT'), name=run_name, config={
            "gpu_name": gpu_name,
            "gpu_memory_gb": f"{gpu_memory:.2f}",
            "dataset_size": len(y_true)
        })

    labels = ["ham", "spam"]
    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(x, -1)

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)
    if use_wandb:
        wandb.log({"overall_accuracy": accuracy})
    print(f'Accuracy: {accuracy:.3f}')

    # Calculate per-label accuracy
    unique_labels = set(y_true_mapped)
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        if use_wandb:
            wandb.log({f"{labels[label]}_accuracy": label_accuracy})
        print(f'Accuracy for label {labels[label]}: {label_accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped,
                                         target_names=labels,
                                         labels=list(range(len(labels))),
                                         output_dict=True)
    if use_wandb:
        wandb.log({
            "precision": {label: metrics['precision'] for label, metrics in class_report.items() if label in labels},
            "recall": {label: metrics['recall'] for label, metrics in class_report.items() if label in labels},
            "f1_score": {label: metrics['f1-score'] for label, metrics in class_report.items() if label in labels}
        })
    print('\nClassification Report:')
    print(classification_report(y_true=y_true_mapped, y_pred=y_pred_mapped,
                                target_names=labels,
                                labels=list(range(len(labels)))))

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true_mapped, y_pred=y_pred_mapped,
                                   labels=list(range(len(labels))))

    if use_wandb:
        # Create and log confusion matrix visualization
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()

    print('\nConfusion Matrix:')
    print(conf_matrix)

    # Generate ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_true_mapped, y_pred_mapped, pos_label=mapping["spam"])
    roc_auc = auc(fpr, tpr)

    if use_wandb:
        # Create and log ROC Curve visualization
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        wandb.log({"roc_curve": wandb.Image(plt)})
        plt.close()

        # Close the wandb run
        wandb.finish()

    print(f'\nROC AUC: {roc_auc:.3f}')