import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix', cmap='Blues'):
    """
    Plots a confusion matrix with optional custom class names and title.

    Parameters:
    - y_true: list or array of true labels
    - y_pred: list or array of predicted labels
    - class_names: list of label names (e.g., ['Class 0', 'Class 1'])
    - title: title of the plot
    - cmap: color map for the heatmap
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, [f'Predicted {label}' for label in class_names])
    plt.yticks(tick_marks, [f'Actual {label}' for label in class_names])

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="black", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_true_valence, y_score_valence, y_true_arousal, y_score_arousal, figsize=(12, 5)):
    """
    Plots ROC curves for both Valence and Arousal in a side-by-side layout.

    Parameters:
    - y_true_valence: Ground truth binary labels for valence
    - y_score_valence: Predicted probabilities for valence
    - y_true_arousal: Ground truth binary labels for arousal
    - y_score_arousal: Predicted probabilities for arousal
    - figsize: Size of the figure
    """
    # ROC for Valence
    fpr_val, tpr_val, _ = roc_curve(y_true_valence, y_score_valence)
    auc_val = auc(fpr_val, tpr_val)

    # ROC for Arousal
    fpr_aro, tpr_aro, _ = roc_curve(y_true_arousal, y_score_arousal)
    auc_aro = auc(fpr_aro, tpr_aro)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Valence Plot
    axes[0].plot(fpr_val, tpr_val, color='blue', lw=2, label=f'ROC curve (AUC = {auc_val:.2f})')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Valence ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    # Arousal Plot
    axes[1].plot(fpr_aro, tpr_aro, color='green', lw=2, label=f'ROC curve (AUC = {auc_aro:.2f})')
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Arousal ROC Curve')
    axes[1].legend(loc="lower right")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
