import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


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