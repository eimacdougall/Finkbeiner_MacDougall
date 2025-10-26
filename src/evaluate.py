import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np

def evaluate_model(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)
    return {"accuracy": accuracy, "conf_matrix": conf_matrix}

def evaluate_regressive_model(y_true, y_pred):
    meanerror = metrics.mean_absolute_error(y_true, y_pred)
    return {"MAE": meanerror}

def plot_residuals(y_true, y_pred, title = "residuals", save_path=None):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", cmap="Blues", normalize=False, save_path=None):
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()

def show_predictions(model_name, X, y_true, y_pred, labels, n=10, save_path=None):
    plt.figure(figsize=(15, 4))
    for i in range(n):
        plt.subplot(2, n//2, i+1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        color = "green" if y_true[i] == y_pred[i] else "red"
        plt.title(f"{labels[y_pred[i]]}\n({labels[y_true[i]]})", color=color, fontsize=9)
        plt.axis('off')
    plt.suptitle(f"{model_name} Predictions (Green=Correct, Red=Wrong)", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_model_accuracies(models, accuracies, colors=None, save_path=None):
    if colors is None:
        colors = ['skyblue'] * len(models)
    plt.figure(figsize=(6, 4))
    plt.bar(models, accuracies, color=colors)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')
    plt.ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()

def get_most_confused_class(y_true, y_pred, labels):
    cm = metrics.confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    i, j = np.unravel_index(cm.argmax(), cm.shape)
    return labels[i], labels[j]

def plot_naive_bayes_means(model, labels, save_path=None):
    means = np.exp(model.feature_log_prob_).reshape(len(labels), 28, 28)
    plt.figure(figsize=(12, 6))
    for i in range(len(labels)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(means[i], cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.suptitle("Naive Bayes Learned Feature Means per Class", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_regression_predictions(y_true, y_pred, title="Regression Predictions", save_path=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_target_distribution(y, labels, title="Target Distribution", save_path=None):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()