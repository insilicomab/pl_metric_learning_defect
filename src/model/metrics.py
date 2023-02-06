from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torchmetrics
from omegaconf import DictConfig


def get_metrics(config: DictConfig):
    metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy(),
                torchmetrics.Precision(
                    num_classes=config.num_classes,
                    average=config.metrics.average,
                ),
                torchmetrics.Recall(
                    num_classes=config.num_classes,
                    average=config.metrics.average,
                ),
                torchmetrics.Specificity(
                    num_classes=config.num_classes,
                    average=config.metrics.average,
                ),
                torchmetrics.F1Score(
                    num_classes=config.num_classes,
                    average=config.metrics.average,
                ),
                torchmetrics.FBetaScore(
                    num_classes=config.num_classes,
                    average=config.metrics.average,
                    beta=config.metrics.f_beta_weight,
                ),
            ]
        )
    
    return metrics


def get_classification_metrics(true, y_pred_proba, y_pred, t_onehot, config: DictConfig):
    accuracy = accuracy_score(true, y_pred)
    precision = precision_score(true, y_pred, average=config.metrics.average)
    recall = recall_score(true, y_pred, average=config.metrics.average)
    f1 = f1_score(true, y_pred, average=config.metrics.average)
    specificity = torchmetrics.functional.specificity(
        torch.tensor(true),
        torch.tensor(y_pred),
        num_classes=config.num_classes,
        average=config.metrics.average
    )
    kappa = cohen_kappa_score(true, y_pred)
    auc = roc_auc_score(t_onehot, y_pred_proba, average='macro')

    results = (accuracy, precision, recall, f1, specificity, kappa, auc)

    return results


def get_classification_report(true, y_pred, int_to_label):
    true = [int_to_label[str(true_)] for true_ in true]
    y_pred = [int_to_label[str(y_pred_)] for y_pred_ in y_pred]
    return classification_report(true, y_pred)


def get_confusion_matrix(true, y_pred, labels):
    cm = confusion_matrix(true, y_pred, labels=labels)
    return cm