from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .loss import get_loss_fn
from .metrics import (get_classification_metrics, get_classification_report,
                      get_confusion_matrix, get_metrics, get_roc_curve)
from .model import EncoderWithHead
from .optimizer import get_optimizer
from .scheduler import get_scheduler


class Net(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

        # label
        self.int_to_label = { v:k for k, v in config.label_map.items() }

        # model
        self.model = EncoderWithHead(
            model_name=config.encoder.model_name,
            pretrained=config.encoder.pretrained,
            layer_name=config.layer.name,
            embedding_size=config.embedding_size,
            num_classes=config.num_classes,
            s=config.layer.s,
            m=config.layer.m,
            eps=config.layer.eps,
            k=config.layer.k,
        )

        # loss function
        self.loss_fn = get_loss_fn(config=self.config)

        # metrics
        metrics = get_metrics(config=self.config)
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
    

    def forward(self, x, t):
        return self.model(x, t)


    def configure_optimizers(self):
        optimizer = get_optimizer(config=self.config, net=self.model)
        scheduler = get_scheduler(config=self.config, optimizer=optimizer)
        return [optimizer], [scheduler]
    

    def training_step(self, batch, batch_idx):
        x, t = batch
        logits = self(x, t)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, t)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_metrics(preds, t)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        x, t = batch
        logits = self(x, t)
        preds = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, t)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.val_metrics(preds, t)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}
    

    def test_step(self, batch, batch_idx):
        x, t = batch
        logits = self(x, t)
        y_proba = F.softmax(logits, dim=1)
        y_pred = torch.argmax(logits, dim=1)
        t_onehot = torch.eye(self.config.num_classes)[t.to('cpu')]
        outputs = {
            'true': t.to('cpu').squeeze(),
            'proba': y_proba.to('cpu'),
            'pred': y_pred.to('cpu').squeeze(), 
            'true_onehot': t_onehot.to('cpu'),
        }
        return outputs
    

    def test_epoch_end(self, outputs):
        true = np.array([output['true'] for output in outputs])

        y_proba = [output['proba'] for output in outputs]
        y_proba = torch.cat(y_proba, dim=0)

        y_pred = np.array([output['pred'] for output in outputs])

        t_onehot = [output['true_onehot'] for output in outputs]
        t_onehot = torch.cat(t_onehot, dim=0)

        self._evaluate((true, y_proba, y_pred, t_onehot))

    
    def _evaluate(self, ys):
        """
        Evaluation pipeline.
        """
        true, y_proba, y_pred, t_onehot = ys
        self._log_metrics(true, y_proba, y_pred, t_onehot)
        self._save_confusion_matrix(true, y_pred)
        self._save_classification_report(true, y_pred)
        self._save_roc_curve(t_onehot, y_proba)
    

    def _log_metrics(self, true, y_proba, y_pred, t_onehot):
        results = get_classification_metrics(true, y_proba, y_pred, t_onehot, config=self.config)
        accuracy, precision, recall, f1, specificity, kappa, auc = results
        self.log('test_accuracy', accuracy)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_fbeta', f1)
        self.log('test_specifity', specificity)
        self.log('test_kappa', kappa)
        self.log('test_auc', auc)


    def _save_classification_report(self, true, y_pred):
        """
        Save classification report to txt.
        """
        cls_report_str = get_classification_report(true, y_pred, self.int_to_label)
        with open('output/classification_report.txt', 'w') as f:
            f.write(cls_report_str)
    

    def _save_confusion_matrix(self, true, y_pred):
        """
        Save confusion matrix.
        """
        cm = get_confusion_matrix(true, y_pred,labels=np.arange(len(self.int_to_label)))
        df_cm = pd.DataFrame(cm, index=self.int_to_label.values(), columns=self.int_to_label.values())
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
        plt.xlabel('Prediction label')
        plt.ylabel('True label')
        plt.savefig('output/confusion_matrix.png')
    

    def _save_roc_curve(self, t_onehot, y_proba):
        """
        Save ROC curve
        """
        fpr, tpr, roc_auc = get_roc_curve(t_onehot, y_proba, self.config)

        plt.figure()
        plt.plot(
            fpr['micro'],
            tpr['micro'],
            label=f'micro-average ROC curve (area = {roc_auc["micro"]:0.2f})',
            color='deeppink',
            linestyle=':',
            linewidth=4,
        )

        plt.plot(
            fpr['macro'],
            tpr['macro'],
            label=f'macro-average ROC curve (area = {roc_auc["macro"]:0.2f})',
            color='navy',
            linestyle=':',
            linewidth=4,
        )

        colors = cycle([
            'aqua', 'darkorange', 'cornflowerblue', 'seagreen', 'tomato'
        ])
        for i, color in zip(range(self.config.num_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f'ROC curve of class {self.int_to_label[i]} (area = {roc_auc[i]:0.2f})',
            )

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multiclass')
        plt.legend(loc='lower right')
        plt.savefig('output/roc_curve.png')