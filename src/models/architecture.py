import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics
from pytorch_lightning.callbacks import Callback


class LoanApprovalModel(pl.LightningModule):
    def __init__(self, input_dim, lr=0.001, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        self.test_confmat = torchmetrics.ConfusionMatrix(task="binary")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.train_acc(y_pred, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.val_acc(y_pred, y)
        self.val_f1(y_pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=False)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.test_acc(y_pred, y)
        self.test_f1(y_pred, y)
        self.test_confmat(y_pred, y)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


class KerasProgressBar(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}", end=" ")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            t_loss = metrics['train_loss'].item()
            v_loss = metrics.get('val_loss', torch.tensor(0.0)).item()
            v_acc = metrics.get('val_acc', torch.tensor(0.0)).item()
            v_f1 = metrics.get('val_f1', torch.tensor(0.0)).item()
            print(f"- loss: {t_loss:.4f} - val_loss: {v_loss:.4f} - val_acc: {v_acc:.4f} - val_f1_score: {v_f1:.4f}")