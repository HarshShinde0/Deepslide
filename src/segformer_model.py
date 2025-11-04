import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from transformers import SegformerForSemanticSegmentation

class LandslideModel(pl.LightningModule):
    def __init__(self, config, alpha=0.5):
        super(LandslideModel, self).__init__()

        self.model_type = config['model_config']['model_type']
        self.in_channels = config['model_config']['in_channels']
        self.num_classes = config['model_config']['num_classes']
        self.alpha = alpha
        self.lr = config['train_config']['lr']

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            ignore_mismatched_sizes=True,
            num_labels=self.num_classes
        )

        # Modify the input layer for 14 channels
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.model.segformer.encoder.patch_embeddings[0].proj.out_channels,
            kernel_size=self.model.segformer.encoder.patch_embeddings[0].proj.kernel_size,
            stride=self.model.segformer.encoder.patch_embeddings[0].proj.stride,
            padding=self.model.segformer.encoder.patch_embeddings[0].proj.padding
        )

        self.weights = torch.tensor([5], dtype=torch.float32).to(self.device)
        self.wce = nn.BCELoss(weight=self.weights)

        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')

        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')

        self.train_recall = torchmetrics.Recall(task='binary')
        self.val_recall = torchmetrics.Recall(task='binary')

        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self(x))

        # Resize y_hat to match the size of y
        y_hat = nn.functional.interpolate(y_hat, size=y.shape[2:], mode='bilinear', align_corners=False)

        wce_loss = self.wce(y_hat, y)
        dice = dice_loss(y_hat, y)

        combined_loss = (1 - self.alpha) * wce_loss + self.alpha * dice

        precision = self.train_precision(y_hat, y)
        recall = self.train_recall(y_hat, y)
        iou = self.train_iou(y_hat, y)
        loss_f1 = self.train_f1(y_hat, y)

        self.log('train_precision', precision)
        self.log('train_recall', recall)
        self.log('train_wce', wce_loss)
        self.log('train_dice', dice)
        self.log('train_iou', iou)
        self.log('train_f1', loss_f1)
        self.log('train_loss', combined_loss)
        return {'loss': combined_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = torch.sigmoid(self(x))

        # Resize y_hat to match the size of y
        y_hat = nn.functional.interpolate(y_hat, size=y.shape[2:], mode='bilinear', align_corners=False)

        wce_loss = self.wce(y_hat, y)
        dice = dice_loss(y_hat, y)

        combined_loss = (1 - self.alpha) * wce_loss + self.alpha * dice

        precision = self.val_precision(y_hat, y)
        recall = self.val_recall(y_hat, y)
        iou = self.val_iou(y_hat, y)
        loss_f1 = self.val_f1(y_hat, y)

        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_wce', wce_loss)
        self.log('val_dice', dice)
        self.log('val_iou', iou)
        self.log('val_f1', loss_f1)
        self.log('val_loss', combined_loss)

        if self.current_epoch % 10 == 0:
            x = (x - x.min()) / (x.max() - x.min())
            x = x[:, 0:3]
            x = x.permute(0, 2, 3, 1)
            y_hat = (y_hat > 0.5).float()

            class_labels = {0: "no landslide", 1: "landslide"}

            self.logger.experiment.log({
                "image": wandb.Image(x[0].cpu().detach().numpy(), masks={ 
                    "predictions": {
                        "mask_data": y_hat[0][0].cpu().detach().numpy(),
                        "class_labels": class_labels
                    },
                    "ground_truth": {
                        "mask_data": y[0][0].cpu().detach().numpy(),
                        "class_labels": class_labels
                    }
                })
            })
        return {'val_loss': combined_loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]

def dice_loss(y_hat, y):
    smooth = 1e-6
    y_hat = y_hat.view(-1)
    y = y.view(-1)
    intersection = (y_hat * y).sum()
    union = y_hat.sum() + y.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice