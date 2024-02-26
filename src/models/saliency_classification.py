import torch
import torchvision
from pytorch_lightning import LightningModule
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt
from src.utils import *
import os
import hydra

from src.losses import SaliencyAwareLoss


class OralSaliencyClassifierModule(LightningModule):

    def __init__(self, weights, num_classes, lr=10e-3, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        weights_cls = weights.split(".")[0]
        weights_name = weights.split(".")[1]
        self.model_name = weights.split("_Weights")[0].lower()

        weights_cls = getattr(torchvision.models, weights_cls)
        weights = getattr(weights_cls, weights_name)

        self.model = getattr(torchvision.models, self.model_name)(weights=weights)
        self._set_model_classifier(weights_cls, num_classes)
        self.preprocess = weights.transforms()
        self.loss = SaliencyAwareLoss(weight_loss=0.7)
        self.loss.requires_grad_(True)
        self.total_predictions = None
        self.total_labels = None
        self.classes = ['Neoplastic', 'Aphthous', 'Traumatic']

    def forward(self, x):
        torch.set_grad_enabled(True)

        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.eval()
        imgs, labels, _ = batch
        x = self.preprocess(imgs)
        y_hat = self(x)
        predictions = torch.argmax(y_hat, dim=1)
        self.log(f'test_accuracy', accuracy_score(labels, predictions), on_step=True, on_epoch=True, logger=True)
        self.log('recall', recall_score(labels, predictions, average='micro'), on_step=True, on_epoch=True, logger=True)
        self.log('precision', precision_score(labels, predictions, average='micro'), on_step=True, on_epoch=True,
                 logger=True)
        self.log('f1', f1_score(labels, predictions, average='micro'), on_step=True, on_epoch=True, logger=True)

        # this accumulation is necessary in order to log confusion matrix of all the test and not just the last step
        if self.total_labels is None:
            self.total_labels = labels.numpy()
            self.total_predictions = predictions.numpy()
        else:
            self.total_labels = np.concatenate((self.total_labels, labels.numpy()), axis=None)
            self.total_predictions = np.concatenate((self.total_predictions, predictions.numpy()), axis=None)

        # check if it's the last test step
        if self.trainer.num_test_batches[0] == batch_idx + 1:
            # logging confusion matrix on wandb
            log_confusion_matrix_wandb(self.logger.__class__.__name__.lower(), self.logger.experiment,
                                       self.total_labels, self.total_predictions, self.classes)
            # get tensorboard logger if present il loggers list
            tb_logger = get_tensorboard_logger(self.trainer.loggers)
            # logging confusion matrix on tensorboard
            log_confusion_matrix_tensorboard(actual=self.total_labels, predicted=self.total_predictions,
                                             classes=self.classes, writer=tb_logger)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        torch.set_grad_enabled(True)
        img, label, mask = batch
        x = self.preprocess(img)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
            # "monitor": "val_loss",
        }

        return [optimizer], [lr_scheduler_config]

    def print_map_stats(self, saliency_map):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Maximum:", np.max(saliency_map))
        print("Minimum:", np.min(saliency_map))
        print("Average:", np.mean(saliency_map))
        print("Shape:", saliency_map.shape)
        print("Type: ", type(saliency_map[0][0]))
        count_elements_gt_05 = 0
        count_elements_gt_05 = np.sum(saliency_map > 0.5)
        print("Numbers greater than 0.5: ", count_elements_gt_05)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    def get_salient_area(self, imgs, labels, stage, batch_index):
        if "vit" in self.model_name:
            target_layers = [self.model.conv_proj]
        elif "convnext" in self.model_name:
            target_layers = [self.model.features[-1][-1]]
        elif "swin" in self.model_name:
            target_layers = [self.model.features[7][1].norm2]
        elif "squeezenet" in self.model_name:
            target_layers = [self.model.features[-1]]

        cam = HiResCAM(model=self, target_layers=target_layers, use_cuda=False)
        result = []

        for index, image in enumerate(imgs):
            label = labels[index]
            target = [ClassifierOutputTarget(label)]
            grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=target)
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (224, 224))

            # save the saliency maps of the first 10 images of each batch during validation
            if stage == "val" and index < 10 and batch_index == 0:
                image_for_plot = image.permute(1, 2, 0).numpy()
                fig, ax = plt.subplots()
                ax.imshow(image_for_plot)
                ax.imshow((grayscale_cam*255).astype('uint8'), cmap='jet', alpha=0.75)  # Overlay saliency map
                os.makedirs(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps',
                            exist_ok=True)
                plt.savefig(os.path.join(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps/saliency_map_epoch_{self.current_epoch}_image_{index}.pdf'), bbox_inches='tight')
                plt.close()

            _, grayscale_cam = cv2.threshold(grayscale_cam, 0.5, 1, cv2.THRESH_BINARY)
            result.append(grayscale_cam)

        result = torch.tensor(result)
        return result

    def _common_step(self, batch, batch_idx, stage):
        torch.set_grad_enabled(True)
        img, label, mask = batch
        x = self.preprocess(img)
        y_hat = self(x)
        salient_area = self.get_salient_area(x, label, stage, batch_idx)

        mask = np.array(mask).squeeze()
        salient_area = np.array(salient_area).squeeze()
        loss = self.loss(label, y_hat, salient_area, mask, self.current_epoch, stage)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)
        loss.requires_grad_(True)
        return loss

    def _set_model_classifier(self, weights_cls, num_classes):
        weights_cls = str(weights_cls)
        if "ConvNeXt" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Flatten(1),
                torch.nn.Linear(self.model.classifier[2].in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "EfficientNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[1].in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "MobileNet" in weights_cls or "VGG" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier[0].in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes),
            )

        elif "DenseNet" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.classifier.in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "MaxVit" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(self.model.classifier[5].in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "ResNet" in weights_cls or "RegNet" in weights_cls or "GoogLeNet" in weights_cls:
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.fc.in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "Swin" in weights_cls:
            self.model.head = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.head.in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "ViT" in weights_cls:
            self.model.heads = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(self.model.hidden_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )
        elif "SqueezeNet1_1" in weights_cls or "SqueezeNet1_0" in weights_cls:
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(kernel_size=13, stride=1, padding=0)
            )
