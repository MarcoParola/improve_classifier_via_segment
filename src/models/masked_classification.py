import torch
import torchvision
from pytorch_lightning import LightningModule
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import accuracy_score
import cv2
import numpy as np

from src.saliency_aware_loss import SaliencyAwareLoss


class OralMaskedClassifierModule(LightningModule):

    def __init__(self, model, weights, num_classes, lr=10e-3, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        assert "." in weights, "Weights must be <MODEL>.<WEIGHTS>"
        weights_cls = weights.split(".")[0]
        weights_name = weights.split(".")[1]

        weights_cls = getattr(torchvision.models, weights_cls)
        weights = getattr(weights_cls, weights_name)

        self.model = getattr(torchvision.models, model)(weights=weights)

        self._set_model_classifier(weights_cls, num_classes)

        self.preprocess = weights.transforms()
        self.loss = SaliencyAwareLoss(weight_loss=0.7)
        self.loss.requires_grad_(True)

    def forward(self, x):
        torch.set_grad_enabled(True)


        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        output = self._common_step(batch, batch_idx, "train")
        return output

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        self._common_step(batch, batch_idx, "test")
        output = self(batch)
        accuracy = accuracy_score(output, batch['target'])
        self.log('test_accuracy', accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        torch.set_grad_enabled(True)

        img, label, mask = batch
        x = self.preprocess(img)
        output = self(x)
        return output

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
        count_elements_gt_05 = 0
        count_elements_gt_05 = np.sum(saliency_map > 0.5)
        print("Numbers greater than 0.5: ", count_elements_gt_05)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


    def get_salient_area(self, imgs, predictions, stage):

        target_layers = [self.model.features[-1][-1]]
        cam = HiResCAM(model=self, target_layers=target_layers, use_cuda=False)
        predictions = torch.argmax(predictions, dim=1)
        result = []
        print("len(imgs) = ", len(imgs))

        for index, image in enumerate(imgs):
            prediction = predictions[index]
            target = [ClassifierOutputTarget(prediction)]
            grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=target)
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, (224, 224))

            # monnezza
            if stage == "val":
                self.print_map_stats(grayscale_cam)


            _, grayscale_cam = cv2.threshold(grayscale_cam, 0.5, 1, cv2.THRESH_BINARY)




            result.append(grayscale_cam)


        result = torch.tensor(result)
        print(result.shape)
        return result

    def _common_step(self, batch, batch_idx, stage):
        torch.set_grad_enabled(True)
        print("len(batch) = ", len(batch))
        img, label, mask = batch
        x = self.preprocess(img)
        print("len(x) = ", len(x))
        y_hat = self(x)
        salient_area = self.get_salient_area(x, y_hat, stage)

        loss = self.loss(label, y_hat, salient_area, mask)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)
        loss = torch.tensor(loss, dtype=torch.float32)
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
        elif "SqueezeNet1_1" in weights_cls:
            self.model.heads = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(6, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )