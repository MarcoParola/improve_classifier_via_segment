import torch
import torchvision
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
import cv2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import hydra
import os

class OralClassifierModule(LightningModule):

    def __init__(self, weights, num_classes, lr=10e-3, max_epochs = 100):
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
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        torch.set_grad_enabled(True)

        return self.model(x)

    def training_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        return self._common_step(batch, batch_idx, "train")

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

        img, label = batch
        x = self.preprocess(img)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-5)

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }

        return [optimizer], [lr_scheduler_config]

    def _common_step(self, batch, batch_idx, stage):
        torch.set_grad_enabled(True)

        imgs, labels = batch
        x = self.preprocess(imgs)
        y_hat = self(x)
        loss = self.loss(y_hat, labels)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)

        if stage == "val" and batch_idx == 0:

            if "vit" in self.model_name:
                target_layers = [self.model.conv_proj]
            elif "convnext" in self.model_name:
                target_layers = [self.model.features[-1][-1]]
                # swin è da cercare meglio il target layer
            elif "swin" in self.model_name:
                target_layers = [self.model.features[0][0]]

            # squeezenet da errore sulla log_confusion_matrix
            elif "squeezenet" in self.model_name:
                target_layers = [self.model.features[-1]]

            cam = HiResCAM(model=self, target_layers=target_layers, use_cuda=False)
            for index, image in enumerate(imgs[0:10]):
                label = labels[index]
                target = [ClassifierOutputTarget(label)]
                grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=target)
                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
                image_for_plot = image.permute(1, 2, 0).numpy()
                fig, ax = plt.subplots()
                ax.imshow(image_for_plot)
                ax.imshow((grayscale_cam * 255).astype('uint8'), cmap='jet', alpha=0.75)  # Overlay saliency map
                os.makedirs(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps',
                            exist_ok=True)
                plt.savefig(os.path.join(
                    f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps/saliency_map_epoch_{self.current_epoch}_image_{index}.jpg'),
                            bbox_inches='tight')
                plt.close()


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
                torch.nn.Linear(64, num_classes)
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
                torch.nn.Linear(6, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(64, num_classes)
            )


