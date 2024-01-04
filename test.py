import torch
import pytorch_lightning
import omegaconf

from sklearn.metrics import classification_report

from src.data.segmentation.datamodule import OralSegmentationDataModule
from src.models.classification import OralClassifierModule
from src.models.masked_classification import OralMaskedClassifierModule

from src.data.classification.datamodule import OralClassificationDataModule
from src.data.classification.dataset import OralClassificationDataset
from src.data.masked_classification.datamodule import OralClassificationMaskedDataModule
from src.data.masked_classification.dataset import OralClassificationMaskedDataset
from src.models.segmentation import FcnSegmentationNet, DeeplabSegmentationNet
from src.saliency.grad_cam import OralGradCam
from src.saliency.lime import OralLime
from src.saliency.shap import OralShap
import hydra

from src.utils import *
from src.log import get_loggers


def predict(trainer, model, data, saliency_map_flag, task, classification_mode):

    if task == 'c' or task == 'classification':
        predictions = trainer.predict(model, data)
        predictions = torch.cat(predictions, dim=0)
        predictions = torch.argmax(predictions, dim=1)

        if classification_mode == 'masked':
            gt = torch.cat([y for _, y, _ in data.test_dataloader()], dim=0)
        elif classification_mode == 'whole':
            gt = torch.cat([y for _, y in data.test_dataloader()], dim=0)

        print(classification_report(gt, predictions))

        class_names = np.array(['Neoplastic', 'Aphthous', 'Traumatic'])
        log_dir = 'logs/oral/' + get_last_version('logs/oral')
        log_confusion_matrix(gt, predictions, classes=class_names, log_dir=log_dir)

        if saliency_map_flag == "grad-cam":
            OralGradCam.generate_saliency_maps_grad_cam(model, data.test_dataloader(), predictions, classification_mode)

    elif task == 's' or task == 'segmentation':
        trainer.test(model, data)

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    # this main load a checkpoint saved and perform test on it

    # save the passed version number before overwriting the configuration with training configuration
    version = str(cfg.checkpoint.version)
    # save the passed saliency map generation method before overwriting the configuration with training configuration
    saliency_map_method = cfg.generate_map
    # find the hydra_run_timestamp.txt file
    f = open('./logs/oral/version_' + version + '/hydra_run_timestamp.txt', "r")
    # read the timestamp inside hydra_run_timestamp.txt
    timestamp = f.read()
    # use the timestamp to build the path to hydra configuration
    path = './outputs/' + timestamp + '/.hydra/config.yaml'
    # load the configuration used during training
    cfg = omegaconf.OmegaConf.load(path)

    # to test is needed: trainer, model and data
    # trainer
    trainer = pytorch_lightning.Trainer(
        logger=get_loggers(cfg),
        # callbacks=callbacks,  shouldn't need callbacks in test
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs,
        # gradient_clip_val=0.1,
        # gradient_clip_algorithm="value"
    )

    model = None
    data = None

    if cfg.task == 'c' or cfg.task == 'classification':
        # whole classification
        if cfg.classification_mode == 'whole':
            # model
            # get the model already trained from checkpoints, default checkpoint is version_0, otherwise specify by cli
            model = OralClassifierModule.load_from_checkpoint(get_last_checkpoint(version))
            model.eval()

            # data
            train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
            data = OralClassificationDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform
            )

        # masked classification
        elif cfg.classification_mode == 'masked':
            model = OralMaskedClassifierModule.load_from_checkpoint(get_last_checkpoint(version))
            model.eval()

            # data
            train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
            data = OralClassificationMaskedDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform
            )

    elif cfg.task == 's' or cfg.task == 'segmentation':
        train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
        data = OralSegmentationDataModule(
            train=cfg.dataset.train,
            val=cfg.dataset.val,
            test=cfg.dataset.test,
            batch_size=cfg.train.batch_size,
            train_transform=train_img_tranform,
            val_transform=val_img_tranform,
            test_transform=test_img_tranform,
            transform=img_tranform
        )
        if cfg.model_seg == 'fcn':
            model = FcnSegmentationNet(lr=cfg.train.lr, epochs=cfg.train.max_epochs,
                                       len_dataset=data.train_dataset.__len__(), batch_size=cfg.train.batch_size)
        elif cfg.model_seg == 'deeplab':
            model = DeeplabSegmentationNet(lr=cfg.train.lr, epochs=cfg.train.max_epochs,
                                           len_dataset=data.train_dataset.__len__(), batch_size=cfg.train.batch_size)
        model.eval()

    predict(trainer, model, data, saliency_map_method, cfg.task, cfg.classification_mode)




if __name__ == "__main__":
    main()
