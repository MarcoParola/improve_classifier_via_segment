import hydra
import torch
import pytorch_lightning as pl
from sklearn.metrics import classification_report
import numpy as np

from src.models.classification import *
from src.data.datamodule import * # TODO: change this
from src.log import LossLogCallback, get_loggers
from src.utils import *
from test import predict


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    callbacks = list()
    callbacks.append(get_early_stopping(cfg))
    callbacks.append(LossLogCallback())
    callbacks.append(HydraTimestampRunCallback())
    loggers = get_loggers(cfg)

    model, data = get_model_and_data(cfg)

    # training
    trainer = pl.Trainer(
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        log_every_n_steps=1,
        max_epochs=cfg.train.max_epochs
    )
    trainer.fit(model, data)

    # prediction
    predict(trainer, model, data, cfg.saliency.method, cfg.classification_mode)




def get_model_and_data(cfg):
    ''' 
    This function returns a model and data based on the provided configuration.
    Depending on the task specified in the configuration, it can return either a classifier or a segmenter.
    Args:
        cfg: configuration
    Returns:
        model: model
        data: data
    '''
    model, data = None, None
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)

    # CLASSIFICATION WHOLE
    if cfg.task == 'c' or cfg.task == 'classification':
        if cfg.classification_mode=='whole':
            # classification model
            model = OralClassifierModule(
                weights=cfg.model.weights,
                num_classes=cfg.model.num_classes,
                lr=cfg.train.lr,
                max_epochs = cfg.train.max_epochs
            )
            # whole data
            data = OralClassificationDataModule(
                train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform = train_img_tranform,
                val_transform = val_img_tranform,
                test_transform = test_img_tranform,
                transform = img_tranform,
            )

        # CLASSIFICATION MASKED
        elif cfg.classification_mode == 'masked':
            # classification model
            model = OralMaskedClassifierModule(
                weights=cfg.model.weights,
                num_classes=cfg.model.num_classes,
                lr=cfg.train.lr,
                # max_epochs = cfg.train.max_epochs
            )
            # masked data
            data = OralClassificationMaskedDataModule(
                train=cfg.dataset.test,
                #train=cfg.dataset.train,
                val=cfg.dataset.val,
                test=cfg.dataset.test,
                batch_size=cfg.train.batch_size,
                train_transform=train_img_tranform,
                val_transform=val_img_tranform,
                test_transform=test_img_tranform,
                transform=img_tranform,
            )


    # SEGMENTATION
    elif cfg.task == 's' or cfg.task == 'segmentation':
        model = None
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

    return model, data







if __name__ == "__main__":
    main()

    