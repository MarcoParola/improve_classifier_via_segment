import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter
from src.utils import *


class LossLogCallback(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())

    def on_train_end(self, trainer, pl_module):
        log_dir = 'logs/oral/' + get_last_version('logs/oral')
        writer = SummaryWriter(log_dir=log_dir)
        for i in range(0, len(self.train_losses)):
            writer.add_scalars('train_val_loss', {'train': self.train_losses[i],
                                                  'val': self.val_losses[i]}, i)
        writer.close()


class HydraTimestampRunCallback(pl.Callback):

    #@hydra.main(version_base=None, config_path="./config", config_name="config")
    def on_train_end(self, trainer, pl_module):
        # absolute path
        log_dir = 'logs/oral/' + get_last_version('logs/oral')
        hydra_current_timestamp = f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}'
        # decompose in day and time
        day = hydra_current_timestamp.split("\\")[-2]
        time = hydra_current_timestamp.split("\\")[-1]
        f = open(log_dir + "/hydra_run_timestamp.txt", "w+")
        # in this way is saved just day and time not all the absolute path
        f.write(day + '/' + time)
        f.close()


def get_loggers(cfg):
    """Returns a list of loggers
    cfg: hydra config
    """
    loggers = list()
    if cfg.log.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        hyperparameters = hp_from_cfg(cfg)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        wandb.config.update(hyperparameters)
        wandb_logger = WandbLogger()
        loggers.append(wandb_logger)

    if cfg.log.tensorboard:
        from pytorch_lightning.loggers import TensorBoardLogger
        tensorboard_logger = TensorBoardLogger(cfg.log.path, name="oral")
        loggers.append(tensorboard_logger)

    return loggers