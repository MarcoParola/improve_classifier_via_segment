import os
import re
import torchvision
import numpy as np  
import pandas as pd
import seaborn as sn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.multiclass import unique_labels
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform


def get_early_stopping(cfg):
    """Returns an EarlyStopping callback
    cfg: hydra config
    """
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=15,
    )
    return early_stopping_callback



def get_transformations(cfg):
    """Returns the transformations for the dataset
    cfg: hydra config
    """
    img_tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg.dataset.resize, antialias=True),
        torchvision.transforms.CenterCrop(cfg.dataset.resize),
        torchvision.transforms.ToTensor(),
    ])
    val_img_tranform, test_img_tranform = None, None


    train_img_tranform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(cfg.dataset.resize, antialias=True),
        torchvision.transforms.CenterCrop(cfg.dataset.resize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    ])
    return train_img_tranform, val_img_tranform, test_img_tranform, img_tranform
    


def log_confusion_matrix(actual, predicted, classes, log_dir):
    """Logs the confusion matrix to tensorboard
    actual: ground truth
    predicted: predictions
    classes: list of classes
    log_dir: path to the log directory
    """
    writer = SummaryWriter(log_dir=log_dir)
    cf_matrix = confusion_matrix(actual, predicted)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None], 
        index=[i for i in classes],
        columns=[i for i in classes])
    plt.figure(figsize=(5, 4))
    img = sn.heatmap(df_cm, annot=True, cmap="Greens").get_figure()
    writer.add_figure("Confusion matrix", img, 0)

    # log metrics on tensorboard
    writer.add_scalar('Accuracy', accuracy_score(actual, predicted))
    writer.add_scalar('recall', recall_score(actual, predicted, average='micro'))
    writer.add_scalar('precision', precision_score(actual, predicted, average='micro'))
    writer.add_scalar('f1', f1_score(actual, predicted, average='micro'))
    writer.close()



def get_last_version(path):
    """Return the last version of the folder in path
    path: path to the folder containing the versions
    """
    folders = os.listdir(path)
    # get the folders starting with 'version_'
    folders = [f for f in folders if re.match(r'version_[0-9]+', f)]
    # get the last folder with the highest number
    last_folder = max(folders, key=lambda f: int(f.split('_')[1]))
    return last_folder


def get_last_checkpoint(version):
    checkpoint_dir = "logs/oral/version_" + str(version) + "/checkpoints"
    # all files in the directory
    all_files = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
    # list of checkpoints
    checkpoint_files = [f for f in all_files if f.endswith('.ckpt')]
    # sort checkpoints
    checkpoint_files_sorted = sorted(checkpoint_files)
    # get last checkpoint
    if checkpoint_files_sorted:
        last_checkpoint = checkpoint_files_sorted[-1]
        full_path = os.path.join(checkpoint_dir, last_checkpoint)
        return full_path


def convert_arrays_to_integers(array1, array2):
    """
    Given two arrays of strings, return two arrays of integers
    such that the integers are a mapping of the strings.  If a
    string in array1 is not in array2, it is mapped to the next
    integer in the sequence.  If a string in array2 is not in
    array1, it is mapped to the next integer in the sequence.
    """
    string_to_int_mapping = {}
    next_integer = 0
    
    for string in array1:
        if string not in string_to_int_mapping:
            string_to_int_mapping[string] = next_integer
            next_integer += 1
    
    result_array1 = [string_to_int_mapping[string] for string in array1]
    result_array2 = [string_to_int_mapping.get(string, next_integer) for string in array2]
    return result_array1, result_array2