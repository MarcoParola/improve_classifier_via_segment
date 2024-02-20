# **Documentation**

The project is composed of the following modules, more details are below:

- [Main scripts for training and test models](#main-scripts-for-training-and-test-models)
- [Pytorch-lightning logic modules (data and models)](#pytorch-lightning-logic-modules)
- [Configuration handling](#configuration-handling)
- [Additional utility scripts](#additional-utility-scripts)


## Main scripts for training and test models

All the experiments consist of train and test classification and segmentation architectures. You can use `train.py` and `test.py`, respectively. Where you have to set the `task` option to specify the task your architecture is solving:
- `task=c` or `task=classification` to run classification experiments 
- `task=s` or `task=segmentation` to run segmentation experiments 

```bash
python train.py task=...
python test.py task=...
```

## Pytorch-lightning logic modules
Since this project is developed using the `pytorch-lightning` framework, two key concepts are `Modules` to declare a new model and `DataModules` to organize of our dataset. Both of them are declared in `src/`, specifically in `src/models/` and `src/data/`, respectively. More information are in the next sections.

### Deep learning models
The three deep learning models are  `src/models/classification.py`,  `src/models/saliency_classification.py` and  `src/models/segmentation.py`.

**`classification.py`** is the model that implements the workflow of the standard classification problem, the model is trained in a supervised fashion and the loss function is the CrossEntropyLoss so experiments which use this model exploits only labels information for backpropagation purposes. This model is exploited both for Experiment 1 and for Experiment 2, the difference among the two experiments lies in the preparation of the data in input to the classifier and will be further explained in the next section. 

**`saliency_classification.py`** is the model exploited in Experiment 3. The core difference, from the model perspective, with respect to `classification.py` is that in this case the loss function is a custom loss defined in this work called CEntIoU which exploits both the information of the labels, in the CrossEntropy component, and of the segment annotations, in the IoU component, which is computed as the Intesection of binarized saliency maps and ground truth masks over their Union. Saliency maps and their IoU with ground truth annotations are computed in the [_common_step](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/models/saliency_classification.py#L130) of this model. 

**`segmentation.py`** is the model that allows to train the model for the segmentation problem. The two topologies configured in this file are FCN and DeepLab. This model is based on a supervised training of a segmentator in which the segment informations of the dataset are used as ground truth. This model is used in this project to train segmentators to be exploited in the segmentation step of the Experiment 2.

### Datasets and datamodules
#### Datasets
In the Dataset objects is mandatory to define three methods: `__init__`, `__get_item__` and `__len__`. Those three methods allows respectively to: build the dataset objects, opening all the necessary files for images and annotations, retrieve one item from our dataset and obtain the length of our dataset. Those three methods are automatically called during the PyTorch Lightning workflow. In this project there are four different Datasets objects:

[`src/data/classification/dataset.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/classification/dataset.py): is the Dataset object exploited for Experiment 1, it is composed of only images and corresponding labels.

[`src/data/saliency_classification/dataset.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/saliency_classification/dataset.py): is the Dataset object exploited for Experiment 3, it is composed of images, labels and segments.  

[`src/data/masked_classification/dataset.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/masked_classification/dataset.py): is the Dataset object exploited for Experiment 2, it is composed of images and labels. The particularity of this Dataset is that in the `__init__` method is loaded a previously trained segmentator, and in the `__get_item__` the image is retrieved, passed in input to the segmentator which performs a forward step, a mask is obtained and applied to the original image, in this way the method returns the masked image and the label.

[`src/data/segmentation/dataset.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/segmentation/dataset.py): is the Dataset object for the training of the segmentator, in this case instead of categorical labels there are segment informations.

### Datamodules
Datamodules are objects which, in the PyTorch Lightning framework, behave as interfaces between the Trainer and the Dataset. The core part of Datamodules is to define train_dataloader, val_dataloader and test_dataloader which are then handled by the Trainer object in order to automatize all the workflow in the Lightning fashion. In this project there is a Datamodule for each Dataset ([`src/data/classification/dataloader.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/classification/dataloader.py), [`src/data/masked_classification/dataloader.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/masked_classification/dataloader.py), [`src/data/saliency_classification/dataloader.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/saliency_classification/dataloader.py) and [`src/data/segmentation/dataloader.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/segmentation/dataloader.py)). When the Dataloader object is created in [train.py](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/train.py) or in [test.py](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/test.py) all the passed parameters are read from hydra configuration.

## Configuration handling
The configuration managed with [Hydra](https://hydra.cc/). Every aspect of the configuration is located in `config/` folder. The file containing all the configuration is `config.yaml`.

## Additional utility scripts

In the `scripts/` folder, there are all independent files not involved in the `pytorch-lightning` workflow for data preparation and visualization.

Due to the possibility of errors in the dataset, such as missing images, run the check-dataset.py script to detect such errors. Returns the elements to be removed from the json file (this can be done manually or via a script).
```bash
python -m scripts.check-dataset --dataset data/coco_dataset.json
```
In this work, the dataset was annotated with more labels than necessary. Some are grouped under more general labels. To aggregate all the labels of the three diseases studied in this work, the following script is executed. In addition, we split the dataset with the holdout method.
```bash
python -m scripts.simplify-dataset --folder data
python -m scripts.split-dataset --folder data
```

You can use the `dataset-stats.py`   script to print the class occurrences for each dataset.
```bash
python -m scripts.dataset-stats --dataset data/dataset.json # entire dataset
python -m scripts.dataset-stats --dataset data/train.json # training set
python -m scripts.dataset-stats --dataset data/test.json # test set
```
