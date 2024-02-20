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
