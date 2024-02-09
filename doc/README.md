# **Documentation**

The project is composed of the following modules, more details are below:

- [Main scripts for training and test models](#main-scripts-for-training-and-test-models)
- [Datasets and datamodules](#datasets-and-datamodules)
- [Deep learning models implemented for each experiment](#deep-learning-models-implemented-for-each-experiments)
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

## Datasets and datamodules

## Deep learning models implemented for each experiment

## Configuration handling

## Additional utility scripts

In the `scripts/` folder, there are all independent files not involved in the `pytorch-lightning` workflow for data preparation and visualization.

Due to the possibility of errors in the dataset, such as missing images, run the check-dataset.py script to detect such errors. Returns the elements to be removed from the json file (this can be done manually or via a script).
```bash
python -m scripts.check-dataset --dataset data\coco_dataset.json
```
In this work, the dataset was annotated with more labels than necessary. Some are grouped under more general labels. To aggregate all the labels of the three diseases studied in this work, the following script is executed. In addition, we split the dataset with the holdout method.
```bash
python -m scripts.simplify-dataset --folder data
python -m scripts.split-dataset --folder data
```

You can use the `dataset-stats.py`   script to print the class occurrences for each dataset.
```bash
python -m scripts.dataset-stats --dataset data\dataset.json # entire dataset
python -m scripts.dataset-stats --dataset data\train.json # training set
python -m scripts.dataset-stats --dataset data\test.json # test set
```
