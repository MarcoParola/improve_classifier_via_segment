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

### **Exp 1** (baseline)
Classification on the whole dataset:

- Train CNN classifier on the whole dataset
- Test CNN classifier on the whole dataset

Specify the pre-trained classification model by setting `model.weights`.
`classification_mode=whole` specifies we are solving the classification without exploiting the segment information.

```bash
# TRAIN classifier on whole images
python train.py task=c classification_mode=whole model.weights=ConvNeXt_Small_Weights.DEFAULT 

# TEST classifier whole images
python test.py task=c classification_mode=whole checkpoint.version=123
```



### **Exp 2**
Classification on the masked dataset:
- Train CNN for segmentation
- test CNN for segmentation
- train CNN classifier on the masked dataset
- test CNN classifier on the masked dataset

<img src="https://github.com/MarcoParola/improve_classifier_via_segment/assets/32603898/028a44df-4ddb-45b6-9df4-5485c30f9b18" alt="drawing" width="280"/>

Specify the pre-trained segmentation model by setting `model_seg`. `classification_mode=masked` specifies we are solving the classification by exploiting the segment information.

The first step of this task is to train a segmentation NN that will be used to generate masks for images in the next step.
```bash
# TRAIN segmentation NN
python train.py task=s model_seg=['fcn', 'deeplab']

# TEST segmentation NN
python test.py task=s model_seg=['fcn', 'deeplab'] checkpoint.version=123
```
After training your segmentation NN insert the version of the model you want to exploit in the masked classification in the `__init__` method of [`src/data/masked_classification/dataset.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/masked_classification/dataset.py).
Specify the pre-trained classification model by setting `model.weights`. Specify the segmentation model previously trained for generate the masks by setting `model_seg`.
```bash
# TRAIN classifier on masked images
python train.py task=c classification_mode=masked model.weights=ConvNeXt_Small_Weights.DEFAULT model_seg=['fcn', 'deeplab'] sgm_type=['soft', 'hard']

# TEST classifier on masked images
python test.py task=c classification_mode=masked model_seg=['fcn', 'deeplab'] checkpoint.version=123
```

### **Exp 3**
Classification on the whole dataset exploiting saliency maps and masks:
- train CNN classifier on the original dataset with backpropagating saliency map error
- test CNN classifier on the whole dataset

<img src="https://github.com/MarcoParola/improve_classifier_via_segment/assets/32603898/b36037ce-553d-49a7-a165-b361ee124ff3" alt="drawing" width="280"/>


Specify the pre-trained classification model by setting `model.weights`.
`classification_mode=saliency` specifies we are solving the classification by exploiting the saliency map information.

```bash
# TRAIN classifier on whole images with saliency map information
python train.py task=c classification_mode=saliency model.weights=ConvNeXt_Small_Weights.DEFAULT 

# TEST classifier on whole images
python test.py task=c classification_mode=saliency checkpoint.version=123
```

This projects exploits [Hydra](https://hydra.cc/) so it's possible both to run experiments with the minimal options as in the aforementioned examples and setting all the others parameters in [`config/config.yaml`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/config/config.yaml) or to run different experiments without changing  [`config/config.yaml`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/config/config.yaml) and overriding the configuration by command line, as example:
```bash
# TRAIN classifier on whole images with saliency map information
python train.py task=c classification_mode=saliency model.weights=ConvNeXt_Small_Weights.DEFAULT

# TRAIN classifier on whole images with saliency map information overriding configuration parameters:
python train.py task='c' classification_mode='saliency' model.weights=ConvNeXt_Small_Weights.DEFAULT train.max_epochs=700 train.lr=0.00001 train.accelerator='gpu' log.tensorboard=True log.wandb=False
```
## Pytorch-lightning logic modules
Since this project is developed using the `pytorch-lightning` framework, two key concepts are `Modules` to declare a new model and `DataModules` to organize of our dataset. Both of them are declared in `src/`, specifically in `src/models/` and `src/data/`, respectively. More information are in the next sections.

### Deep learning models
The three deep learning models are  `src/models/classification.py`,  `src/models/saliency_classification.py` and  `src/models/segmentation.py`.

**`classification.py`** is the model that implements the workflow of the standard classification problem, the model is trained in a supervised fashion and the loss function is the CrossEntropyLoss so experiments which use this model exploits only labels information for backpropagation purposes. This model is exploited both for Experiment 1 and for Experiment 2, the difference among the two experiments lies in the preparation of the data in input to the classifier and will be further explained in the next section. 

**`saliency_classification.py`** is the model exploited in Experiment 3. The core difference, from the model perspective, with respect to `classification.py` is that in this case the loss function is a custom loss defined in this work called CEntIoU which exploits both the information of the labels, in the CrossEntropy component, and of the segment annotations, in the IoU component, which is computed as the Intesection of binarized saliency maps and ground truth masks over their Union. Saliency maps and their IoU with ground truth annotations are computed in the [_common_step](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/models/saliency_classification.py#L130) of this model. 
Saliency maps, which are a core aspect of this model, are computed exploiting HiResCAM object from [Grad-cam](https://github.com/jacobgil/pytorch-grad-cam). In order to do so it's important to adequately choose the target layers (layers of the networks of which the gradients are kept in consideration for the generation of the saliency maps) of each topology otherwise maps will show higher activation on features which are not interesting for our task and in particular which are not comparable with segment informations in order to compute IoU. This works has been developed experimenting with four [torchvision models](https://github.com/pytorch/vision/tree/main/torchvision/models) (Squeezenet, Convnext, ViT and Swin) so target layers are chosen only for those. Saliency maps are then binaryzed with a treshold of 0.5 and passed, coupled with ground truth segment informations, to the loss computation. PyTorch Lightning in the evaluation step automatically disables gradients computation in order to speed up the process, gradients are necessary to compute saliency maps so in the saliency_classification model is specified `torch.set_grad_enabled(True)` at the start of the [_common_step](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/models/saliency_classification.py#L130).
Saliency maps are available in this work also during the test phase, for explainability purposes. Running [test.py](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/test.py) with the option generate_map="grad-cam" or setting it in [config.yaml](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/config/config.yaml) allows to save in the output folder, under the current runtime output folder of [Hydra](https://hydra.cc/), the saliency maps generated for all the images of the test set.

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
