# **Oral_classification_segment**
Github repo to improve classification performance by exploiting segment information

## **Install**

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```bash
python -m venv env
env/Scripts/activate
python -m pip install -r requirements.txt
mkdir data
```
Then you can download the oral coco-dataset (both images and json file) from TODO-put-link. Copy them into `data` folder and unzip the file `oral1.zip`.

## **Usage**
In order to reproduce the experiments, we organize the workflow in 2 parts: 
- Data preparation and visualization
- Deep learning experiments

### Data preparation and visualization
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

## Deep learning experiments

All the experiments consist of train and test classification and segmentation architectures. You can use `train.py` and `test.py`, respectively. Where you have to set the `task` option to specify the task your architecture are solving:
- `task=c` or `task=classification` to run classification experiments 
- `task=s` or `task=segmentation` to run segmentation experiments 

```bash
python train.py task=...
python test.py task=...
```

### **Exp 1** (baseline)
Classification on whole dataset:

- Train CNN classifier on whole dataset
- Test CNN classifier on whole dataset

Specify the pretrained classification model by setting `model.weights`.
`classification_mode=whole` is to specify we are solving the classification without exploiting the segment information.

```bash
# TRAIN classifier on whole images
python train.py task=c classification_mode=whole model.weights=ResNet50_Weights.IMAGENET1K_V2 

# TEST classifier whole images
python test.py task=c classification_mode=whole ...
```



### **Exp 2**
Classification on masked dataset:

- Train CNN for segmentation
- test CNN for segmentation
- train CNN classifier on masked dataset
- test CNN classifier on masked dataset

Specify the pretraine segmentation model by setting `model_seg.weights`. `classification_mode=masked` is to specify we are solving the classification exploiting the segment information.

```bash
# TRAIN segmentation NN
python train.py task=s model_seg.weights=...

# TEST segmentation NN
python test.py task=s model_seg.weights=...
```

Specify the pretrained classification model by setting `model.weights`. Specify the segmentation model previoulsy trained for generate the masks by setting `model_seg.weights`.
```bash
# TRAIN classifier on masked images
python train.py task=c classification_mode=masked model.weights=...

# TEST classifier on masked images
python test.py task=c classification_mode=masked model.weights=...
```

### **Exp 3**
Classification on whole dataset exploiting saliency maps and masks:
- train CNN classifier on original dataset with backpropagating saliency map error
- test CNN classifier on whole dataset

Specify the pretrained classification model by setting `model.weights`.
`classification_mode=saliency` is to specify we are solving the classification exploiting the saliency map information.

```bash
# TRAIN classifier on whole images with saliency map information
python train.py task=c classification_mode=saliency model.weights=ResNet50_Weights.IMAGENET1K_V2 

# TEST classifier on whole images
python test.py task=c classification_mode=saliency ...
```






### Visualize logs

```bash
python -m tensorboard.main --logdir=logs
```
