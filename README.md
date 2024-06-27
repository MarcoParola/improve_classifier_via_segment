# **Improve classifier via segment**
[![size](https://img.shields.io/github/languages/code-size/MarcoParola/improve_classifier_via_segment?style=plastic)]()
[![license](https://img.shields.io/static/v1?label=OS&message=Linux&color=green&style=plastic)]()
[![Python](https://img.shields.io/static/v1?label=Python&message=3.10&color=blue&style=plastic)]()

Github repo to improve classification performance by exploiting segment information.
[Here](https://marcoparola.github.io/improve_classifier_via_segment/) the project web page

![cent_vs_centIou](https://github.com/MarcoParola/improve_classifier_via_segment/assets/32603898/5b7f4460-4a60-4f29-8c52-5de3f95970b8)

## **Install**

To install the project, simply clone the repository and get the necessary dependencies:
```sh
git clone https://github.com/MarcoParola/improve_classifier_via_segment.git
cd improve_classifier_via_segment
```

Create the virtualenv (you can also use conda) and install the dependencies of *requirements.txt*

```bash
python -m venv env
. env/bin/activate
python -m pip install -r requirements.txt
mkdir data
```
Then you can download the oral coco-dataset (both images and json file) from TODO-put-link. Copy them into `data` folder and unzip the file `oral1.zip`.

Next, create a new project on [Weights & Biases](https://wandb.ai/site) named `improve_classifier_via_segment`. Edit `entity` parameter in [config.yaml](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/config/config.yaml#L42) by sett. Log in and paste your API key when prompted.
```sh
wandb login 
```

## **Usage**

Here is a quick overview of the main use of the repo. Further information is available in the [official doc](doc/README.md).

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
python train.py task=s model_seg='fcn'

# TEST segmentation NN
python test.py task=s model_seg='fcn' checkpoint.version=123
```
After training your segmentation NN insert the version of the model you want to exploit in the masked classification in the `__init__` method of [`src/data/masked_classification/dataset.py`](https://github.com/MarcoParola/improve_classifier_via_segment/blob/main/src/data/masked_classification/dataset.py).
Specify the pre-trained classification model by setting `model.weights`. Specify the segmentation model previously trained for generate the masks by setting `model_seg`.
```bash
# TRAIN classifier on masked images
python train.py task=c classification_mode=masked model.weights=ConvNeXt_Small_Weights.DEFAULT model_seg='fcn' sgm_type='soft'

# TEST classifier on masked images
python test.py task=c classification_mode=masked model_seg='fcn' checkpoint.version=123
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

# TEST classifier on whole images with saliency map information
python test.py task=c classification_mode=saliency checkpoint.version=123
```


### Visualize logs

```bash
python -m tensorboard.main --logdir=logs
```
