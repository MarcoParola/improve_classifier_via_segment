import cv2
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import os
import hydra
import matplotlib.pyplot as plt

import torch
from torchsummary import summary

class OralGradCam:

    def generate_saliency_maps_grad_cam(model, dataloader, predictions, classification_mode):
        # put in evaluation mode the model
        model.eval()
        if "vit" in model.model_name:
            target_layers = [model.model.conv_proj]
        elif "convnext" in model.model_name:
            target_layers = [model.model.features[-1][-1]]
        # swin è da cercare meglio il target layer
        elif "swin" in model.model_name:
            target_layers = [model.model.features[0][0]]
        elif "squeezenet" in model.model_name:
            target_layers = [model.model.features[-1]]

        cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=False)

        # iterate the dataloader passed
        if classification_mode == 'saliency':
            for batch_index, (images, _, _) in enumerate(dataloader):
                for image_index, image in enumerate(images):
                    # this is needed to work with a single image
                    input_tensor = image.unsqueeze(0)
                    # get the label predicted for current image
                    predicted_class = predictions[batch_index * len(images) + image_index].item()
                    # use the predicted label as target for grad-cam
                    targets = [ClassifierOutputTarget(predicted_class)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    #visualization = show_cam_on_image(image.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
                    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
                    image_for_plot = image.permute(1, 2, 0).numpy()
                    fig, ax = plt.subplots()
                    ax.imshow(image_for_plot)
                    ax.imshow((grayscale_cam * 255).astype('uint8'), cmap='jet', alpha=0.75)

                    # put the generated map over the starting image
                    #visualization_image = Image.fromarray((visualization * 255).astype(np.uint8))
                    # create the folder in which save the images
                    os.makedirs(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps_test', exist_ok=True)
                    #visualization_image.save(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps_test/saliency_map_batch_{batch_index}_image_{image_index}.jpg')
                    plt.savefig(os.path.join(
                        f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps_test/saliency_map_batch_{batch_index}_image_{image_index}.pdf'),
                        bbox_inches='tight')
                    plt.close()

        elif classification_mode == 'whole' or classification_mode == 'masked':
            for batch_index, (images, _) in enumerate(dataloader):
                for image_index, image in enumerate(images):
                    # this is needed to work with a single image
                    input_tensor = image.unsqueeze(0)
                    # get the label predicted for current image
                    predicted_class = predictions[batch_index * len(images) + image_index].item()
                    # use the predicted label as target for grad-cam
                    targets = [ClassifierOutputTarget(predicted_class)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    # visualization = show_cam_on_image(image.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)
                    grayscale_cam = cv2.resize(grayscale_cam, (224, 224))
                    image_for_plot = image.permute(1, 2, 0).detach().numpy()
                    fig, ax = plt.subplots()
                    ax.imshow(image_for_plot)
                    ax.imshow((grayscale_cam * 255).astype('uint8'), cmap='jet', alpha=0.75)

                    # put the generated map over the starting image
                    # visualization_image = Image.fromarray((visualization * 255).astype(np.uint8))
                    # create the folder in which save the images
                    os.makedirs(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps_test',
                                exist_ok=True)
                    # visualization_image.save(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps_test/saliency_map_batch_{batch_index}_image_{image_index}.jpg')
                    plt.savefig(os.path.join(
                        f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps_test/saliency_map_batch_{batch_index}_image_{image_index}.pdf'),
                        bbox_inches='tight')
                    plt.close()

