import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
import lime
from lime import lime_image

import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.segmentation import mark_boundaries


class OralLime:

    def create_maps_lime(model, dataloader, predictions):
        for batch_index, (images, _) in enumerate(dataloader):
            for image_index, image in enumerate(images):
                print(len(image))
                print(type(image))
                image_np = image.cpu().numpy()
                print("1")

                # Transpose the channels to be in the order of RGB
                image_np = np.transpose(image_np, (1, 2, 0))

                # Normalize pixel values
                image_np = image_np / 255.0

                # from numpy array to torch tensor
                #image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()

                # from torch tensor to pil
                #image_pil = transforms.ToPILImage()(image_tensor)

                # from pil image to numpy array
                #image_np = np.array(image_pil)

                predicted_class = predictions[batch_index * len(images) + image_index].item()

                # custom classifier function that works with numpy arrays
                def classifier_fn(images):
                    # Convert NumPy array to PyTorch tensor
                    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
                    # Make predictions using the model
                    return model.model(images).detach().cpu().numpy()

                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(image_np,
                                                         classifier_fn,
                                                         labels=predicted_class,  num_samples=150)
                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,
                                                            num_features=1, hide_rest=False)
                img_boundry1 = mark_boundaries(temp / 255.0, mask)
                img_boundry1 = Image.fromarray((img_boundry1 * 255).astype(np.uint8))

                fig, ax = plt.subplots()
                ax.imshow(img_boundry1)
                ax.imshow(image.permute(1, 2, 0).numpy(), alpha=0.5)


                ax.axis('off')  # Turn off axis labels
                plt.savefig(os.path.join('test_lime', f"lime_explanation_{batch_index}_{image_index}_positive.png"))


                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False,
                                                            num_features=1, hide_rest=False)
                img_boundry2 = mark_boundaries(temp / 255.0, mask)
                img_boundry2 = Image.fromarray((img_boundry2 * 255).astype(np.uint8))
                fig, ax = plt.subplots()
                ax.imshow(img_boundry2)
                ax.imshow(image.permute(1, 2, 0).numpy(), alpha=0.5)
                ax.axis('off')  # Turn off axis labels
                plt.savefig(os.path.join('test_lime', f"lime_explanation_{batch_index}_{image_index}_negative.png"))
                break
