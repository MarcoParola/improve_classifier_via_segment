import cv2
import matplotlib.pyplot as plt
import torch
import os
import json
from PIL import Image, ImageDraw
import numpy as np

from src.models.segmentation import FcnSegmentationNet, DeeplabSegmentationNet
from src.utils import get_last_checkpoint


class OralClassificationMaskedDataset(torch.utils.data.Dataset):
    def __init__(self, sgm_type, segmenter, annonations, transform=None):

        self.annonations = annonations
        self.transform = transform

        with open(annonations, "r") as f:
            self.dataset = json.load(f)

        self.images = dict()
        for image in self.dataset["images"]:
            self.images[image["id"]] = image

        self.categories = dict()
        for i, category in enumerate(self.dataset["categories"]):
            self.categories[category["id"]] = i

        # qui bisogna fissare il migliore fcn ed il migliore deeplab invece di caricare la version
        if segmenter == 'fcn':
            self.segment_model = FcnSegmentationNet.load_from_checkpoint(get_last_checkpoint(136))
            self.segment_model.sgm_type = sgm_type
        elif segmenter == 'deeplab':
            self.segment_model = DeeplabSegmentationNet.load_from_checkpoint(get_last_checkpoint(136))
            self.segment_model.sgm_type = sgm_type

    def __len__(self):
        return len(self.dataset["annotations"])

    def __getitem__(self, idx):
        annotation = self.dataset["annotations"][idx]
        image = self.images[annotation["image_id"]]
        image_path = os.path.join(os.path.dirname(self.annonations), "oral1", image["file_name"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        mask = self.segment_model(image.unsqueeze(0))


        '''
        mask = mask.squeeze()
        mask = mask.detach().numpy().astype(np.uint8)

        mask = np.stack((mask, mask, mask), axis=-1)

        print("mask_shape: ", mask.shape)

        image = image * 255
        image = image.numpy().astype(np.uint8)
        image = image.transpose(2, 1, 0)
        print("image_shape: ", image.shape)
        #masked_image = cv2.bitwise_and(image, image, mask=mask)
        masked_image = image*(1-mask)


        masked_image = masked_image.transpose(2, 1, 0)
        masked_image = torch.tensor(masked_image, dtype=torch.float32)
        '''
        mask = mask.squeeze(0)
        mask = mask.repeat(3, 1, 1)


        masked_image = torch.mul(image, mask)


        category = self.categories[annotation["category_id"]]

        return masked_image, category

    def get_all_data(self, idx):
        annotation = self.dataset["annotations"][idx]
        image = self.images[annotation["image_id"]]
        image_path = os.path.join(os.path.dirname(self.annonations), "oral1", image["file_name"])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        mask = self.segment_model(image.unsqueeze(0))
        mask = mask.squeeze(0)
        mask = mask.repeat(3, 1, 1)

        masked_image = torch.mul(image, mask)

        category = self.categories[annotation["category_id"]]

        return masked_image, category, mask, image


if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms

    import torch

    dataset = OralClassificationMaskedDataset(
        'fcn', "data/train.json",
        transform=transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor()
        ])
    )
    image_masked, label = dataset.__getitem__(8)
    image_masked = image_masked
    image_masked = image_masked.detach().numpy()
    image_masked = image_masked.transpose(2, 1, 0)
    image_masked = (image_masked - np.min(image_masked)) / (np.max(image_masked) - np.min(image_masked))

    plt.imshow(image_masked)
    plt.show()
