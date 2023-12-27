import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


class OralSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, annonations, transform=None):
        """
        Args:
            annonations (string): Path to the json annonations file with coco json file.
            transform (callable, optional): Optional transform to be appliedon a sample.
        """
        self.annonations = annonations
        self.transform = transform

        with open(annonations, "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        # retrieve image
        image_id = self.dataset["images"][idx]["id"]
        segments = [element["segmentation"] for element in self.dataset["annotations"] if
                    element["image_id"] == image_id]

        lengths = [len(segment) for segment in segments[0]]
        segments = [segment for segment in segments[0] if len(segment) == max(lengths)]
        image_path = os.path.join(os.path.dirname(self.annonations), "oral1", self.dataset["images"][idx]["file_name"])
        image = Image.open(image_path).convert("RGB")

        # generate mask
        width, height = image.size
        mask = Image.new('L', (width, height))
        for segment in segments:
            ImageDraw.Draw(mask).polygon(segment, outline=1, fill=1)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # scale and repeat mask on all channels
        mask = mask / mask.max()
        return image, mask