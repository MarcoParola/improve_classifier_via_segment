import torch
import os
import json
from PIL import Image, ImageDraw


class OralClassificationSaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, annonations, transform=None):
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

    def __len__(self):
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        # retrieve image
        image_id = self.dataset["images"][idx]["id"]
        segments = [element["segmentation"] for element in self.dataset["annotations"] if
                    element["image_id"] == image_id]
        annotation = self.dataset["annotations"][idx]
        image = self.images[annotation["image_id"]]

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
            seed = torch.randint(0, 100000, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)


        # scale and repeat mask on all channels
        mask = mask / mask.max()

        category = self.categories[annotation["category_id"]]

        return image, category, mask
