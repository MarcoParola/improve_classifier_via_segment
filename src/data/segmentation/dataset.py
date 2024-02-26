import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageOps
import hydra



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
            seed = torch.randint(0, 100000, (1,)).item()
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        # scale and repeat mask on all channels
        mask = mask / mask.max()

        image = self.fill_black(image)

        return image, mask


    
    def fill_black(self, image):
        np_image = transforms.ToPILImage()(image)
        np_image = np.array(np_image)
        black_pixels = np.all(np_image == [0, 0, 0], axis=-1)
        for i in range(3):  
            np_image[..., i][black_pixels] = np_image[..., i][~black_pixels].mean()
        filled_image = Image.fromarray(np_image)
        filled_tensor = transforms.ToTensor()(filled_image)
        return filled_tensor



@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg):
    import sys
    sys.path.append(".")
    from src.utils import get_transformations

    torch.manual_seed(42)
    train_img_tranform, val_img_tranform, test_img_tranform, img_tranform = get_transformations(cfg)
    train_dataset = OralSegmentationDataset(cfg.dataset.train, transform=train_img_tranform)
    for i in range(2):

        img, mask = train_dataset.__getitem__(i)
        plt.imshow(img.permute(1, 2, 0))
        plt.imshow(mask.permute(1, 2, 0), alpha=.4, cmap='gray')
        plt.show()

if __name__ == '__main__':
    main()
