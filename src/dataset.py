import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

class ThyroidDataset(Dataset):

    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))

    def parse_xml(self, xml_path):

        tree = ET.parse(xml_path)
        root = tree.getroot()

        label = int(root.find("object").find("name").text)

        bbox = root.find("object").find("bndbox")

        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        return label, xmin, ymin, xmax, ymax

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        xml_path = os.path.join(
            self.annotation_dir,
            img_name.replace(".jpg", ".xml")
        )

        image = Image.open(img_path).convert("RGB")

        label, xmin, ymin, xmax, ymax = self.parse_xml(xml_path)

    
        image = image.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            image = self.transform(image)

        return image, label