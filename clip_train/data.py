import os
import json

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from clip import tokenize


class COCODataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        data_json = os.path.join(root_dir, "annotations", f"captions_{split}2017.json")
        with open(data_json, "r") as json_file:
            self.json_data = json.load(json_file)
            self.img_id_to_filepath = dict()
            self.img_id_to_captions = dict()

        imagedirs = self.json_data["images"]
        self.labels = {"image_ids": list()}
        for imgdir in imagedirs:
            self.img_id_to_filepath[imgdir["id"]] = os.path.join(
                root_dir, f"{split}2017", imgdir["file_name"]
            )
            self.img_id_to_captions[imgdir["id"]] = list()
            self.labels["image_ids"].append(imgdir["id"])

        capdirs = self.json_data["annotations"]
        for capdir in capdirs:
            # there are in average 5 captions per image
            self.img_id_to_captions[capdir["image_id"]].append(np.array(capdir["caption"]))

    def __len__(self):
        return len(self.labels["image_ids"])

    def __getitem__(self, idx):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][idx]]
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        captions = self.img_id_to_captions[self.labels["image_ids"][idx]]
        # randomly draw one of all available captions per image
        caption = captions[np.random.randint(0, len(captions))]
        texts = tokenize([str(caption)])[0]
        return image, texts
