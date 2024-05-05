
from torchrs.datasets import RSICD, UCMCaptions
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset
import json
import os
from typing import List, Dict
from PIL import Image

# datasets: https://github.com/201528014227051/RSICD_optimal?tab=readme-ov-file


class NWPUCaptions(Dataset):
    splits = ["train", "val", "test"]

    def __init__(self, root, split, transform=None):
        assert split in self.splits

        self.image_root = "images"
        self.root = root
        self.split = split
        self.transform = transform
        self.captions = self.load_captions(os.path.join(root, "dataset_nwpu.json"), split)

    @staticmethod
    def load_captions(path: str, split: str) -> List[Dict]:
        with open(path) as f:
            file = json.load(f)
            captions = []
            for category in file.keys():
                for sample in file[category]:
                    if sample["split"] == split:
                        sample["category"] = category
                        captions.append(sample)
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        sample = self.captions[idx]
        path = os.path.join(self.root, self.image_root, sample["category"], sample["filename"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        sentences = []
        for key in sample.keys():
            # if key starts with "raw" then it is a caption
            if key.startswith("raw"):      
                sentences.append(sample[key])
        return dict(x=x, captions=sentences)



def get_datasets(transform):

    # RSICD datasets
    rscid_dataset_train = RSICD(
        root="data/rsicd/",
        split="train", 
        transform=transform
    )
    rscid_dataset_val = RSICD(
        root="data/rsicd/",
        split="val", 
        transform=transform
    )
    rscid_dataset_test = RSICD(
        root="data/rsicd/",
        split="test", 
        transform=transform
    )

    # UCM Captions datasets
    ucm_dataset_train = UCMCaptions(
        root="data/ucm/",
        split="train", 
        transform=transform
    )
    ucm_dataset_val = UCMCaptions(
        root="data/ucm/",
        split="val", 
        transform=transform
    )
    ucm_dataset_test = UCMCaptions(
        root="data/ucm/",
        split="test", 
        transform=transform
    )

    # NWPUCaptions datasets
    nwpucaptions_dataset_train = NWPUCaptions(
        root="data/nwpu/",
        split="train", 
        transform=transform
    )
    nwpucaptions_dataset_val = NWPUCaptions(
        root="data/nwpu/",
        split="val", 
        transform=transform
    )
    nwpucaptions_dataset_test = NWPUCaptions(
        root="data/nwpu/",
        split="test", 
        transform=transform
    )

    dataset_train = ConcatDataset([rscid_dataset_train, ucm_dataset_train, nwpucaptions_dataset_train])
    dataset_val = ConcatDataset([rscid_dataset_val, ucm_dataset_val, nwpucaptions_dataset_val])
    dataset_test = ConcatDataset([rscid_dataset_test, ucm_dataset_test, nwpucaptions_dataset_test])
    return dataset_train, dataset_val, dataset_test