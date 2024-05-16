
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

class SidneyCaptions(Dataset):
    splits = ["train", "val", "test"]

    def __init__(self, root, split, transform=None):
        assert split in self.splits

        self.image_root = "images"
        self.root = root
        self.split = split
        self.transform = transform
        captions_file = os.path.join(root, 'filenames', "descriptions_SIDNEY.txt")
        split_file = os.path.join(root, 'filenames', f"filenames_{split}.txt")
        self.captions = self.load_captions(captions_file, split_file)

    @staticmethod
    def load_captions(captions_file: str, split_file: str) -> List[Dict]:
        captions = open(captions_file)
        split = open(split_file)
        samples = []
        split = split.read().splitlines()
        captions = captions.read().splitlines()
        captions = [caption.split() for caption in captions]
        captions = [ {'id': caption[0], 'caption': caption[1:]} for caption in captions ]

        for s in split:
            id = s.split('.')[0]
            c = []
            for caption in captions:
                if caption['id'] == id:
                   c.append(' '.join(caption['caption']).strip())
            samples.append({'filename': s, 'captions': c})
            
        
        return samples

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        sample = self.captions[idx]
        path = os.path.join(self.root, self.image_root, sample["filename"])
        x = Image.open(path).convert("RGB")
        x = self.transform(x)
        return dict(x=x, captions=sample["captions"])

def get_datasets(transform, dataset_names=["rsicd", "ucm", "nwpu", "sydney"]):
    datasets_train = []
    datasets_val = []

    # RSICD datasets
    if "rsicd" in dataset_names and os.path.exists("data/rsicd/RSICD_images/00001.jpg"):
        rscid_dataset_train = RSICD(
            root="data/rsicd/",
            split="train", 
            transform=transform
        )
        datasets_train.append(rscid_dataset_train)

        rscid_dataset_val = RSICD(
            root="data/rsicd/",
            split="val", 
            transform=transform
        )
        datasets_val.append(rscid_dataset_val)
    elif "rsicd" in dataset_names:
        # warn user that RSICD dataset is not available
        print("RSICD dataset is not available")

    # UCM Captions datasets
    if "ucm" in dataset_names and os.path.exists("data/ucm/images/1.tif"):
        ucm_dataset_train = UCMCaptions(
            root="data/ucm/",
            split="train", 
            transform=transform
        )
        datasets_train.append(ucm_dataset_train)

        ucm_dataset_val = UCMCaptions(
            root="data/ucm/",
            split="val", 
            transform=transform
        )
        datasets_val.append(ucm_dataset_val)
    elif "ucm" in dataset_names:
        # warn user that UCM dataset is not available
        print("UCM dataset is not available")

    # NWPUCaptions datasets
    if "nwpu" in dataset_names and os.path.exists("data/nwpu/images/airplane/airplane_001.jpg"):
        nwpucaptions_dataset_train = NWPUCaptions(
            root="data/nwpu/",
            split="train", 
            transform=transform
        )
        datasets_train.append(nwpucaptions_dataset_train)

        nwpucaptions_dataset_val = NWPUCaptions(
            root="data/nwpu/",
            split="val", 
            transform=transform
        )
        datasets_val.append(nwpucaptions_dataset_val)
    elif "nwpu" in dataset_names:
        # warn user that NWPUCaptions dataset is not available
        print("NWPUCaptions dataset is not available")

    # Sidney Captions datasets
    if "sydney" in dataset_names and os.path.exists("data/sidney/images/1.tif"):
        sydney_dataset_train = SidneyCaptions(
            root="data/sidney/",
            split="train", 
            transform=transform
        )
        datasets_train.append(sydney_dataset_train)

        sydney_dataset_val = SidneyCaptions(
            root="data/sidney/",
            split="val", 
            transform=transform
        )
        datasets_val.append(sydney_dataset_val)
    elif "sydney" in dataset_names:
        # warn user that Sidney dataset is not available
        print("Sidney dataset is not available")

    return ConcatDataset(datasets_train), ConcatDataset(datasets_val)

def get_test_datasets(transform):
    datasets = {}

    # RSICD datasets
    if os.path.exists("data/rsicd/RSICD_images/00001.jpg"):
        rscid_dataset = RSICD(
            root="data/rsicd/",
            split="test", 
            transform=transform
        )
        datasets['rsicd'] = rscid_dataset
    else:
        # warn user that RSICD dataset is not available
        print("RSICD dataset is not available")

    # UCM Captions datasets
    if os.path.exists("data/ucm/images/1.tif"):
        ucm_dataset = UCMCaptions(
            root="data/ucm/",
            split="test", 
            transform=transform
        )
        datasets["ucm"] = ucm_dataset
    else:
        # warn user that UCM dataset is not available
        print("UCM dataset is not available")

    # NWPUCaptions datasets
    if os.path.exists("data/nwpu/images/airplane/airplane_001.jpg"):
        nwpucaptions_dataset = NWPUCaptions(
            root="data/nwpu/",
            split="test", 
            transform=transform
        )
        datasets["nwpu"] = nwpucaptions_dataset
    else:
        # warn user that NWPUCaptions dataset is not available
        print("NWPUCaptions dataset is not available")

    # Sidney Captions datasets
    if os.path.exists("data/sidney/images/1.tif"):
        sydney_dataset = SidneyCaptions(
            root="data/sidney/",
            split="test", 
            transform=transform
        )
        datasets["sydney"] = sydney_dataset
    else:
        # warn user that Sidney dataset is not available
        print("Sidney dataset is not available")

    return datasets