import sys
import json
import os
import pickle

import torch
import torchvision
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import (
    DTD,
    EuroSAT,
    Food101,
    ImageNet,
    INaturalist,
    OxfordIIITPet,
    Places365,
)
from tqdm import tqdm

from escher.cbd_utils import load_clip_model
from escher.cbd_utils.cub import CUBDataset
from escher.cbd_utils.imagenetv2 import openai_imagenet_classes
from escher.cbd_utils.nabirds import NABirds

# TODO(atharvas): This is needed for legacy pickled files cause
# I'm too lazy to repickle legacy files.
sys.modules['dataset_loader'] = sys.modules[__name__]

# TODO(pyuan95) fix inat dataloader
# from datasets.inat.inat_data_loader import load_datasets_jointtrain
load_datasets_jointtrain = None

# TODO(atharvas): Make this a config file
dataset2pth = {
    "imagenet": "/var/local/atharvas/f/raw_datasets/ImageNet",
    "imagenetv2": "/var/local/atharvas/f/raw_datasets/imagenetv2",
    "cub": "/var/local/atharvas/f/raw_datasets/cub",
    "cub_sparrows": "/var/local/atharvas/f/raw_datasets/cub",
    "eurosat": "/var/local/atharvas/f/raw_datasets/eurosat",
    "food101": "/var/local/atharvas/f/raw_datasets/food101",
    "pets": "/var/local/atharvas/f/raw_datasets/pets",
    "dtd": "/var/local/atharvas/f/raw_datasets/d2d",
    "places365": "/var/local/atharvas/f/raw_datasets/places365",
    "inat": "/mnt/sdd3/datasets_atharvas/iNat2021/",
    "nabirds": "/var/local/atharvas/f/raw_datasets/nabirds",
    "flowers": "/var/local/atharvas/f/raw_datasets/flowers",
    "cars": "/var/local/atharvas/f/raw_datasets/cars",
    "cifar100": "/var/local/atharvas/f/raw_datasets/cifar100",
}


def _transform(n_px):
    return transforms.Compose(
        [
            transforms.Resize(n_px, interpolation=Image.BICUBIC),
            transforms.CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def process_class_name(cls_name, dataset):
    if dataset == "places365":
        cls_name = list(reversed(cls_name.split("/")[2:]))
        return " ".join(cls_name).lower().replace("_", " ")
    if dataset == "eurosat":
        # Mapping is taken from "Visual Classification Via Description From Large Language Models" code
        # They "renamed" the classes in this way.
        return {
            "AnnualCrop": "annual crop land",
            "Forest": "forest",
            "HerbaceousVegetation": "brushland or shrubland",
            "Highway": "highway or road",
            "Industrial": "industrial buildings or commercial buildings",
            "Pasture": "pasture land",
            "PermanentCrop": "permanent crop land",
            "Residential": "residential buildings or homes or apartments",
            "River": "river",
            "SeaLake": "lake or sea",
        }[cls_name]

    return cls_name.replace("_", " ").lower()


class MockProcessedDataset:
    def __init__(self, classes, cls2index, index2cls, images, labels):
        self.classes = classes
        self.cls2index = cls2index
        self.index2cls = index2cls
        self.images = images
        self.labels = labels


class ProcessedDataset:
    """
    A dataset where each image has already been encoded with CLIP
    """

    def __init__(
        self,
        dataset,
        clip_model_name,
        dataset_name,
        use_open_clip=False,
        batch_size=512,
        device=0,
    ):
        dataset_name = dataset_name.lower()
        self.cls2index = {
            process_class_name(k, dataset_name): v
            for k, v in dataset.class_to_idx.items()
        }
        self.index2cls = {v: k for k, v in self.cls2index.items()}
        self.classes = list(self.index2cls.values())
        dataloader = DataLoader(
            dataset, batch_size, shuffle=False, num_workers=8, pin_memory=False
        )
        device = torch.device(device)
        # load model
        model, _ = load_clip_model(
            clip_model_name, use_open_clip=use_open_clip, device=device
        )
        model.eval()
        model.requires_grad_(False)
        cache_pth = f"/var/local/atharvas/f/learning_descriptors/cache/{dataset_name}_{clip_model_name.replace('/', '-')}_{int(use_open_clip)}"

        if os.path.exists(f"{cache_pth}_img_encodings.pt"):
            print("Loading image encodings from file")
            self.images = torch.load(
                f"{cache_pth}_img_encodings.pt", weights_only=False
            )
            self.labels = torch.load(f"{cache_pth}_labels.pt", weights_only=False)
            return
        all_image_encodings = []
        all_labels = []
        for _, batch in enumerate(
            tqdm(dataloader, desc=f"getting image embeddings for {dataset_name}")
        ):
            images, labels = batch
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)

                all_image_encodings.append(model.encode_image(images))
                all_labels.append(labels)

        self.images = (
            torch.concatenate(all_image_encodings).detach().to(torch.float32).cpu()
        )
        self.labels = torch.concatenate(all_labels).detach().to(torch.float32).cpu()
        assert len(self.index2cls) == len(self.cls2index)
        torch.save(self.images, f"{cache_pth}_img_encodings.pt")
        torch.save(self.labels, f"{cache_pth}_labels.pt")


def get_dataset(
    clip_model_name, dataset, transform=True, val_only=False, image_size=224
):
    dataset = dataset.lower()
    hparams = {}
    # hyperparameters

    hparams["model_size"] = clip_model_name
    hparams["dataset"] = dataset.lower()

    hparams["batch_size"] = 64 * 10
    hparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    hparams["image_size"] = image_size
    if hparams["model_size"] == "ViT-L/14@336px" and hparams["image_size"] != 336:
        print(
            f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 336.'
        )
        hparams["image_size"] = 336
    elif hparams["model_size"] == "RN50x4" and hparams["image_size"] != 288:
        print(
            f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.'
        )
        hparams["image_size"] = 288
    elif hparams["model_size"] == "RN50x16" and hparams["image_size"] != 384:
        print(
            f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.'
        )
        hparams["image_size"] = 384
    elif hparams["model_size"] == "RN50x64" and hparams["image_size"] != 448:
        print(
            f'Model size is {hparams["model_size"]} but image size is {hparams["image_size"]}. Setting image size to 288.'
        )
        hparams["image_size"] = 448

    # PyTorch datasets
    tfms = _transform(hparams["image_size"]) if transform else lambda x: x
    data_dir = dataset2pth[dataset]
    if dataset == "imagenet":
        dsclass = ImageNet
        val_dataset = dsclass(data_dir, split="val", transform=tfms)
        if val_only:
            return val_dataset
        train_dataset = dsclass(data_dir, split="train", transform=tfms)
        return train_dataset, val_dataset
    elif dataset == "imagenetv2":
        val_dataset = ImageNetV2(
            location=data_dir, transform=tfms, variant="matched-frequency"
        )
        classes_to_load = openai_imagenet_classes
        val_dataset.classes = classes_to_load
        val_dataset.class_to_idx = {
            classes_to_load[i]: i for i in range(len(classes_to_load))
        }
        if val_only:
            return val_dataset
        train_dataset = ImageNetV2(
            location=data_dir, transform=tfms, variant="threshold-0.7"
        )
        train_dataset.classes = classes_to_load
        train_dataset.class_to_idx = {
            classes_to_load[i]: i for i in range(len(classes_to_load))
        }
        return train_dataset, val_dataset
    elif dataset == "cub":
        val_dataset = CUBDataset(root=data_dir, train=False, transform=tfms)
        if val_only:
            return val_dataset
        train_dataset = CUBDataset(root=data_dir, train=True, transform=tfms)
        return train_dataset, val_dataset
    elif dataset == "cub_sparrows":
        val_dataset = CUBDataset(
            root=data_dir, train=False, transform=tfms, keep_str="sparrow"
        )
        if val_only:
            return val_dataset
        train_dataset = CUBDataset(
            root=data_dir, train=True, transform=tfms, keep_str="sparrow"
        )
        return train_dataset, val_dataset
    elif dataset == "eurosat":
        val_dataset = EuroSAT(root=data_dir, transform=tfms, download=True)
        if val_only:
            return val_dataset

        # TODO(pyuan95) create a train/val set here; maybe split it half/half?
        return val_dataset, val_dataset
    elif dataset == "places365":
        val_dataset = Places365(root=data_dir, split="val", transform=tfms)
        if val_only:
            return val_dataset
        train_dataset = Places365(root=data_dir, split="train-standard", transform=tfms)
        return train_dataset, val_dataset

    elif dataset == "food101":
        val_dataset = Food101(
            root=data_dir, split="test", transform=tfms, download=True
        )
        if val_only:
            return val_dataset
        train_dataset = Food101(
            root=data_dir, split="train", transform=tfms, download=True
        )
        return train_dataset, val_dataset

    elif dataset == "pets":
        # hparams['data_dir'] = pathlib.Path(PETS_DIR)
        test_dataset = OxfordIIITPet(
            root=data_dir, split="test", transform=tfms, download=True
        )
        if val_only:
            return test_dataset
        train_dataset = OxfordIIITPet(
            root=data_dir, split="trainval", transform=tfms, download=True
        )
        return train_dataset, test_dataset
    elif dataset == "dtd":
        val_dataset = DTD(root=data_dir, transform=tfms, split="val")
        if val_only:
            return val_dataset
        train_dataset = DTD(root=data_dir, transform=tfms, split="train")
        return train_dataset, val_dataset
    elif dataset == "inat":
        val_dataset = INaturalist(
            root=data_dir, version="2021_valid", transform=tfms, download=False
        )

        descriptors_path = os.path.join(
            "descriptors/cbd_descriptors/descriptors_inat.json"
        )
        with open(descriptors_path, "r") as f:
            descriptors = json.load(f)
            descriptorsid = {
                k.lower().replace(" ", "").split("(")[0]: (k, v)
                for k, v in descriptors.items()
            }
            descriptorsid["malusdomestica"] = descriptorsid["malus×domestica"]

        category_to_descriptor_name = {
            category: "".join(category.split("_")[-2:]).replace("×", "x").lower()
            for category in val_dataset.all_categories
        }

        cls2index = {}
        for category_idx, _ in val_dataset.index:
            category = val_dataset.all_categories[category_idx]
            common_name, _ = descriptorsid[category_to_descriptor_name[category]]
            cls2index[common_name] = category_idx
        val_dataset.class_to_idx = cls2index

        if val_only:
            return val_dataset
        train_dataset = INaturalist(
            root=data_dir, version="2021_train", transform=tfms, download=False
        )

        cls2index = {}
        for category_idx, _ in train_dataset.index:
            category = train_dataset.all_categories[category_idx]
            common_name, _ = descriptorsid[category_to_descriptor_name[category]]
            cls2index[common_name] = category_idx
        train_dataset.class_to_idx = cls2index
        return train_dataset, val_dataset
    elif dataset == "nabirds":
        val_dataset = NABirds(
            root=data_dir, train=False, transform=tfms, download=False
        )
        descriptors_path = "descriptors/cbd_descriptors/descriptors_nabirds.json"
        with open(descriptors_path, "r") as f:
            all_descriptors = json.load(f)

        class_to_index = {}
        for idx, cls in enumerate(all_descriptors.keys()):
            class_to_index[cls] = idx

        val_dataset.class_to_idx = class_to_index

        # This is equvilant to the class_to_index generated with the following code:
        # import re
        # descriptors_path = '/var/local/atharvas/f/raw_datasets/nabirds/nabirds/cbd_descriptors.json'
        # with open(descriptors_path, 'r') as f:
        #     all_descriptors = json.load(f)

        # all_labels = set(val_dataset.data['target'].map(val_dataset.label_map))
        # observed_classes = [val_dataset.class_names[str(i)] for i in all_labels]
        # # remove duplicates in observed classes by adding the parent class name to the child class name
        # # if the parent class is unique
        # class_names = val_dataset.class_names
        # observed_class_names = {}
        # for idx, class_name in class_names.items():
        #     if class_name in observed_classes and observed_classes.count(class_name) > 1:
        #         parent = val_dataset.class_names[str(val_dataset.class_hierarchy[str(idx)])]
        #         observed_class_names[idx] = class_name + f" ({parent})"
        #     else:
        #         observed_class_names[idx] = class_name

        # all_observed_idxs = set(map(str, val_dataset.data.target.values))
        # renamed_all_descriptors = {}
        # class_to_index = {}
        # for idx, cls in observed_class_names.items():
        #     if idx not in all_observed_idxs:
        #         continue
        #     if cls in all_descriptors:
        #         # all_descriptors[cls] = all_descriptors.pop(str(idx))
        #         renamed_all_descriptors[cls] = all_descriptors[cls]
        #         class_to_index[cls] = val_dataset.label_map[int(idx)]
        #     else:
        #         og_cls, parent = re.match(r"(.*) \((.*)\)", cls).groups()
        #         merged_descriptors = all_descriptors[og_cls] + all_descriptors[parent]
        #         renamed_all_descriptors[cls] = merged_descriptors
        #         class_to_index[cls] = val_dataset.label_map[int(idx)]
        # In [109]: class_to_index == class_to_index2
        # Out[109]: True

        if val_only:
            return val_dataset
        train_dataset = NABirds(
            root=data_dir, train=True, transform=tfms, download=False
        )
        train_dataset.class_to_idx = class_to_index

        return train_dataset, val_dataset
    elif dataset == "flowers":
        val_dataset = torchvision.datasets.Flowers102(
            root=data_dir, split="val", transform=tfms, download=True
        )
        with open(f"{val_dataset.root}/flowers-102/labels.txt", "r") as f:
            labels = f.readlines()
            labels = [label.strip().strip("'") for label in labels]
        val_dataset.class_to_idx = {label: idx for idx, label in enumerate(labels)}
        if val_only:
            return val_dataset
        train_dataset = torchvision.datasets.Flowers102(
            root=data_dir, split="train", transform=tfms, download=True
        )
        train_dataset.class_to_idx = {label: idx for idx, label in enumerate(labels)}
        return train_dataset, val_dataset
    elif dataset == "cars":
        val_dataset = torchvision.datasets.StanfordCars(
            root=data_dir, split="test", transform=tfms, download=False
        )
        if val_only:
            return val_dataset
        train_dataset = torchvision.datasets.StanfordCars(
            root=data_dir, split="train", transform=tfms, download=False
        )
        return train_dataset, val_dataset
    elif dataset == "cifar100":
        val_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, transform=tfms, download=True
        )
        if val_only:
            return val_dataset
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, transform=tfms, download=True
        )
        return train_dataset, val_dataset


def get_processed_dataset(
    clip_model_name,
    dataset_name,
    image_size=224,
    device=0,
    use_open_clip=False,
    only_return_val=False,
):
    name = os.path.join(
        "/var/local/atharvas/f/learning_descriptors/cache",
        f"processed_dataset_{clip_model_name.replace('/', '-') + ('open' if use_open_clip else '')}_{dataset_name}{{split_str}}.pkl",
    )
    train_name = name.format(split_str="train")
    val_name = name.format(split_str="val")

    if os.path.exists(val_name):
        with open(val_name, "rb") as f:
            processed_val_dataset = pickle.load(f)
    else:
        val_dataset = get_dataset(
            clip_model_name, dataset_name, image_size=image_size, val_only=True
        )
        processed_val_dataset = ProcessedDataset(
            val_dataset,
            clip_model_name,
            dataset_name,
            use_open_clip=use_open_clip,
            device=device,
        )
        with open(val_name, "wb") as f:
            pickle.dump(processed_val_dataset, f)

    if only_return_val:
        return processed_val_dataset

    if os.path.exists(train_name):
        with open(train_name, "rb") as f:
            processed_train_dataset = pickle.load(f)
    else:
        train_dataset, val_dataset = get_dataset(
            clip_model_name, dataset_name, image_size=image_size, val_only=False
        )
        processed_train_dataset = ProcessedDataset(
            train_dataset,
            clip_model_name,
            dataset_name,
            use_open_clip=use_open_clip,
            device=device,
        )
        with open(train_name, "wb") as f:
            pickle.dump(processed_train_dataset, f)

    # eurosat doesn't have a real val set
    # if only_return_val then the whole dataset is returned
    # otherwise, we split it
    if dataset_name == "eurosat":
        processed_val_dataset = ProcessedDataset(
            val_dataset,
            clip_model_name,
            dataset_name,
            use_open_clip=use_open_clip,
            device=device,
        )

    train_images, test_images, train_labels, test_labels = train_test_split(
        processed_train_dataset.images.numpy(),
        processed_train_dataset.labels.numpy(),
        test_size=0.1,
        random_state=123,
        stratify=processed_train_dataset.labels,
    )

    mock_processed_train_dataset = MockProcessedDataset(
        processed_train_dataset.classes,
        processed_train_dataset.cls2index,
        processed_train_dataset.index2cls,
        torch.from_numpy(train_images),
        torch.from_numpy(train_labels),
    )
    mock_processed_test_dataset = MockProcessedDataset(
        processed_train_dataset.classes,
        processed_train_dataset.cls2index,
        processed_train_dataset.index2cls,
        torch.from_numpy(test_images),
        torch.from_numpy(test_labels),
    )

    return (
        mock_processed_train_dataset,
        mock_processed_test_dataset,
        processed_val_dataset,
    )
