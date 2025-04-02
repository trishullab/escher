"""
evaluate zero-shot performance
"""

import json
import os
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader

from escher.lm4cv.dataset import ImagenetA


def load_yaml(pth):
    with open(pth, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


def clean_label(true_labels):
    true_labels = np.array(true_labels)
    if np.min(true_labels) > 0:
        true_labels -= np.min(true_labels)
    return true_labels


def get_labels(dataset, shots=None):
    if dataset == "cub":
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/cub/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)
        train_test_split = pd.read_csv(
            os.path.join(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
                "cub",
                "train_test_split.txt",
            ),
            sep=" ",
            names=["img_id", "is_training_img"],
        )
        train_test_split = train_test_split["is_training_img"].values
        train_indices = np.where(train_test_split == 1)
        test_indices = np.where(train_test_split == 0)
        train_labels, test_labels = (
            true_labels[train_indices],
            true_labels[test_indices],
        )

    elif dataset == "cifar100":
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/cifar-100-python/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)
        train_labels, test_labels = true_labels[:-10000], true_labels[-10000:]

    elif dataset == "cifar10" or dataset == "cifar10-p":
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/cifar-10-batches-py/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)
        train_labels, test_labels = true_labels[:-10000], true_labels[-10000:]

    elif dataset == "food":
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/food-101/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)
        train_labels, test_labels = true_labels[:-25250], true_labels[-25250:]

    elif dataset == "flower":
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/flowers-102/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)

        train_labels, test_labels = true_labels[:-1020], true_labels[-1020:]

    elif dataset == "cars":

        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/stanford_cars/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 8144 + 8041
        train_labels, test_labels = true_labels[:8144], true_labels[-8041:]

    elif dataset == "imagenet":
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/imagenet/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 50000 + 1281167

        train_labels, test_labels = true_labels[:1281167], true_labels[-50000:]

    elif dataset == "imagenet-a":
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/imagenet/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 50000 + 1281167

        train_labels, test_labels = true_labels[:1281167], true_labels[-50000:]

        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        def filter_labels(labels):
            idxes = np.where((labels < 398) & (labels != 69))
            return labels[idxes]

        train_labels = filter_labels(train_labels)

        train_labels[np.where(train_labels > 69)] -= 1

        testset = ImagenetA(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/imagenet-a"
        )
        test_labels = testset.labels

    elif dataset == "imagenet-animal":
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/imagenet/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 50000 + 1281167

        train_labels, test_labels = true_labels[:1281167], true_labels[-50000:]

        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        def filter_labels(labels):
            idxes = np.where((labels < 398) & (labels != 69))
            return labels[idxes]

        train_labels = filter_labels(train_labels)
        test_labels = filter_labels(test_labels)

        train_labels[np.where(train_labels > 69)] -= 1
        test_labels[np.where(test_labels > 69)] -= 1

    elif dataset == "oxford_pets":
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/oxford-iiit-pet/image_class_labels.txt",
            "r",
        ) as file:
            true_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 3680 + 3669

        train_labels, test_labels = true_labels[:3680], true_labels[-3669:]

    elif dataset == "places365":
        # import IPython; IPython.embed()
        # data/places365/places365_val.txt
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/places365/places365_val.txt",
            "r",
        ) as file:
            test_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]

        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/places365/places365_train_standard.txt",
            "r",
        ) as file:
            train_labels = [
                eval(line.split(" ")[1]) for line in file.read().strip().split("\n")
            ]

    elif dataset == "inat":
        trainset = torchvision.datasets.INaturalist(
            root="data/inat/", version="2021_train", transform=None, download=False
        )
        testset = torchvision.datasets.INaturalist(
            root="data/inat/", version="2021_valid", transform=None, download=False
        )

        train_labels = list(map(lambda x: x[0], trainset.index))
        test_labels = list(map(lambda x: x[0], testset.index))

    elif dataset == "nabirds":
        from dataset import NABirds

        trainset = NABirds(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/nabirds",
            train=True,
            download=False,
        )
        testset = NABirds(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/nabirds",
            train=False,
            download=False,
        )
        train_labels = trainset.data["target"].map(trainset.label_map).values.tolist()
        test_labels = testset.data["target"].map(testset.label_map).values.tolist()
    else:
        raise NotImplementedError

    return train_labels, test_labels


def get_image_dataloader(
    dataset, preprocess, preprocess_eval=None, shuffle=False, shots=None
):
    if shots is not None:
        train_labels, _ = get_labels(dataset, shots=None)

        labels2idx = defaultdict(list)
        for idx, (label) in enumerate(train_labels):
            labels2idx[label].append(idx)

        # subsample shots samples from each class
        for label in labels2idx:
            labels2idx[label] = np.random.choice(
                labels2idx[label], shots, replace=False
            )
        idxes = list(chain(*labels2idx.values()))
    else:
        idxes = None

    if dataset == "cub":
        # Load dataset
        from dataset import Cub2011

        train_dataset = Cub2011(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            mode="train",
            transform=preprocess,
        )
        if idxes is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, idxes)
        test_dataset = Cub2011(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            mode="test",
            transform=preprocess,
        )
        classes = (
            test_dataset.data.filepath.sort_values()
            .apply(
                lambda x: x.split("/")[0]
                .split(".")[1]
                .strip()
                .replace("_", " ")
                .strip()
            )
            .unique()
            .tolist()
        )
        train_dataset.classes = classes
        test_dataset.classes = classes
        print("Train dataset:", len(train_dataset))
        print("Test dataset:", len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=96, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)

    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data",
            train=True,
            download=True,
            transform=preprocess,
        )
        if idxes is not None:
            trainset = torch.utils.data.Subset(trainset, idxes)
        testset = torchvision.datasets.CIFAR100(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data",
            train=False,
            download=True,
            transform=preprocess,
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False
        )

    elif dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data",
            train=True,
            download=True,
            transform=preprocess,
        )
        if idxes is not None:
            trainset = torch.utils.data.Subset(trainset, idxes)
        testset = torchvision.datasets.CIFAR10(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data",
            train=False,
            download=True,
            transform=preprocess,
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False
        )

    elif dataset == "food":
        trainset = torchvision.datasets.Food101(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            split="train",
            download=True,
            transform=preprocess,
        )
        if idxes is not None:
            trainset = torch.utils.data.Subset(trainset, idxes)
        testset = torchvision.datasets.Food101(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            split="test",
            download=True,
            transform=preprocess,
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False
        )

    elif dataset == "flower":
        trainset = torchvision.datasets.Flowers102(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            split="train",
            download=True,
            transform=preprocess,
        )
        if idxes is not None:
            trainset = torch.utils.data.Subset(trainset, idxes)
        testset = torchvision.datasets.Flowers102(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            split="val",
            download=True,
            transform=preprocess,
        )

        if not os.path.exists(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/flowers-102/image_class_labels.txt"
        ):
            import requests

            url = "https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt"
            r = requests.get(url)
            with open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/flowers-102/dataset_labels.txt",
                "w",
            ) as f:
                f.write(r.text)
        with open(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/flowers-102/image_class_labels.txt",
            "r",
        ) as file:
            classes = [
                eval(line.split(" ")[0]) for line in file.read().strip().split("\n")
            ]

        trainset.classes = classes
        testset.classes = classes
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False
        )

    elif dataset == "cars":
        trainset = torchvision.datasets.StanfordCars(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            split="train",
            download=True,
            transform=preprocess,
        )
        if idxes is not None:
            trainset = torch.utils.data.Subset(trainset, idxes)
        testset = torchvision.datasets.StanfordCars(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            split="test",
            download=True,
            transform=preprocess,
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False
        )

    elif (
        dataset == "imagenet" or dataset == "imagenet-animal" or dataset == "imagenet-a"
    ):
        trainset = torchvision.datasets.ImageNet(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/imagenet",
            split="train",
            transform=preprocess,
        )
        testset = torchvision.datasets.ImageNet(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/imagenet",
            split="val",
            transform=preprocess,
        )

        if dataset == "imagenet-animal" or dataset == "imagenet-a":

            def filter_dataset(dataset):
                targets = np.array(dataset.targets)
                idxes = np.where((targets < 398) & (targets != 69))
                dataset.targets = targets[idxes].tolist()
                dataset.samples = [dataset.samples[i] for i in idxes[0]]

            filter_dataset(trainset)
            filter_dataset(testset)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=False
        )

        if dataset == "imagenet-a":
            testset = ImagenetA(
                root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/imagenet-a",
                preprocess=preprocess,
            )

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False
        )

    elif dataset == "oxford_pets":
        trainset = torchvision.datasets.OxfordIIITPet(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            split="trainval",
            transform=preprocess,
            download=True,
        )
        testset = torchvision.datasets.OxfordIIITPet(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/",
            split="test",
            transform=preprocess,
            download=True,
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False
        )

    elif dataset == "places365":
        trainset = torchvision.datasets.Places365(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/places365",
            split="train-standard",
            transform=preprocess,
            download=False,
        )
        testset = torchvision.datasets.Places365(
            root="/var/local/atharvas/f/learning_descriptors/LM4CV/data/places365",
            split="val",
            transform=preprocess,
            download=False,
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False
        )

    elif dataset == "inat":
        trainset = torchvision.datasets.INaturalist(
            root="data/inat/",
            version="2021_train",
            transform=preprocess,
            download=False,
        )
        testset = torchvision.datasets.INaturalist(
            root="data/inat/",
            version="2021_valid",
            transform=preprocess,
            download=False,
        )

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=1024, num_workers=8, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=1024, num_workers=8, shuffle=False
        )

    elif dataset == "nabirds":
        from dataset import NABirds

        trainset = NABirds(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/nabirds",
            train=True,
            download=False,
            transform=preprocess,
        )
        testset = NABirds(
            "/var/local/atharvas/f/learning_descriptors/LM4CV/data/nabirds",
            train=False,
            download=False,
            transform=preprocess,
        )

        with open("cls2attributes/nabirds_cls2attributes.json", "r") as f:
            cls2attributes = json.load(f)
            classes = list(cls2attributes.keys())
        trainset.classes = classes
        testset.classes = classes

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=512, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=512, shuffle=False
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    return train_loader, test_loader


def get_output_dim(dataset, shots=None):
    # return len(np.unique(get_labels(dataset)[0]))
    return len(np.unique(get_labels(dataset, shots=shots)[0]))


def get_folder_name(dataset):
    if dataset == "cub":
        return "cub"
    elif dataset == "cifar100":
        return "cifar-100-python"
    elif dataset == "cifar10":
        return "cifar-10-batches-py"
    elif dataset == "flower":
        return "flowers-102"
    elif dataset == "food":
        return "food-101"
    elif dataset == "imagenet" or dataset == "imagenet-animal":
        return "imagenet"
    elif dataset == "imagenet-a":
        return "imagenet-a"
    elif dataset == "cars":
        return "stanford_cars"
    elif dataset == "oxford_pets":
        return "oxford-iiit-pet"
    elif dataset == "waterbirds":
        return "waterbird_complete95_forest2water2"
    elif dataset == "places365":
        return "places365"
    elif dataset == "inat":
        return "inat"
    elif dataset == "nabirds":
        return "nabirds"
    else:
        raise NotImplementedError


def get_attributes(cfg):
    if "attributes" in cfg and isinstance(cfg["attributes"], list):
        return cfg["attributes"]
    if "attributes_pth" in cfg:
        print("Using attributes from file:", cfg["attributes_pth"])
        with open(cfg["attributes_pth"], "r") as file:
            attributes = file.read().strip().split("\n")
        return attributes

    if cfg["attributes"] == "random":
        """
        Generate random attributes
        """
        import random
        import urllib.request

        word_url = "https://www.mit.edu/~ecprice/wordlist.10000"
        response = urllib.request.urlopen(word_url)
        long_txt = response.read().decode()
        word_list = long_txt.splitlines()

        print(len(word_list))

        random_words = []
        for i in range(512):
            words = random.choices(word_list, k=random.randint(1, 5))
            random_words.append(" ".join(words))

        attributes = random_words
        return attributes

    elif cfg["attributes"] == "cub":
        # return open("/var/local/atharvas/f/learning_descriptors/LM4CV/data/cub/cub_attributes_kg_1iter.txt", 'r').read().strip().split("\n")
        # return open("/var/local/atharvas/f/learning_descriptors/LM4CV/data/cub/cub_attributes_kg.txt", 'r').read().strip().split("\n")
        # return open("/var/local/atharvas/f/learning_descriptors/LM4CV/data/cub/cub_attributes_dedup_08.txt", 'r').read().strip().split("\n")
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/cub/cub_attributes.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )

    elif cfg["attributes"] == "flower":
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/flowers-102/flower_attributes.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )

    elif cfg["attributes"] == "food":
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/food-101/food_attributes.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )

    elif cfg["attributes"] == "cars":
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/stanford_cars/cars_attributes.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )

    elif cfg["attributes"] == "imagenet":
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/imagenet/imagenet_attributes.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )

    elif cfg["attributes"] == "imagenet-animal":
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/imagenet/imagenet_animal_attributes.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )

    elif cfg["attributes"] == "cifar10":
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/cifar-10-batches-py/cifar10_attributes.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )

    elif cfg["attributes"] == "cifar100":
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/cifar-100-python/cifar100_attributes.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )

    elif cfg["attributes"] == "oxford_pets":
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/oxford-iiit-pet/oxford_pets_attributes.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )
    elif cfg["attributes"] == "places365":
        return (
            open(
                "/var/local/atharvas/f/learning_descriptors/LM4CV/data/places365/places365_attributes_kg_1iter.txt",
                "r",
            )
            .read()
            .strip()
            .split("\n")
        )
    else:
        raise NotImplementedError


def get_prefix(cfg):
    if cfg["attributes"] == "cbm":
        return ""
    if (
        cfg["dataset"] == "cub"
        or cfg["dataset"] == "waterbirds"
        or cfg["dataset"] == "nabirds"
    ):
        # return "A photo of a bird with "
        return "The bird has "
    elif cfg["dataset"] == "inat":
        return "A photo of a creature with "
    elif cfg["dataset"] == "cifar100":
        return "A photo of an object with "
    elif cfg["dataset"] == "cifar10":
        # return "A photo of an object with "
        return "A blur photo of an object with"
    elif cfg["dataset"] == "cifar10-p":
        return "A photo of an object with "
    elif cfg["dataset"] == "flower":
        return "A photo of the flower with "
    elif cfg["dataset"] == "food":
        return "A photo of the food with "
    elif cfg["dataset"] == "cars":
        return "A photo of the car with "
    elif cfg["dataset"] == "oxford_pets":
        return "A photo of the animal with "
    elif cfg["dataset"] == "imagenet":
        return "A photo of an object with "
    elif cfg["dataset"] in ["imagenet-animal", "imagenet-a"]:
        return "A photo of an animal with "
    elif cfg["dataset"] == "places365":
        return "A photo of a place with "
    else:
        raise NotImplementedError
