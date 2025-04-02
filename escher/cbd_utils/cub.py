import os

import torch
from torchvision import datasets


class CUBDataset(datasets.ImageFolder):
    """
    Wrapper for the CUB-200-2011 dataset.
    Method DatasetBirds.__getitem__() returns tuple of image and its corresponding label.
    Dataset per https://github.com/slipnitskaya/caltech-birds-advanced-classification
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
        train=True,
        bboxes=False,
        keep_str: str = None,
    ):
        img_root = os.path.join(root, "images")

        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.redefine_class_to_idx(keep_str)

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train

        # obtain sample ids filtered by split
        path_to_splits = os.path.join(root, "train_test_split.txt")
        indices_to_use = list()
        with open(path_to_splits, "r") as in_file:
            for line in in_file:
                idx, use_train = line.strip("\n").split(" ", 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        # obtain filenames of images
        path_to_index = os.path.join(root, "images.txt")
        filenames_to_use = set()
        with open(path_to_index, "r") as in_file:
            for line in in_file:
                idx, fn = line.strip("\n").split(" ", 2)
                if int(idx) in indices_to_use:
                    if keep_str and keep_str not in fn.lower():
                        continue
                    filenames_to_use.add(fn)

        img_paths_cut = {
            "/".join(img_path.rsplit("/", 2)[-2:]): idx
            for idx, (img_path, lb) in enumerate(self.imgs)
        }
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        if bboxes:
            # get coordinates of a bounding box
            path_to_bboxes = os.path.join(root, "bounding_boxes.txt")
            bounding_boxes = list()
            with open(path_to_bboxes, "r") as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(
                        lambda x: float(x), line.strip("\n").split(" ")
                    )
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

        self.reindex_idxs()

    def __getitem__(self, index):
        # generate one sample
        sample, target = super(CUBDataset, self).__getitem__(index)

        if self.bboxes is not None:
            # squeeze coordinates of the bounding box to range [0, 1]
            width, _height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target

    def redefine_class_to_idx(self, keep_str):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split(".")[-1].replace("_", " ")
            split_key = k.split(" ")
            if len(split_key) > 2:
                k = "-".join(split_key[:-1]) + " " + split_key[-1]
            if keep_str and keep_str not in k.lower():
                continue
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict

    def reindex_idxs(self):
        # adjust so smallest class is 0 for these variables
        # 'class_to_idx', 'samples', 'targets', 'imgs',

        # adjust class_to_idx
        min_idx = min(self.class_to_idx.values())
        self.class_to_idx = {k: v - min_idx for k, v in self.class_to_idx.items()}

        # adjust samples
        self.samples = [(p, t - min_idx) for p, t in self.samples]

        # adjust imgs
        self.imgs = [(p, t - min_idx) for p, t in self.imgs]

        # adjust targets
        self.targets = [t - min_idx for t in self.targets]


def extract_sparrows_from_cub_descriptors():
    from escher.cbd_utils.utils import load_obj, save_obj

    cub_descriptors = load_obj("descriptors/cbd_descriptors/descriptors_cub.json")
    sparrow_descriptors = {
        k: v for k, v in cub_descriptors.items() if "sparrow" in k.lower()
    }
    save_obj(
        sparrow_descriptors, "descriptors/cbd_descriptors/descriptors_cub_sparrows.json"
    )
