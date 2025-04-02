"""
evaluate zero-shot performance
"""

import copy
import json
import os
import sys
import time

import clip
import numpy as np
import open_clip
import torch
import torch.utils
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from escher.lm4cv.dataset import FeatureDataset

from .dataset_utils import (
    get_attributes,
    get_folder_name,
    get_image_dataloader,
    get_labels,
)


class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def set_seed(seed):
    if seed == -1:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def construct_attributes_save_path(cfg):

    if (
        cfg["cluster_feature_method"] == "random"
        or cfg["cluster_feature_method"] == "kmeans"
    ):
        mahalanobis = False
    else:
        mahalanobis = cfg["mahalanobis"]

    if not mahalanobis:
        return (
            cfg["dataset"]
            + "_"
            + cfg["attributes"]
            + "_"
            + "_"
            + cfg["cluster_feature_method"]
            + "_"
            + str(cfg["num_attributes"])
            + "_"
            + str(cfg["reinit"])
            + ".txt"
        )
    else:
        return (
            cfg["dataset"]
            + "_"
            + cfg["attributes"]
            + "_"
            + "_"
            + cfg["cluster_feature_method"]
            + "_"
            + str(cfg["num_attributes"])
            + "_"
            + "mahalanobis"
            + "_"
            + str(cfg["division_power"])
            + "_"
            + str(cfg["reinit"])
            + ".txt"
        )


def get_model(cfg, model, input_dim, output_dim):

    if cfg["num_attributes"] == "full":
        num_attributes = len(get_attributes(cfg))
    else:
        num_attributes = cfg["num_attributes"]

    if model == ["linear", "bn", "linear"]:
        model = nn.Sequential(
            nn.Linear(input_dim, num_attributes, bias=False),
            nn.BatchNorm1d(num_attributes),
            nn.Linear(num_attributes, output_dim),
        )
    elif model == ["bn", "linear"]:
        model = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, output_dim, bias=False),
        )
    elif model == ["linear", "linear"]:
        model = nn.Sequential(
            nn.Linear(input_dim, num_attributes, bias=False),
            nn.Linear(num_attributes, output_dim),
        )
    elif model == ["linear"]:
        model = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
        )

    else:
        raise NotImplementedError

    return model


def get_feature_dataloader(cfg):

    if cfg["model_type"] == "clip":
        model, preprocess = clip.load(cfg["model_size"])
    elif cfg["model_type"] == "open_clip":
        if "openclip_pretrain" in cfg:
            pretrained = cfg["openclip_pretrain"]
        else:
            pretrained = None
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["model_size"], pretrained=pretrained, device="cuda"
        )
    else:
        raise NotImplementedError

    train_loader, test_loader = get_image_dataloader(
        cfg["dataset"], preprocess, shots=cfg.get("shots", None)
    )

    train_features = get_image_embeddings(
        cfg, cfg["dataset"], model, train_loader, "train", shots=cfg.get("shots", None)
    )
    test_features = get_image_embeddings(
        cfg, cfg["dataset"], model, test_loader, "test", shots=cfg.get("shots", None)
    )

    if cfg["dataset"] == "imagenet-animal":
        train_labels, test_labels = get_labels(
            "imagenet-animal", shots=cfg.get("shots", None)
        )
        train_labels, test_labels = np.array(train_labels), np.array(test_labels)
        (train_idxes,) = np.where((train_labels < 398) & (train_labels != 69))
        train_features = train_features[train_idxes]

        test_idxes = np.where((test_labels < 398) & (test_labels != 69))
        test_features = test_features[test_idxes]

    if cfg["dataset"] == "waterbirds":
        (
            train_labels,
            test_labels,
            train_group_array,
            test_group_array,
            train_selected_indices,
            test_selected_indices,
        ) = get_labels(cfg["dataset"], shots=cfg.get("shots", None))
        if len(train_labels) != len(train_features):
            train_features = train_features[train_selected_indices]
            test_features = test_features[test_selected_indices]

        train_score_dataset = FeatureDataset(
            train_features, train_labels, train_group_array
        )
        test_score_dataset = FeatureDataset(
            test_features, test_labels, test_group_array
        )

    else:
        train_labels, test_labels = get_labels(
            cfg["dataset"], shots=cfg.get("shots", None)
        )
        train_score_dataset = FeatureDataset(train_features, train_labels)
        test_score_dataset = FeatureDataset(test_features, test_labels)

    train_loader = MultiEpochsDataLoader(
        train_score_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=16
    )
    test_loader = MultiEpochsDataLoader(
        test_score_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=16
    )

    return train_loader, test_loader


def get_score_dataloader(cfg, attribute_embeddings):

    if cfg["model_type"] == "clip":
        model, preprocess = clip.load(cfg["model_size"])
    elif cfg["model_type"] == "open_clip":
        pretrained = getattr(cfg, "openclip_pretrain", [])
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["model_size"], pretrained=pretrained, device="cuda"
        )
    else:
        raise NotImplementedError
    train_loader, test_loader = get_image_dataloader(
        cfg["dataset"], preprocess, shots=cfg.get("shots", None)
    )
    print("Get Embeddings...")
    train_features = get_image_embeddings(
        cfg, cfg["dataset"], model, train_loader, "train", shots=cfg.get("shots", None)
    )
    test_features = get_image_embeddings(
        cfg, cfg["dataset"], model, test_loader, "test", shots=cfg.get("shots", None)
    )
    if cfg["dataset"] == "imagenet-animal":
        train_labels, test_labels = get_labels(
            "imagenet-animal", shots=cfg.get("shots", None)
        )
        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        train_idxes = np.where((train_labels < 398) & (train_labels != 69))
        train_features = train_features[train_idxes]

        test_idxes = np.where((test_labels < 398) & (test_labels != 69))
        test_features = test_features[test_idxes]

    if cfg["dataset"] == "waterbirds":
        (
            train_labels,
            test_labels,
            train_group_array,
            test_group_array,
            train_selected_indices,
            test_selected_indices,
        ) = get_labels(cfg["dataset"], shots=cfg.get("shots", None))
        if len(train_labels) != len(train_features):
            train_features = train_features[train_selected_indices]
            test_features = test_features[test_selected_indices]
    else:
        train_labels, test_labels = get_labels(
            cfg["dataset"], shots=cfg.get("shots", None)
        )

    print("Initializing Feature Dataset")
    train_feature_dataset = FeatureDataset(train_features, train_labels)
    test_feature_dataset = FeatureDataset(test_features, test_labels)
    train_loader = DataLoader(
        train_feature_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=8,
    )
    test_loader = DataLoader(
        test_feature_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=8
    )

    train_scores = extract_concept_scores(train_loader, model, attribute_embeddings)
    test_scores = extract_concept_scores(test_loader, model, attribute_embeddings)

    if cfg["dataset"] == "waterbirds":
        train_score_dataset = FeatureDataset(
            train_scores, train_labels, train_group_array
        )
        test_score_dataset = FeatureDataset(test_scores, test_labels, test_group_array)
    else:
        train_score_dataset = FeatureDataset(train_scores, train_labels)
        test_score_dataset = FeatureDataset(test_scores, test_labels)

    train_loader = MultiEpochsDataLoader(
        train_score_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=16
    )
    test_loader = MultiEpochsDataLoader(
        test_score_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=16
    )

    return train_loader, test_loader


def calculate_worst_group_acc(predictions, labels, groups):
    time.time()
    comparison = predictions == labels
    worst_group_acc = 1
    for i in range(4):
        indices = torch.where(groups == i)
        acc = torch.sum(comparison[indices]) / len(indices[0])
        worst_group_acc = min(worst_group_acc, acc)
    time.time()
    return worst_group_acc


def get_metrics(labels, predictions, str_labels=None):
    # get the mAP, mAR, mF1, mACC, AUROC,
    report = classification_report(
        labels, predictions, target_names=str_labels, zero_division=0, output_dict=True
    )
    confusion_mat = confusion_matrix(y_pred=predictions, y_true=labels)
    return dict(report=report, confusion_mat=confusion_mat)


def train_model(
    cfg, epochs, model, train_loader, test_loader, regularizer=None, configs=None
):
    if "inat" in cfg["dataset"]:
        return train_model_multigpu(
            cfg=cfg,
            epochs=epochs,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            regularizer=regularizer,
            configs=configs,
        )
    model.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-6)
    )
    scaler = torch.amp.GradScaler()
    loss_function = torch.nn.CrossEntropyLoss()
    best_acc = torch.tensor(0).cuda()
    best_metrics = None
    last_best_acc = None
    best_model_state = copy.deepcopy(model.state_dict())

    no_break = False
    if epochs < 0:
        print("No Early Stopping")
        epochs = -epochs
        no_break = True

    loader = tqdm if cfg.get("use_tqdm", False) else lambda x: x

    PATIENCE_THRESHOLD = (
        1000  # if best accuracy doesn't improve for 500 epochs, stop training
    )
    patience = 0

    for epoch in range(epochs):
        # Train:
        training_metrics = (
            None if cfg.get("no_metrics", False) else {"training_report": []}
        )
        model.train()
        for idx, batch in enumerate(loader(train_loader)):
            s, t = (
                batch[0].cuda(non_blocking=True).float(),
                batch[1].cuda(non_blocking=True).long(),
            )

            with torch.amp.autocast():
                output = model(s)
                loss = loss_function(output, t)

                if regularizer == "mahalanobis":
                    mahalanobis_loss = (
                        mahalanobis_distance(
                            model[0].weight
                            / model[0].weight.data.norm(dim=-1, keepdim=True),
                            configs["mu"].cuda(),
                            configs["sigma_inv"].cuda(),
                        )
                        - configs["mean_distance"]
                    ) / (configs["mean_distance"] ** cfg["division_power"])
                    loss = loss + torch.abs(mahalanobis_loss)
                elif regularizer == "cosine":
                    weight = model[0].weight / model[0].weight.norm(
                        dim=-1, keepdim=True
                    )
                    loss += (
                        cfg["lambda"]
                        * torch.sum(
                            (weight - configs["mu"].unsqueeze(0).cuda()) ** 2
                        ).mean()
                    )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if training_metrics is not None:
                training_metrics["training_report"].append(
                    output.topk(cfg.get("topk", 5)).indices.detach().cpu()
                )

        # Evaluate:
        model.eval()
        with torch.no_grad():
            predictions, labels = [], []
            for idx, batch in enumerate(loader(test_loader)):
                s = batch[0].cuda(non_blocking=True).float()
                with autocast():
                    output = model(s)
                pred = output.argmax(dim=-1)
                predictions.append(pred)
                labels.append(batch[1].cuda(non_blocking=True).long())

            predictions = torch.cat(predictions)
            labels = torch.cat(labels)
            acc = (predictions == labels).float().mean() * 100

        # Update best model if accuracy improves
        if acc > best_acc:
            best_acc = acc
            best_metrics = get_metrics(labels.cpu(), predictions.cpu())
            if training_metrics is not None:
                best_metrics["training_report"] = torch.cat(
                    training_metrics["training_report"]
                ).numpy()
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience > PATIENCE_THRESHOLD:
                break

        # Early stopping check
        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}], Best accuracy:",
                best_acc.item(),
                "Last accuracy:",
                acc.item(),
            )

            sys.stdout.flush()

            if not no_break and (
                last_best_acc is not None and best_acc == last_best_acc
            ):
                break
            last_best_acc = best_acc

    # Load best model state
    model.load_state_dict(best_model_state)
    return model, best_acc, best_metrics


def mahalanobis_distance(x, mu, sigma_inv):
    x = x - mu.unsqueeze(0)
    return torch.diag(x @ sigma_inv @ x.T).mean()


def filter_features(features, labels):
    # remove any feature and label where label == -1
    mask = labels != -1
    return features[mask], labels[mask]


def train_model_only(
    cfg,
    model,
    train_features,
    train_labels,
    val_features,
    val_labels,
    epochs=None,
    regularizer=None,
    configs=None,
):
    dataset = TensorDataset(*filter_features(train_features.cpu(), train_labels.cpu()))
    val_dataset = TensorDataset(*filter_features(val_features.cpu(), val_labels.cpu()))
    train_loader = MultiEpochsDataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = MultiEpochsDataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    epochs = cfg["epochs"] if epochs is None else epochs
    model.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-6)
    )
    scaler = torch.amp.GradScaler("cuda")
    loss_function = torch.nn.CrossEntropyLoss()
    best_acc = 0
    best_metrics = None
    last_best_acc = None
    best_model_state = copy.deepcopy(model.state_dict())

    no_break = False
    if epochs < 0:
        print("No Early Stopping")
        epochs = -epochs
        no_break = True

    PATIENCE_THRESHOLD = (
        1000  # if best accuracy doesn't improve for 500 epochs, stop training
    )
    patience = 0

    for epoch in range(epochs):
        # Train:
        training_metrics = (
            None if cfg.get("no_metrics", False) else {"training_report": []}
        )
        model.train()
        for idx, batch in enumerate(train_loader):
            s, t = (
                batch[0].cuda(non_blocking=True).float(),
                batch[1].cuda(non_blocking=True).long(),
            )

            with torch.amp.autocast("cuda"):
                output = model(s)
                loss = loss_function(output, t)

                if regularizer == "mahalanobis":
                    mahalanobis_loss = (
                        mahalanobis_distance(
                            model[0].weight
                            / model[0].weight.data.norm(dim=-1, keepdim=True),
                            configs["mu"].cuda(),
                            configs["sigma_inv"].cuda(),
                        )
                        - configs["mean_distance"]
                    ) / (configs["mean_distance"] ** cfg["division_power"])
                    loss = loss + torch.abs(mahalanobis_loss)
                elif regularizer == "cosine":
                    weight = model[0].weight / model[0].weight.norm(
                        dim=-1, keepdim=True
                    )
                    loss += (
                        cfg["lambda"]
                        * torch.sum(
                            (weight - configs["mu"].unsqueeze(0).cuda()) ** 2
                        ).mean()
                    )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        with torch.no_grad():
            predictions, labels = [], []
            for idx, batch in enumerate(val_loader):
                s = batch[0].cuda(non_blocking=True).float()
                with torch.amp.autocast("cuda"):
                    output = model(s)
                pred = output.argmax(dim=-1)
                predictions.append(pred)
                labels.append(batch[1].cuda(non_blocking=True).long())

            predictions = torch.cat(predictions)
            labels = torch.cat(labels)
            acc = (predictions == labels).float().mean() * 100

        # Update best model if accuracy improves
        if acc > best_acc:
            best_acc = acc
            best_metrics = get_metrics(labels.cpu(), predictions.cpu())
            if training_metrics is not None:
                best_metrics["training_report"] = torch.cat(
                    training_metrics["training_report"]
                ).numpy()
            best_model_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience > PATIENCE_THRESHOLD:
                break

        # Early stopping check
        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}], Best accuracy:",
                best_acc.item(),
                "Last accuracy:",
                acc.item(),
            )

            sys.stdout.flush()

            if not no_break and (
                last_best_acc is not None and best_acc == last_best_acc
            ):
                break
            last_best_acc = best_acc

    # Load best model state
    model.load_state_dict(best_model_state)
    return model, best_acc, best_metrics


def get_image_embeddings(cfg, dataset, model, loader, mode="train", shots=None):

    if dataset == "imagenet-a" and mode == "train":
        folder_name = get_folder_name("imagenet")
    else:
        folder_name = get_folder_name(dataset)

    model_name = cfg["model_type"] + "_" + cfg["model_size"].split("/")[-1]
    if shots is not None:
        model_name += f"_shots_{shots}"

    if model_name == "clip_32":
        filename = f"./data/{folder_name}/{mode}_embeddings.npy"
    else:
        filename = f"./data/{folder_name}/{model_name}_{mode}_embeddings.npy"

    num_images = len(loader.dataset)
    feature_dim = 768

    if os.path.exists(filename):
        try:
            features = np.load(filename, allow_pickle=True)
        except Exception as e:
            print(e)
            features = np.memmap(
                filename, dtype="float32", mode="r", shape=(num_images, feature_dim)
            )
    else:
        print("Extract and pre-save image features...")
        with torch.no_grad():
            # if cfg['dataset'] == 'inat':
            # Adjust num_workers as needed
            loader = torch.utils.data.DataLoader(
                loader.dataset,
                batch_size=2048,  # Try increasing batch_size
                num_workers=16,  # Increase num_workers for faster data loading
                pin_memory=True,
            )
            # Create a memory-mapped file to save features directly to disk
            # features = np.memmap(
            #     filename,
            #     dtype='float32',
            #     mode='w+',
            #     shape=(num_images, feature_dim)
            # )
            features = np.zeros((num_images, feature_dim), dtype="float32")

            idx = 0
            for batch in tqdm(
                loader, total=len(loader), desc="Extracting image features"
            ):
                images = batch[0].cuda(non_blocking=True)
                # If your model is not in eval mode, set it
                model.eval()
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                image_features = image_features.cpu().numpy()
                batch_size = image_features.shape[0]

                # # Write features to the memory-mapped file
                # features[idx:idx + batch_size] = image_features
                features[idx : idx + batch_size] = image_features
                idx += batch_size

            # Flush changes to disk
            # features.flush()
            np.save(filename, features)
    return features


def extract_concept_scores(loader, model, attribute_embeddings):
    with torch.no_grad():
        scores = []

        for i, (image_features, _) in tqdm(
            enumerate(loader), total=len(loader), desc="extracting concept scores"
        ):
            image_features = image_features.cuda().half()
            # target = target.cuda()
            # image_features = model.encode_image(images).float()
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ attribute_embeddings.half().T.cuda()
            scores.append(logits.cpu().numpy())

        scores = np.concatenate(scores, axis=0)

    return scores


def train_model_multigpu(
    cfg, epochs, model, train_loader, test_loader, regularizer=None, configs=None
):
    model = model.cuda()
    model = torch.nn.DataParallel(
        model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]
    )  # Use all 7 GPUs
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-6)
    )
    scaler = GradScaler()
    loss_function = torch.nn.CrossEntropyLoss()
    best_acc = 0
    best_metrics = None
    last_best_acc = None
    best_model_state = copy.deepcopy(model.state_dict())

    no_break = False
    if epochs < 0:
        print("No Early Stopping")
        epochs = -epochs
        no_break = True

    loader = tqdm if cfg.get("use_tqdm", False) else lambda x: x
    with open(f"cls2attributes/{cfg['dataset']}_cls2attributes.json", "r") as f:
        cls2attributes = json.load(f)
        classes = list(cls2attributes.keys())
    train_logits = np.memmap(
        f"./data/{cfg['dataset']}/train_logits.npy",
        dtype="float32",
        mode="w+",
        shape=(len(train_loader.dataset), len(classes)),
    )
    val_logits = np.memmap(
        f"./data/{cfg['dataset']}/val_logits.npy",
        dtype="float32",
        mode="w+",
        shape=(len(test_loader.dataset), len(classes)),
    )
    for epoch in range(epochs):
        # Train:
        training_metrics = (
            None if cfg.get("no_metrics", False) else {"training_report": []}
        )
        model.train()
        for idx, batch in enumerate(loader(train_loader)):
            s, t = (
                batch[0].cuda(non_blocking=True).float(),
                batch[1].cuda(non_blocking=True).long(),
            )

            with autocast():
                output = model(s)
                loss = loss_function(output, t)

                if regularizer == "mahalanobis":
                    weight = model.module[0].weight / model.module[0].weight.data.norm(
                        dim=-1, keepdim=True
                    )
                    mahalanobis_loss = (
                        mahalanobis_distance(
                            weight, configs["mu"].cuda(), configs["sigma_inv"].cuda()
                        )
                        - configs["mean_distance"]
                    ) / (configs["mean_distance"] ** cfg["division_power"])
                    loss = loss + torch.abs(mahalanobis_loss)
                elif regularizer == "cosine":
                    weight = model.module[0].weight / model.module[0].weight.norm(
                        dim=-1, keepdim=True
                    )
                    loss += (
                        cfg["lambda"]
                        * torch.sum(
                            (weight - configs["mu"].unsqueeze(0).cuda()) ** 2
                        ).mean()
                    )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if training_metrics is not None:
                # training_metrics['training_report'].append(
                #     output.topk(cfg.get('topk', 5)).indices.detach().cpu()
                # )
                train_logits[
                    idx * cfg["batch_size"] : (idx + 1) * cfg["batch_size"]
                ] = (output.detach().cpu().numpy())

        if training_metrics is not None:
            train_logits.flush()

        # Evaluate:
        model.eval()
        with torch.no_grad():
            predictions, labels = [], []
            for idx, batch in enumerate(loader(test_loader)):
                s = batch[0].cuda(non_blocking=True).float()
                with autocast():
                    output = model(s)
                pred = output.argmax(dim=-1)
                predictions.append(pred)
                labels.append(batch[1].cuda(non_blocking=True).long())
                if training_metrics is not None:
                    val_logits[
                        idx * cfg["batch_size"] : (idx + 1) * cfg["batch_size"]
                    ] = (output.detach().cpu().numpy())

            predictions = torch.cat(predictions)
            labels = torch.cat(labels)
            acc = (predictions == labels).float().mean() * 100

        # Update best model if accuracy improves
        if acc > best_acc:
            best_acc = acc
            best_metrics = get_metrics(labels.cpu(), predictions.cpu())
            if training_metrics is not None:
                # best_metrics['training_report'] = torch.cat(training_metrics['training_report']).numpy()
                # save train_logits
                np.save(f"./data/{cfg['dataset']}/best_train_logits.npy", train_logits)
                np.save(f"./data/{cfg['dataset']}/best_val_logits.npy", val_logits)
                best_metrics["logits"] = (
                    f"./data/{cfg['dataset']}/best_train_logits.npy",
                    f"./data/{cfg['dataset']}/best_val_logits.npy",
                )
                best_metrics["acc"] = acc.item()
            best_model_state = copy.deepcopy(model.state_dict())

        # Early stopping check
        if epoch % 10 == 0:
            print(
                f"Epoch [{epoch}], Best accuracy:",
                best_acc.item(),
                "Last accuracy:",
                acc.item(),
            )
            sys.stdout.flush()
            if not no_break and (
                last_best_acc is not None and best_acc == last_best_acc
            ):
                break
            last_best_acc = best_acc

    # Load best model state
    model.load_state_dict(best_model_state)
    # convert the model back to single GPU

    return model, best_acc, best_metrics
