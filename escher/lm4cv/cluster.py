import clip
import numpy as np
import open_clip
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from escher.lm4cv.dataset import FeatureDataset

from .utils.train_utils import (
    extract_concept_scores,
    get_attributes,
    get_feature_dataloader,
    get_image_dataloader,
    get_image_embeddings,
    get_labels,
    get_model,
    mahalanobis_distance,
    train_model,
    train_model_only,
)
from .utils.dataset_utils import get_output_dim, get_prefix

# from .utils.train_utils import *


def cluster(cfg):

    if cfg["model_type"] == "clip":
        model, preprocess = clip.load(cfg["model_size"])
    elif cfg["model_type"] == "open_clip":
        pretrained = getattr(cfg, "openclip_pretrain", [])
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["model_size"], pretrained=pretrained, device="cuda"
        )
        model = model.cuda()
        tokenizer = open_clip.get_tokenizer(cfg["model_size"])
    else:
        raise NotImplementedError

    attributes = get_attributes(cfg)
    attribute_embeddings = []
    batch_size = 256
    for i in tqdm(
        range((len(attributes) // batch_size) + 1), desc="Embedding Attributes"
    ):
        sub_attributes = attributes[i * batch_size : (i + 1) * batch_size]
        if cfg["model_type"] == "clip":
            clip_attributes_embeddings = clip.tokenize(
                [get_prefix(cfg) + attr for attr in sub_attributes]
            ).cuda()
        elif cfg["model_type"] == "open_clip":
            clip_attributes_embeddings = tokenizer(
                [get_prefix(cfg) + attr for attr in sub_attributes]
            ).cuda()

        batch_embeddings = model.encode_text(clip_attributes_embeddings).detach().cpu()
        attribute_embeddings.append(batch_embeddings)

    attribute_embeddings = torch.concatenate(attribute_embeddings).float()
    attribute_embeddings = attribute_embeddings / attribute_embeddings.norm(
        dim=-1, keepdim=True
    )

    print("num_attributes: ", cfg["num_attributes"])
    if cfg["num_attributes"] == "full":
        return attributes, attribute_embeddings

    if cfg["cluster_feature_method"] == "random":
        selected_idxes = np.random.choice(
            np.arange(len(attribute_embeddings)),
            size=cfg["num_attributes"],
            replace=False,
        )

    elif cfg["cluster_feature_method"] == "similarity":
        if cfg["model_type"] == "clip":
            model, preprocess = clip.load(cfg["model_size"])
        else:
            raise NotImplementedError
        train_loader, test_loader = get_image_dataloader(
            cfg["dataset"], preprocess, shots=cfg.get("shots", None)
        )
        print("Get Embeddings...")
        train_features = get_image_embeddings(
            cfg, cfg["dataset"], model, train_loader, "train"
        )
        test_features = get_image_embeddings(
            cfg, cfg["dataset"], model, test_loader, "test"
        )
        if cfg["dataset"] == "imagenet-animal":
            train_labels, test_labels = get_labels(
                "imagenet", shots=cfg.get("shots", None)
            )
            train_labels, test_labels = np.array(train_labels), np.array(test_labels)

            train_idxes = np.where((train_labels < 398) & (train_labels != 69))
            train_features = train_features[train_idxes]

            test_idxes = np.where((test_labels < 398) & (test_labels != 69))
            test_features = test_features[test_idxes]
        train_labels, test_labels = get_labels(
            cfg["dataset"], shots=cfg.get("shots", None)
        )

        print("Initializing Feature Dataset")
        train_feature_dataset = FeatureDataset(train_features, train_labels)
        train_loader = DataLoader(
            train_feature_dataset, batch_size=cfg["batch_size"], shuffle=False
        )
        train_scores = extract_concept_scores(train_loader, model, attribute_embeddings)

        train_scores = np.array(train_scores)
        mean_scores = np.mean(train_scores, axis=0)
        assert len(mean_scores) == len(attribute_embeddings)
        selected_idxes = np.argsort(mean_scores)[::-1][: cfg["num_attributes"]].astype(
            int
        )

    else:
        if cfg["cluster_feature_method"] == "linear":
            mu = torch.mean(attribute_embeddings, dim=0)
            sigma_inv = torch.tensor(np.linalg.inv(torch.cov(attribute_embeddings.T)))
            configs = {
                "mu": mu,
                "sigma_inv": sigma_inv,
                "mean_distance": np.mean(
                    [
                        mahalanobis_distance(embed, mu, sigma_inv)
                        for embed in attribute_embeddings
                    ]
                ),
            }

            model = get_model(
                cfg,
                cfg["linear_model"],
                input_dim=attribute_embeddings.shape[-1],
                output_dim=get_output_dim(cfg["dataset"], shots=cfg.get("shots", None)),
            )
            train_loader, test_loader = get_feature_dataloader(cfg)
            if cfg["mahalanobis"]:
                best_model, best_acc, _ = train_model(
                    cfg,
                    cfg["linear_epochs"],
                    model,
                    train_loader,
                    test_loader,
                    regularizer="mahalanobis",
                    configs=configs,
                )
            else:
                if cfg.get("cosine", False):
                    best_model, best_acc, _ = train_model(
                        cfg,
                        cfg["linear_epochs"],
                        model,
                        train_loader,
                        test_loader,
                        regularizer="cosine",
                        configs=configs,
                    )

                else:
                    best_model, best_acc, _ = train_model(
                        cfg,
                        cfg["linear_epochs"],
                        model,
                        train_loader,
                        test_loader,
                        regularizer=None,
                        configs=configs,
                    )

            centers = best_model[0].weight.detach().cpu().numpy()

        elif cfg["cluster_feature_method"] == "kmeans":
            kmeans = KMeans(n_clusters=cfg["num_attributes"], random_state=0).fit(
                attribute_embeddings
            )
            centers = kmeans.cluster_centers_

        elif cfg["cluster_feature_method"] == "svd":
            u, s, vh = np.linalg.svd(
                attribute_embeddings.numpy().astype(np.float32), full_matrices=False
            )

            u = u[: cfg["num_attributes"], :]
            centers = u @ np.diag(s) @ vh
        else:
            raise NotImplementedError

        selected_idxes = []
        for center in centers:
            center = center / torch.tensor(center).norm().numpy()
            distances = np.sum(
                (attribute_embeddings.numpy() - center.reshape(1, -1)) ** 2, axis=1
            )
            # sorted_idxes = np.argsort(distances)[::-1]
            sorted_idxes = np.argsort(distances)
            count = 0
            while sorted_idxes[count] in selected_idxes:
                count += 1
            selected_idxes.append(sorted_idxes[count])
        selected_idxes = np.array(selected_idxes)

    if cfg["cluster_feature_method"] == "linear":
        return (
            best_acc,
            best_model,
            [attributes[i] for i in selected_idxes],
            torch.tensor(attribute_embeddings[selected_idxes]),
        )
    else:
        return [attributes[i] for i in selected_idxes], torch.tensor(
            attribute_embeddings[selected_idxes]
        )


def cluster_model_only(cfg, train_dataset, test_dataset):

    if cfg["model_type"] == "clip":
        model, preprocess = clip.load(cfg["model_size"])
    elif cfg["model_type"] == "open_clip":
        pretrained = getattr(cfg, "openclip_pretrain", [])
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["model_size"], pretrained=pretrained, device="cuda"
        )
        model = model.cuda()
        tokenizer = open_clip.get_tokenizer(cfg["model_size"])
    else:
        raise NotImplementedError

    attributes = get_attributes(cfg)
    attribute_embeddings = []
    batch_size = 256
    for i in tqdm(
        range((len(attributes) // batch_size) + 1), desc="Embedding Attributes"
    ):
        sub_attributes = attributes[i * batch_size : (i + 1) * batch_size]
        if cfg["model_type"] == "clip":
            clip_attributes_embeddings = clip.tokenize(
                [get_prefix(cfg) + attr for attr in sub_attributes]
            ).cuda()
        elif cfg["model_type"] == "open_clip":
            clip_attributes_embeddings = tokenizer(
                [get_prefix(cfg) + attr for attr in sub_attributes]
            ).cuda()

        batch_embeddings = model.encode_text(clip_attributes_embeddings).detach().cpu()
        attribute_embeddings.append(batch_embeddings)

    attribute_embeddings = torch.concatenate(attribute_embeddings).float()
    attribute_embeddings = attribute_embeddings / attribute_embeddings.norm(
        dim=-1, keepdim=True
    )

    print("num_attributes: ", cfg["num_attributes"])
    if cfg["num_attributes"] == "full":
        return attributes, attribute_embeddings

    if cfg["cluster_feature_method"] == "random":
        selected_idxes = np.random.choice(
            np.arange(len(attribute_embeddings)),
            size=cfg["num_attributes"],
            replace=False,
        )

    elif cfg["cluster_feature_method"] == "similarity":
        if cfg["model_type"] == "clip":
            model, preprocess = clip.load(cfg["model_size"])
        else:
            raise NotImplementedError
        train_loader, test_loader = get_image_dataloader(
            cfg["dataset"], preprocess, shots=cfg.get("shots", None)
        )
        print("Get Embeddings...")

        # train_features = get_image_embeddings(cfg, cfg['dataset'], model, train_loader, 'train')
        # test_features = get_image_embeddings(cfg, cfg['dataset'], model, test_loader, 'test')
        # if cfg['dataset'] == 'imagenet-animal':
        #     train_labels, test_labels = get_labels("imagenet", shots=cfg.get('shots', None))
        #     train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        #     train_idxes = np.where((train_labels < 398) & (train_labels!=69))
        #     train_features = train_features[train_idxes]

        #     test_idxes = np.where((test_labels < 398) & (test_labels!=69))
        #     test_features = test_features[test_idxes]
        # train_labels, test_labels = get_labels(cfg['dataset'], shots=cfg.get('shots', None))
        train_features = train_dataset.tensors[0]
        train_labels = train_dataset.tensors[1]
        test_dataset.tensors[0]
        test_dataset.tensors[1]

        print("Initializing Feature Dataset")
        train_feature_dataset = FeatureDataset(train_features, train_labels)
        train_loader = DataLoader(
            train_feature_dataset, batch_size=cfg["batch_size"], shuffle=False
        )
        train_scores = extract_concept_scores(train_loader, model, attribute_embeddings)

        train_scores = np.array(train_scores)
        mean_scores = np.mean(train_scores, axis=0)
        assert len(mean_scores) == len(attribute_embeddings)
        selected_idxes = np.argsort(mean_scores)[::-1][: cfg["num_attributes"]].astype(
            int
        )

    else:
        if cfg["cluster_feature_method"] == "linear":
            mu = torch.mean(attribute_embeddings, dim=0)
            sigma_inv = torch.tensor(np.linalg.inv(torch.cov(attribute_embeddings.T)))
            configs = {
                "mu": mu,
                "sigma_inv": sigma_inv,
                "mean_distance": np.mean(
                    [
                        mahalanobis_distance(embed, mu, sigma_inv)
                        for embed in attribute_embeddings
                    ]
                ),
            }

            model = get_model(
                cfg,
                cfg["linear_model"],
                input_dim=attribute_embeddings.shape[-1],
                output_dim=len(np.unique(train_dataset.tensors[1].numpy())),
            )
            # train_loader, test_loader = get_feature_dataloader(cfg)
            # train_loader = MultiEpochsDataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=16)
            # test_loader = MultiEpochsDataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=16)
            if cfg["mahalanobis"]:
                best_model, best_acc, _ = train_model_only(
                    cfg,
                    model,
                    train_features=train_dataset.tensors[0],
                    train_labels=train_dataset.tensors[1],
                    val_features=test_dataset.tensors[0],
                    val_labels=test_dataset.tensors[1],
                    regularizer="mahalanobis",
                    configs=configs,
                    epochs=cfg["linear_epochs"],
                )
            else:
                if cfg.get("cosine", False):
                    best_model, best_acc, _ = train_model_only(
                        cfg,
                        model,
                        train_features=train_dataset.tensors[0],
                        train_labels=train_dataset.tensors[1],
                        val_features=test_dataset.tensors[0],
                        val_labels=test_dataset.tensors[1],
                        regularizer="cosine",
                        configs=configs,
                        epochs=cfg["linear_epochs"],
                    )

                else:
                    best_model, best_acc, _ = train_model_only(
                        cfg,
                        model,
                        train_features=train_dataset.tensors[0],
                        train_labels=train_dataset.tensors[1],
                        val_features=test_dataset.tensors[0],
                        val_labels=test_dataset.tensors[1],
                        regularizer=None,
                        configs=configs,
                        epochs=cfg["linear_epochs"],
                    )

            centers = best_model[0].weight.detach().cpu().numpy()

        elif cfg["cluster_feature_method"] == "kmeans":
            kmeans = KMeans(n_clusters=cfg["num_attributes"], random_state=0).fit(
                attribute_embeddings
            )
            centers = kmeans.cluster_centers_

        elif cfg["cluster_feature_method"] == "svd":
            u, s, vh = np.linalg.svd(
                attribute_embeddings.numpy().astype(np.float32), full_matrices=False
            )

            u = u[: cfg["num_attributes"], :]
            centers = u @ np.diag(s) @ vh
        else:
            raise NotImplementedError

        selected_idxes = []
        for center in centers:
            center = center / torch.tensor(center).norm().numpy()
            distances = np.sum(
                (attribute_embeddings.numpy() - center.reshape(1, -1)) ** 2, axis=1
            )
            # sorted_idxes = np.argsort(distances)[::-1]
            sorted_idxes = np.argsort(distances)
            count = 0
            while sorted_idxes[count] in selected_idxes:
                count += 1
            selected_idxes.append(sorted_idxes[count])
        selected_idxes = np.array(selected_idxes)

    if cfg["cluster_feature_method"] == "linear":
        return (
            best_acc,
            best_model,
            [attributes[i] for i in selected_idxes],
            torch.tensor(attribute_embeddings[selected_idxes]),
        )
    else:
        return [attributes[i] for i in selected_idxes], torch.tensor(
            attribute_embeddings[selected_idxes]
        )
