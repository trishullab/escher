import argparse

import yaml

from .cluster import cluster, cluster_model_only
from .utils.train_utils import (
    get_feature_dataloader,
    get_model,
    get_score_dataloader,
    set_seed,
    train_model,
)
from .utils.dataset_utils import get_output_dim


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="cub_bn.yaml", help="configurations for training"
    )
    parser.add_argument("--shots", type=int, default=None, help="number of shots")
    parser.add_argument(
        "--outdir", default="./outputs", help="where to put all the results"
    )
    return parser.parse_args()


def get_model_only(cfg, train_dataset, test_dataset):
    set_seed(cfg["seed"])
    # added this in so that the baselines have the same number of steps.
    # cfg["epochs"] = -1 * abs(cfg["epochs"] * 2)
    # cfg["batch_size"] = max(4096, cfg["batch_size"])
    print({k: v if k != "attributes" else "Omitted" for k, v in cfg.items()})

    if cfg["cluster_feature_method"] == "linear" and cfg["num_attributes"] != "full":
        acc, model, attributes, attributes_embeddings = cluster_model_only(
            cfg, train_dataset, test_dataset
        )
    else:
        attributes, attributes_embeddings = cluster_model_only(
            cfg, train_dataset, test_dataset
        )

    if cfg["reinit"] and cfg["num_attributes"] != "full":
        assert cfg["cluster_feature_method"] == "linear"
        model[0].weight.data = attributes_embeddings.cuda() * model[0].weight.data.norm(
            dim=-1, keepdim=True
        )
        for param in model[0].parameters():
            param.requires_grad = False

    else:
        model = get_model(
            cfg,
            cfg["score_model"],
            input_dim=len(attributes),
            output_dim=cfg["num_labels"],
        )

    return model, attributes, attributes_embeddings


def model(cfg):
    set_seed(cfg["seed"])
    # added this in so that the baselines have the same number of steps.
    cfg["epochs"] = -1 * abs(cfg["epochs"] * 2)
    cfg["batch_size"] = max(4096, cfg["batch_size"])
    print(cfg)

    if cfg["cluster_feature_method"] == "linear" and cfg["num_attributes"] != "full":
        acc, model, attributes, attributes_embeddings = cluster(cfg)
    else:
        attributes, attributes_embeddings = cluster(cfg)

    if cfg["reinit"] and cfg["num_attributes"] != "full":
        assert cfg["cluster_feature_method"] == "linear"
        train_loader, test_loader = get_feature_dataloader(cfg)
        model[0].weight.data = attributes_embeddings.cuda() * model[0].weight.data.norm(
            dim=-1, keepdim=True
        )
        for param in model[0].parameters():
            param.requires_grad = False
        best_model, best_acc, best_metrics = train_model(
            cfg, cfg["epochs"], model, train_loader, test_loader
        )

    else:
        model = get_model(
            cfg,
            cfg["score_model"],
            input_dim=len(attributes),
            output_dim=get_output_dim(cfg["dataset"], shots=cfg.get("shots", None)),
        )
        train_loader, test_loader = get_score_dataloader(cfg, attributes_embeddings)
        best_model, best_acc, best_metrics = train_model(
            cfg, cfg["epochs"], model, train_loader, test_loader
        )

    return best_model, best_acc, best_metrics, attributes, (train_loader, test_loader)


if __name__ == "__main__":
    args = parse_config()

    with open(f"{args.config}", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if args.shots is not None:
        cfg["shots"] = args.shots

    model(cfg)
