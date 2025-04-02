from collections import defaultdict

import torch
from escher.utils.dataset_loader import get_processed_dataset


def getfinaliter(exp_name, idx=None, exp_finder=None):
    """
    Given a path 'exp_name' and an optional iteration index,
    returns dictionary information for each iteration or a specific iteration.
    """
    import os
    import re

    if exp_finder is None:
        exp_finder = re.compile(r"iter(\d+).*")

    all_iter = defaultdict(dict)
    history_path = os.path.join(exp_name, "history")

    for f in os.listdir(history_path):
        fullf = os.path.join(history_path, f)
        match = exp_finder.match(f)
        if match:
            iter_num = int(match.group(1))
            name = f.replace(f"iter{iter_num}.", "")
            all_iter[iter_num][name] = fullf
            all_iter[iter_num]["descriptors.json"] = os.path.join(
                exp_name, f"descriptors_{iter_num}.json"
            )

    if idx is None:
        return all_iter

    sorted_keys = sorted(all_iter.keys())
    return all_iter[sorted_keys[idx]]


def query_or_default(kwargs, default_args):
    def fn(key):
        return kwargs.get(key, default_args[key])

    return fn


def get_model(
    dataset_name,
    log_iteration,
    device,
    dataset,
    salt="",
    return_descriptors=False,
    clip_model_name="ViT-L/14",
    **kwargs,
):
    """
    Return a ZeroShotModel (and optionally its descriptors) for a specific iteration.
    """
    from escher.iteration import default_args, get_initial_descriptors
    from escher.models.utils import get_model

    initial_descriptors = get_initial_descriptors(
        dataset_name=dataset_name,
        perc_initial_descriptors=kwargs.get("perc_initial_descriptors", 1.0),
        salt=salt,
    )
    _default_args = default_args()
    args = query_or_default(kwargs, _default_args)
    model, library, start_iteration = get_model(
        initial_descriptors=initial_descriptors,
        clip_model_name=clip_model_name,
        scoring_clip_model_name=args("scoring_clip_model_name"),
        use_open_clip=args("use_open_clip"),
        openai_model=args("openai_model"),
        openai_temp=args("openai_temp"),
        correlation_matrix_threshold=args("correlation_matrix_threshold"),
        selection_proportion=args("selection_proportion"),
        prompt_type=args("prompt_type"),
        classwise_topk=args("classwise_topk"),
        distance_type=args("distance_type"),
        topk=args("topk"),
        subselect=args("subselect"),
        decay_factor=args("decay_factor"),
        shots=args("shots"),
        salt=salt,
        algorithm="zero-shot",
        dataset_name=dataset_name,
        iteration=log_iteration,
        lm4cv_kwarg_set=args("lm4cv_kwarg_set"),
    )
    print(f"Loaded model from iteration {start_iteration} onto device {device}")
    model.initialize(
        clip_model_name=clip_model_name,
        scoring_clip_model_name=args("scoring_clip_model_name"),
        open_clip=args("use_open_clip"),
        device=device,
        cls2index=dataset.cls2index,
        classes=dataset.classes,
        deduplicate_descriptors=args("deduplicate_descriptors"),
    )
    if return_descriptors:
        return model, model.get_descriptors()
    return model


def get_dataset(
    dataset_name,
    clip_model_name="ViT-L/14",
    device="cpu",
    use_open_clip=False,
    image_size=224,
):
    """
    Return train, test, val sets from your custom dataset_loader.
    """
    train_dataset, test_dataset, val_dataset = get_processed_dataset(
        clip_model_name=clip_model_name,
        dataset_name=dataset_name,
        device=device,
        use_open_clip=use_open_clip,
        image_size=image_size,
    )
    return train_dataset, test_dataset, val_dataset


def get_scores(model, val_dataset, test_dataset, verbose_output=False):
    """
    Returns predicted CLIP scores for the val/test sets.
    If verbose_output=True, it also returns the raw (class, attribute) scores.
    """

    # Compute val/test scores
    val_scores = model.calculate_clip_scores(val_dataset.images)
    test_scores = model.calculate_clip_scores(test_dataset.images)

    val_labels = torch.tensor(val_dataset.labels)
    test_labels = torch.tensor(test_dataset.labels)

    # Average out the attribute dimension for each class, if that's your logic
    fin_val_scores = torch.stack([v.mean(-1) for v in val_scores], dim=1)
    fin_test_scores = torch.stack([v.mean(-1) for v in test_scores], dim=1)

    if verbose_output:
        return (
            val_labels,
            val_scores,
            fin_val_scores,
            test_labels,
            test_scores,
            fin_test_scores,
        )

    return val_labels, fin_val_scores, test_labels, fin_test_scores


def get_model_attrs(
    idx,
    label,
    image_scores,
    descriptors,
    all_scores,
    classes,
    baseline_prediction_label=None,
):
    """
    Returns top descriptors (and their scores) for two predictions:
      - X (pred_x) = the 'primary' index
      - Y (pred_y) = the 'secondary' index
    baseline_prediction_label can override pred_y if needed.
    """
    # If user gives baseline_prediction_label, then pred_x is the new model's best,
    # but pred_y is the baseline's chosen index (or vice versa).
    if baseline_prediction_label is not None:
        pred_x = image_scores.argmax().item()
        pred_y = baseline_prediction_label
    else:
        # pred_x = ground-truth label, pred_y = best predicted label
        pred_x = label
        pred_y = image_scores.argmax().item()

    pred_x = int(pred_x)
    pred_y = int(pred_y)
    score_x = image_scores[pred_x].item()
    score_y = image_scores[pred_y].item()

    x = classes[pred_x]
    y = classes[pred_y]

    weighted_embeddings_x = all_scores[pred_x][idx]
    weighted_embeddings_y = all_scores[pred_y][idx]

    attrs_x = descriptors[x]
    attrs_y = descriptors[y]

    best_attr_idxs_x = weighted_embeddings_x.argsort(descending=True)
    best_attr_idxs_y = weighted_embeddings_y.argsort(descending=True)

    best_attrs_pair_x = [
        (attrs_x[i], weighted_embeddings_x[i].item()) for i in best_attr_idxs_x
    ]
    best_attrs_pair_y = [
        (attrs_y[i], weighted_embeddings_y[i].item()) for i in best_attr_idxs_y
    ]

    # Remove duplicates while preserving score order
    uniq_best_attrs_pair_x = sorted(
        list(set(best_attrs_pair_x)), key=lambda z: z[1], reverse=True
    )
    uniq_best_attrs_pair_y = sorted(
        list(set(best_attrs_pair_y)), key=lambda z: z[1], reverse=True
    )

    best_attrs_pair_x = uniq_best_attrs_pair_x[:10]
    best_attrs_pair_y = uniq_best_attrs_pair_y[:10]

    return {
        "x": x,
        "y": y,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "score_x": score_x,
        "score_y": score_y,
        "attrs_x": attrs_x,
        "attrs_y": attrs_y,
        "best_attrs_pair_x": best_attrs_pair_x,
        "best_attrs_pair_y": best_attrs_pair_y,
    }
