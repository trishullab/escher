from collections import defaultdict

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# def pairwise_emd(A):
#     n_class = A.shape[1]
#     emd_matrix = np.zeros((n_class, n_class))

#     for i in range(n_class):
#         for j in range(i, n_class):
#             if i >= j:
#                 continue
#             # Calculate EMD between class i and class j using Wasserstein distance
#             emd = wasserstein_distance(
#                 list(range(A.shape[0])), list(range(A.shape[0])), A[:, i], A[:, j]
#             )
#             emd_matrix[i, j] = emd
#             emd_matrix[j, i] = emd  # Symmetry

#     zero_one_normalized_emd_matrix = (emd_matrix - emd_matrix.min()) / (
#         emd_matrix.max() - emd_matrix.min()
#     )
#     return zero_one_normalized_emd_matrix


def pairwise_emd(A):
    # Normalize columns of A to sum to 1 (probability distributions)
    A_normalized = A / np.sum(A, axis=0, keepdims=True)

    # Compute cumulative distribution functions (CDFs)
    # Compute pairwise absolute differences between CDFs
    # Sum over the positions to get the Wasserstein distances
    cdf_A = np.cumsum(A_normalized, axis=0)
    abs_diff = np.abs(cdf_A[:, :, np.newaxis] - cdf_A[:, np.newaxis, :])
    emd_matrix = np.sum(abs_diff, axis=0)
    # Normalize the EMD matrix to be between 0 and 1
    emd_min = emd_matrix.min()
    emd_max = emd_matrix.max()
    zero_one_normalized_emd_matrix = (emd_matrix - emd_min) / (emd_max - emd_min)
    # return emd_matrix
    return zero_one_normalized_emd_matrix


def get_confusion_matrix(logits, classes, topk):
    predictions = logits.softmax(dim=-1)
    conf_maxpred = predictions.argsort(dim=-1, descending=True)[:, 1:]
    conf_maxvals = predictions.gather(1, conf_maxpred)

    if topk < 0:
        nstd = -topk
        mean_vals = conf_maxvals.mean(dim=1, keepdim=True)
        std_vals = conf_maxvals.std(dim=1, unbiased=False, keepdim=True)
        threshold = mean_vals + nstd * std_vals
        mask_conf_maxvals = conf_maxvals < threshold
    else:
        indices = torch.arange(conf_maxvals.size(1), device=conf_maxvals.device)
        mask_conf_maxvals = indices.unsqueeze(0) >= topk

    conf_maxpred = conf_maxpred.masked_fill(mask_conf_maxvals, -1)

    # Filter predictions and pad to max length
    lengths = (conf_maxpred != -1).sum(dim=1)
    max_len = lengths.max()
    padded_preds = torch.full(
        (conf_maxpred.size(0), max_len),
        -1,
        dtype=torch.int64,
        device=conf_maxpred.device,
    )
    for i in range(conf_maxpred.size(0)):
        valid_length = lengths[i]
        if valid_length > 0:
            padded_preds[i, :valid_length] = conf_maxpred[i][conf_maxpred[i] != -1]

    # Compute confusion matrix efficiently
    num_classes = len(classes)
    pair_i = padded_preds.unsqueeze(2).expand(-1, -1, max_len)
    pair_j = padded_preds.unsqueeze(1).expand(-1, max_len, -1)
    valid_mask = (pair_i != -1) & (pair_j != -1)
    pair_i = pair_i[valid_mask]
    pair_j = pair_j[valid_mask]
    indices = pair_i * num_classes + pair_j
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion_matrix = counts.view(num_classes, num_classes)
    return confusion_matrix


def get_true_confusion_matrix(predictions, classes, labels, perc_labels):
    # Perc labels controls how many labels to consider for each class.
    # For each class, we will consider the first p% of the labels.
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy().astype(int)
    N = len(classes)

    label_idxs = defaultdict(list)
    for i, label in enumerate(labels):
        label_idxs[label].append(i)
    filt_idxs = []
    for _, idxs in label_idxs.items():
        n_to_select = int(perc_labels * len(idxs))
        filt_idxs.extend(idxs[:n_to_select])
    filt_idxs = np.array(filt_idxs)
    filt_idxs.sort()
    filtered_predictions = predictions[filt_idxs]
    filtered_labels = labels[filt_idxs]

    confusion_matrix = np.zeros([N] * 2)  # score_matrix[label, pred] = cnt
    for label, pred in zip(filtered_labels, filtered_predictions):
        confusion_matrix[label, pred] += 1

    return confusion_matrix


def calculate_class_correlation(
    classes, logits, report=None, topk=3, distance_type="emd", optimization_type=""
):
    confusion_matrix = get_confusion_matrix(torch.tensor(logits), classes, topk).numpy()
    predictions = torch.tensor(logits).softmax(-1).numpy()
    # correlation = np.corrcoef(predictions.T)
    # for each class, find the N classes with the lowest EMD
    if distance_type == "emd":
        classwise_distance = pairwise_emd(logits - logits.min())
        optimization_type = "maximize" if optimization_type == "" else optimization_type
    elif distance_type == "cosine":
        classwise_distance = (predictions.T @ predictions) / (
            np.linalg.norm(predictions.T, axis=1)[:, None]
            @ np.linalg.norm(predictions.T, axis=1)[None, :]
        )
        optimization_type = "minimize" if optimization_type == "" else optimization_type
    elif distance_type == "pearson":
        classwise_distance = np.corrcoef(logits.T)
        optimization_type = "minimize" if optimization_type == "" else optimization_type
    elif distance_type == "confusion":
        classwise_distance = confusion_matrix.copy()
        optimization_type = "minimize" if optimization_type == "" else optimization_type
    elif distance_type == "pca":
        pca = PCA(n_components=len(classes))
        reduced_logits = pca.fit_transform(logits.T)
        # Compute pairwise distances between classes in PCA space
        distances = pdist(reduced_logits, metric="euclidean")
        classwise_distance = squareform(distances)  # Shape: (n_classes, n_classes)
        optimization_type = "maximize" if optimization_type == "" else optimization_type
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")

    maxpred = classwise_distance.argsort()
    maxvals = classwise_distance[
        np.arange(classwise_distance.shape[0])[:, None], maxpred
    ]

    if topk < 0:
        nstd = -1 * topk
        if optimization_type == "minimize":
            # for an ideal classifier, we want to minimize similarity with this distance metric.
            # so the confused classes have a high value.
            # We will select the classes that are nstd _larger_ than the mean.
            # mask_maxvals = (maxvals.mean(1) - (nstd * maxvals.std(1)))[:, None] > maxvals
            # mask_maxvals = (maxvals.mean(1) + (nstd * maxvals.std(1)))[:, None] < maxvals
            mask_maxvals = (maxvals.mean(1) + (nstd * maxvals.std(1)))[
                :, None
            ] >= maxvals
        else:
            # This will select the classes that are nstd _smaller_ than the mean.
            # We want to modify these classes so the distance is larger and hence
            # optimzed to be maximally dissimilar.
            mask_maxvals = (maxvals.mean(1) - (nstd * maxvals.std(1)))[
                :, None
            ] <= maxvals
    else:
        if optimization_type == "minimize":
            # Only keep the topk indices off the right side of the array
            mask_maxvals = (
                np.arange(maxvals.shape[1])[None, ::-1].repeat(maxvals.shape[0], axis=0)
                >= topk
            )
        else:
            # Only keep the topk indices off the left side of the array
            mask_maxvals = (
                np.arange(maxvals.shape[1])[None, :].repeat(maxvals.shape[0], axis=0)
                >= topk
            )
    if distance_type == "confusion":
        # any value that is 0 should also be masked out
        mask_maxvals = mask_maxvals | (maxvals == 0)

    if distance_type == "pearson" or distance_type == "cosine":
        # any value < 0 should also be masked out
        # also, a similarity < 0.80 is not worth considering.
        mask_maxvals = mask_maxvals | (maxvals < 0)
        mask_maxvals = mask_maxvals | (maxvals < 0.80)

    maxpred[mask_maxvals] = -1
    correlated_classes = {}
    correlation_list = []
    for i, c1 in enumerate(classes):
        if report and (report[str(i)]["f1-score"] >= 0.9):
            continue
        # cv, ci = torch.tensor(-1*classwise_distance[i])
        best_idxs = maxpred[i][maxpred[i] != -1]
        for idx in best_idxs:
            c2 = classes[idx]
            if c2 == c1:
                continue
            if c1 not in correlated_classes:
                correlated_classes[c1] = []
            correlated_classes[c1].append(classes[idx.item()])
            # (index of class 1, index of class 2, EMD)
            correlation_list.append((i, idx, classwise_distance[i, idx]))

    return classwise_distance, correlation_list


def get_confusions(model, dataloader, classes):
    N = len(classes)
    confusion_matrix = np.zeros([N] * 2)  # score_matrix[label, pred] = cnt
    all_logits = []
    all_preds, all_labels = [], []
    for batch in dataloader:
        images, labels = batch
        with torch.no_grad():
            logits = model.to(images.device).forward(images)  # (N, C)
            predictions = logits.argmax(-1).cpu()
        assert len(labels) == len(predictions)
        for label_index, pred_index in zip(labels, predictions):
            confusion_matrix[label_index, pred_index] += 1
        all_logits.append(logits.cpu().numpy())
        all_preds.append(predictions.numpy())
        all_labels.append(labels.cpu().numpy())
    score_matrix = confusion_matrix.copy()
    score_matrix[
        np.arange(N), np.arange(N)
    ] -= 1e9  # we cannot confuse class X with class X of course...
    score_matrix = score_matrix.flatten()
    indices = np.argsort(score_matrix)
    all_logits = np.concatenate(all_logits)
    # returns a list of clsA, clsB, similarity score
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    str_labels = [str(i) for i in range(len(classes))]
    report = classification_report(
        all_labels,
        all_preds,
        target_names=str_labels,
        zero_division=0,
        output_dict=True,
    )

    return (
        [(i // N, i % N, score_matrix[i]) for i in reversed(indices)],
        confusion_matrix,
        all_logits,
        report,
    )
