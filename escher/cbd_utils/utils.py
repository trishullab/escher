import json
import pickle
import string
from multiprocessing.pool import ThreadPool
from typing import List, Tuple

import clip
import Levenshtein
import networkx as nx
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from escher.cbd_utils.gpt_utils import (
    get_class_suggestions,
    get_class_suggestions_programmatic,
    get_class_suggestions_w_descriptors,
    get_class_suggestions_w_descriptors_history,
    parse_program,
)

from .gpt_utils import cache_completion, openai_client, vllm_client


def load_clip_model(clip_model_name, use_open_clip, device, return_preprocessor=False):
    print("using clip model:", clip_model_name, "use_open_clip", use_open_clip)
    if use_open_clip and "hf-hub" in clip_model_name:
        clip_model, preprocess = open_clip.create_model_from_pretrained(
            clip_model_name, device=device
        )
        tokenizer = open_clip.get_tokenizer(clip_model_name)
    # elif 'google' in clip_model_name:
    #     from transformers import AutoProcessor, AutoModel
    #     clip_model = AutoModel.from_pretrained(clip_model_name).to(device)
    #     preprocess = AutoProcessor.from_pretrained(clip_model_name)
    #     tokenizer = preprocess.tokenize
    elif use_open_clip:
        clip_model_name = clip_model_name.replace("/", "-")
        name2preptrained = {
            "ViT-B-32": "laion2b_s34b_b79k",
            "ViT-B-16": "laion2b_s34b_b88k",
            "ViT-L-14": "laion2b_s32b_b82k",
            "ViT-H-14": "laion2b_s32b_b79k",
            "ViT-H-14-quickgelu": "metaclip_fullcc",  # default
        }
        if ":" in clip_model_name:
            clip_model_name, pretrained = clip_model_name.split(":")
            name2preptrained[clip_model_name] = pretrained
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "{}".format(clip_model_name),
            device=device,
            pretrained=(
                name2preptrained[clip_model_name]
                if clip_model_name in name2preptrained
                else None
            ),
        )
        tokenizer = open_clip.get_tokenizer(clip_model_name)
    else:
        clip_model, preprocess = clip.load("{}".format(clip_model_name), device=device)
        tokenizer = clip.tokenize

    if return_preprocessor:
        return clip_model, tokenizer, preprocess
    return clip_model, tokenizer


def clip_forward(model, image, text):
    image_embeddings = model.encode_image(image)
    text_embeddings = model.encode_text(text)

    score = (
        F.normalize(image_embeddings, dim=-1) @ F.normalize(text_embeddings, dim=-1).T
    )

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    if "logit_bias" in model.__dict__ and model.logit_bias is not None:
        logit_bias = model.logit_bias
    else:
        logit_bias = 0

    logits_per_image = (logit_scale * score) + logit_bias
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


# def get_clip_text_embeddings(prompts, clip_model, tokenizer, device):
#     res = []

#     def process_text(prompts):
#         token = torch.cat([tokenizer(prompt, context_length=clip_model.context_length) for prompt in prompts]).to(device)
#         with torch.no_grad():
#             txt_feat = clip_model.encode_text(token)
#         return txt_feat.to(torch.float32).to(device)

#     bs = 2048
#     for i in tqdm(range(np.ceil(len(prompts) / bs).astype(int))):
#         res.append(process_text(prompts[i * bs : (i + 1) * bs]))
#     return torch.concatenate(res, dim=0).cpu()


# def calculate_text_embeddings(
#     cls2descriptions, cls2index, clip_model, tokenizer, device
# ):
#     cls2embeddings = [None for _ in cls2descriptions]
#     for cls_name in cls2descriptions:
#         cls_index = cls2index[cls_name]
#         cls2embeddings[cls_index] = get_clip_text_embeddings(
#             cls2descriptions[cls_name], clip_model, tokenizer, device
#         )
#     assert all([z is not None for z in cls2embeddings])  # check that there are no Nones
#     return cls2embeddings


def get_clip_text_embeddings(prompts, clip_model, tokenizer, device, batch_size=2048):
    res = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="getting text embeddings"):
        batch_prompts = prompts[i : i + batch_size]
        with torch.no_grad():
            tokens = tokenizer(
                batch_prompts, context_length=clip_model.context_length
            ).to(device)
            txt_feat = clip_model.encode_text(tokens)
            txt_feat = txt_feat.to(torch.float32).cpu()
            res.append(txt_feat)
    return torch.cat(res, dim=0)


def calculate_text_embeddings(
    cls2descriptions, cls2index, clip_model, tokenizer, device
):
    all_prompts = []
    descriptor_indices = []

    # Flatten all descriptors and keep track of their class indices
    for cls_name in cls2descriptions:
        cls_index = cls2index[cls_name]
        descriptions = cls2descriptions[cls_name]
        all_prompts.extend(descriptions)
        descriptor_indices.extend([cls_index] * len(descriptions))

    # Process all descriptors together
    embeddings = get_clip_text_embeddings(all_prompts, clip_model, tokenizer, device)

    # Group embeddings by class
    cls2embeddings = [[] for _ in range(len(cls2descriptions))]
    for idx, cls_index in enumerate(descriptor_indices):
        cls2embeddings[cls_index].append(embeddings[idx])

    # Convert lists of embeddings to tensors
    for i in range(len(cls2embeddings)):
        cls2embeddings[i] = torch.stack(cls2embeddings[i])

    return cls2embeddings


# def calculate_scores(
#     image_embeddings: torch.Tensor,
#     text_embeddings: List[torch.Tensor],
#     model_scaling: Tuple = None,
# ) -> List[np.ndarray]:
#     """
#     Returns a list L.
#     L[i] contains a (n_images, n_descriptors) shaped numpy array of clip scores for class i.
#     """
#     unreduced_scores = []
#     for class_index in range(len(text_embeddings)):
#         text_embeddings_for_class = text_embeddings[class_index].to(
#             image_embeddings.dtype
#         )
#         score = (F.normalize(image_embeddings, dim=-1) @ F.normalize(text_embeddings_for_class, dim=-1).T)
#         if model_scaling is not None:
#             logit_scale, logit_bias = model_scaling
#             score = torch.sigmoid(score * logit_scale.exp() + logit_bias)
#         unreduced_scores.append(score)
#     return unreduced_scores


def calculate_scores(
    image_embeddings: torch.Tensor,
    text_embeddings: List[torch.Tensor],
    model_scaling: Tuple = None,
    batch_size: int = 2048,
) -> List[torch.Tensor]:
    """
    Returns a list L.
    L[i] contains a (n_images, n_descriptors) shaped tensor of clip scores for class i.
    """
    device = image_embeddings.device  # Ensure computations are on the correct device

    # Pre-normalize image embeddings once
    image_embeddings = F.normalize(image_embeddings, dim=-1).to(torch.float32)

    # Concatenate all normalized text embeddings
    normalized_text_embeddings = []
    class_lengths = []
    for t in text_embeddings:
        t = F.normalize(t.to(torch.float32), dim=-1)
        normalized_text_embeddings.append(t)
        class_lengths.append(t.shape[0])

    # Concatenate and move to device
    normalized_text_embeddings = torch.cat(normalized_text_embeddings, dim=0).to(device)

    n_images = image_embeddings.shape[0]
    normalized_text_embeddings.shape[0]

    # Compute scores in batches to manage memory
    scores_list = []
    for start_idx in range(0, n_images, batch_size):
        end_idx = min(start_idx + batch_size, n_images)
        batch_image_embeddings = image_embeddings[start_idx:end_idx].to(device)
        # Efficient matrix multiplication
        batch_scores = batch_image_embeddings @ normalized_text_embeddings.T

        if model_scaling is not None:
            logit_scale, logit_bias = model_scaling
            batch_scores = torch.sigmoid(batch_scores * logit_scale.exp() + logit_bias)

        scores_list.append(batch_scores.cpu())

    # Concatenate batch scores
    scores = torch.cat(scores_list, dim=0)

    # Split scores back into per-class lists
    unreduced_scores = []
    start = 0
    for length in class_lengths:
        end = start + length
        class_scores = scores[:, start:end]
        unreduced_scores.append(class_scores)
        start = end

    return unreduced_scores


def reduce_to_class_scores_by_mean(unreduced_scores):
    """
    Returns a tensor T of shape (n_images, n_classes).
    T[i, j] is the mean clip score on image i for the descriptors of class j.
    """
    scores = [arr.mean(1) for arr in unreduced_scores]  # (n_classes, n_images)
    return np.stack(scores).T  # (n_images, n_classes)


def load_obj(pth):
    ext = pth.split(".")[-1]
    if ext == "pkl":
        with open(pth, "rb") as f:
            return pickle.load(f)
    elif ext == "json":
        with open(pth, "r") as f:
            return json.load(f)
    elif ext == "yaml":
        with open(pth, "r") as f:
            return yaml.safe_load(f)
    elif ext == "txt":
        with open(pth, "r") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported extension: {ext}, for file: {pth}")


def save_obj(obj, pth):
    ext = pth.split(".")[-1]
    if ext == "pkl":
        with open(pth, "wb") as f:
            pickle.dump(obj, f)
    elif ext == "json":
        with open(pth, "w") as f:
            json.dump(obj, f)
    elif ext == "yaml":
        with open(pth, "w") as f:
            yaml.dump(obj, f)
    elif ext == "txt":
        with open(pth, "w") as f:
            f.write(obj)
    else:
        raise ValueError(f"Unsupported extension: {ext}, for file: {pth}")


def make_descriptor_sentence(class_name, descriptor):
    def helper():
        if descriptor.startswith("a") or descriptor.startswith("an"):
            return f"which is {descriptor}"
        elif (
            descriptor.startswith("has")
            or descriptor.startswith("often")
            or descriptor.startswith("typically")
            or descriptor.startswith("may")
            or descriptor.startswith("can")
        ):
            return f"which {descriptor}"
        elif descriptor.startswith("used"):
            return f"which is {descriptor}"
        else:
            return f"which has {descriptor}"

    return f"{class_name}, {helper()}"


def process_descriptions(cls2descriptions):
    result = {}
    for clsname, ds in cls2descriptions.items():
        result[clsname] = [make_descriptor_sentence(clsname, d) for d in ds]
        if len(result[clsname]) == 0:
            result[clsname] = [clsname]
    return result


def kg_to_cls2descriptions(kg, classes, should_process_descriptions=True):
    # converts a knowledge graph into cls2descriptions dict expected by LaBo.
    def dfs_succ(node, ignore):
        if node in ignore:
            return set()

        result = {node}
        ignore.add(node)
        for edge in kg.out_edges(node):
            result.update(dfs_succ(edge[1], ignore))
        return result

    def dfs_pred(node, ignore):
        if node in ignore:
            return set()

        result = {node}
        ignore.add(node)
        for edge in kg.in_edges(node):
            result.update(dfs_pred(edge[0], ignore))
        return result

    cls2descriptions = {}
    for cls in classes:
        attrs = set()
        ignore = set(classes) - set([cls])
        for node in dfs_pred(cls, set()):
            attrs.update(dfs_succ(node, ignore))
        cls2descriptions[cls] = list(attrs - set([cls]))
    if should_process_descriptions:
        cls2descriptions = process_descriptions(cls2descriptions)
    return cls2descriptions


def merge_graph_nodes(graph, node1, node2):
    """
    Merges node2 into node1 in the provided graph, combining edges.
    """
    # Move all edges from node2 to node1
    for successor in list(graph.successors(node2)):
        if successor != node1:
            graph.add_edge(node1, successor)
    for predecessor in list(graph.predecessors(node2)):
        if predecessor != node1:
            graph.add_edge(predecessor, node1)
    graph.remove_node(node2)
    return graph


def get_gpt_completions(
    messages_list,
    openai_model,
    temp,
    output_is_json=False,
    completion=False,
    **api_kwargs,
):
    # sends all message completion requests in parallel
    # assert output_is_json ^ completion, "output_is_json and completion cannot be both True"
    if openai_model.startswith("LOCAL:"):
        openai_model = openai_model.replace("LOCAL:", "")
        client = vllm_client
    else:
        client = openai_client

    @cache_completion("/var/local/atharvas/f/learning_descriptors/cache/cache.db")
    def get_gpt_completion(messages):
        if completion:
            return client.completions.create(
                model=openai_model, prompt=messages, temperature=temp, **api_kwargs
            )
        if output_is_json:
            return client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=temp,
                response_format={"type": "json_object"},
                **api_kwargs,
            )
        return client.chat.completions.create(
            model=openai_model, messages=messages, temperature=temp, **api_kwargs
        )

    print(f"getting {len(messages_list)} gpt completions")
    p = ThreadPool(128)
    completions = []
    it = tqdm(p.imap(get_gpt_completion, messages_list), desc="getting gpt completions")
    for x in it:
        completions.append(x)

    if completion:
        return [c.choices[0].text for c in completions]
    # completions = p.map(get_gpt_completion, messages_list)
    return [c.choices[0].message.content for c in completions]


def get_text_distance(a, b):
    a, b = (
        [c for c in a if c not in string.punctuation],
        [c for c in b if c not in string.punctuation],
    )
    a, b = "".join(a).lower(), "".join(b).lower()

    return Levenshtein.distance(a, b)


def map_cls2descriptions_class_names(cls2descriptions, class_names):
    """
    Sometimes the class names in cbd_descriptors is not the same as the class names in the dataset.
    For each class name in the dataset, we will find the closest key in cls2descriptions and use those descriptors.
    """
    result_cls2descriptions = {}
    lowered_keys = {c.lower(): c for c in cls2descriptions.keys()}
    for class_name in class_names:
        if class_name in lowered_keys:
            result_cls2descriptions[class_name] = cls2descriptions[
                lowered_keys[class_name]
            ]
            continue
        other_class_name = min(
            cls2descriptions.keys(),
            key=lambda x: get_text_distance(x, class_name),
        )
        result_cls2descriptions[class_name] = cls2descriptions[other_class_name]

    return result_cls2descriptions


def choose_confusion_candidates(
    potential_confusions, topk, resolution_history=None, decay_factor=None
):
    if (resolution_history is not None) and (decay_factor is not None):
        weight = np.array(
            [
                (
                    max(
                        cnt
                        * 0.5 ** (resolution_history[curr, candidate] / decay_factor),
                        0,
                    )
                    if decay_factor > 0
                    else cnt
                )
                for curr, candidate, cnt in potential_confusions
            ]
        )
    else:
        weight = np.array([max(cnt, 0) for _, _, cnt in potential_confusions])

    if weight.sum() > 0:
        weight = weight / weight.sum()
    else:
        weight = np.ones_like(weight) / len(weight)  # Fallback if all weights are zero

    assert abs(sum(weight) - 1) < 0.001
    choices_indices = np.random.choice(
        np.arange(len(potential_confusions)),
        size=min(len(potential_confusions), abs(topk)),
        p=weight,
        replace=False,
    )
    choices = [potential_confusions[i] for i in choices_indices]

    return choices


def construct_gpt_queries(
    choices,
    prompt_type,
    classes,
    descriptors,
    raw_descriptors,
    correlation_matrix,
    conversational_history=None,
):
    class_indices = []
    gpt_queries = []
    for i, candidate, cnt in choices:
        match prompt_type:
            case "confound":
                gpt_queries.append(
                    get_class_suggestions(classes[i], classes[candidate])
                )
            case "confound_w_descriptors":
                if conversational_history:
                    gpt_queries.append(
                        get_class_suggestions_w_descriptors_history(
                            classes[i],
                            classes[candidate],
                            descriptors[classes[i]],
                            descriptors[classes[candidate]],
                            conversational_history,
                        )
                    )
                else:
                    gpt_queries.append(
                        get_class_suggestions_w_descriptors(
                            classes[i],
                            classes[candidate],
                            descriptors[classes[i]],
                            descriptors[classes[candidate]],
                        )
                    )
            case "programmatic":
                gpt_queries.append(
                    get_class_suggestions_programmatic(
                        cls_desc_pairs=[
                            (
                                classes[i],
                                raw_descriptors[classes[i]],
                            ),
                            (
                                classes[candidate],
                                raw_descriptors[classes[candidate]],
                            ),
                        ],
                        correlation=correlation_matrix[i, candidate].item() * 100,
                    )
                )
        # we are going to take the suggestions returned by GPT for the query we just added
        # and add it to class i
        class_indices.append(i)

    return gpt_queries, class_indices


def parse_gpt_output(prompt_type, gpt_output):
    if prompt_type == "programmatic":
        parsed_suggestions = [
            parse_program(json.loads(s)["program"]) for s in gpt_output
        ]
    else:
        # parsed_suggestions = [json.loads(s)["features"] for s in gpt_output]
        parsed_suggestions = []
        for i, s in enumerate(gpt_output):
            try:
                parsed_suggestions.append(json.loads(s)["features"])
            except Exception as e:
                print("failed to parse suggestion:", s)
                print("error:", e)
    return parsed_suggestions


def populate_conversational_history(
    iteration, conversational_history, correlation_matrix, classes, distance_type
):
    for hi, history in enumerate(conversational_history[iteration]):
        clsA, clsB = history["classes"]
        clsA_idx = classes.index(clsA)
        clsA_predictions = correlation_matrix[clsA_idx]
        if distance_type == "pearson":
            error_rate = clsA_predictions[classes.index(clsB)]
        elif distance_type == "pca" or distance_type == "emd":
            error_rate = (
                1 - clsA_predictions[classes.index(clsB)] / clsA_predictions.max()
            )
        else:
            error_rate = clsA_predictions[classes.index(clsB)] / clsA_predictions.sum()
        conversational_history[iteration][hi]["feedback"] = error_rate
    return conversational_history
