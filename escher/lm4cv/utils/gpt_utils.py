import ast
import functools
import json
import os
import pickle
import re
import sqlite3
import threading

import clip
import networkx as nx
import numpy as np
import openai
import torch
from tqdm.auto import tqdm
from escher.cbd_utils.server import openai_client, OPENAI_TEMP, OPENAI_MODEL
from escher.cbd_utils import find_json_block, lock


def message_builder_cls(cls):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that returns output as a list of JSON objects with the feature label (make sure to give enough visual details), the relation to the object (one of `attribute_of` or `subclass_of`), and the probability of visually identifying the object based on this feature.",
        },
        {
            "role": "user",
            "content": "What visual features can be used to identify a bald eagle? A visual feature is a characteristic of an object that can be observed visually.",
        },
        {
            "role": "assistant",
            "content": """[
  {
    "feature_label": "white head",
    "relation_to_object": "attribute_of",
    "importance": 0.9
  },
  {
    "feature_label": "curved yellow beak",
    "relation_to_object": "attribute_of",
    "importance": 0.8
  },
  {
    "feature_label": "white tail feathers",
    "relation_to_object": "attribute_of",
    "importance": 0.7
  },
  {
    "feature_label": "yellow talons",
    "relation_to_object": "attribute_of",
    "importance": 0.7
  },
  {
    "feature_label": "eagle",
    "relation_to_object": "subclass_of",
    "importance": 0.8
  },
  {
    "feature_label": "bird",
    "relation_to_object": "subclass_of",
    "importance": 0.8
  }
]""",
        },
        {
            "role": "user",
            "content": f"What visual features can be used to identify a {cls}? A visual feature is a characteristic of an object that can be observed visually.",
        },
    ]


def message_builder_question(question):
    question_proc = question.strip().replace("\n", "")
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that returns output as a list of JSON objects with the feature label (make sure to give enough visual details), the relation to the object (one of `attribute_of` or `subclass_of`), and the probability of visually identifying the object based on this feature.",
        },
        {
            "role": "user",
            "content": 'What visual features will be helpful to answer the question:  "Is this a picture of a bald eagle? Please answer yes or no." A visual feature is a characteristic of an object that can be observed visually.',
        },
        {
            "role": "assistant",
            "content": """[
  {
    "feature_label": "white head",
    "relation_to_object": "attribute_of",
    "importance": 0.9
  },
  {
    "feature_label": "curved yellow beak",
    "relation_to_object": "attribute_of",
    "importance": 0.8
  },
  {
    "feature_label": "white tail feathers",
    "relation_to_object": "attribute_of",
    "importance": 0.7
  },
  {
    "feature_label": "yellow talons",
    "relation_to_object": "attribute_of",
    "importance": 0.7
  },
  {
    "feature_label": "eagle",
    "relation_to_object": "subclass_of",
    "importance": 0.8
  },
  {
    "feature_label": "bird",
    "relation_to_object": "subclass_of",
    "importance": 0.8
  }
]""",
        },
        {
            "role": "user",
            "content": f'What visual features will be helpful to answer the question: "{question_proc}" A visual feature is a characteristic of an object that can be observed visually.',
        },
    ]


def establish_kg_db(db_loc):
    conn = sqlite3.connect(db_loc)
    c = conn.cursor()
    # Create table
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS completions
    (argument_blob BLOB, result_blob BLOB)
    """
    )
    conn.commit()
    conn.close()


def get_db_connection(db_loc):
    conn = sqlite3.connect(db_loc)
    c = conn.cursor()
    return c, conn


def cache_completion(db_loc):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            c, conn = get_db_connection(db_loc)
            try:
                # Serialize arguments to a blob
                args_blob = pickle.dumps((args, kwargs))

                # Check the cache
                with lock:
                    c.execute(
                        "SELECT result_blob FROM completions WHERE argument_blob = ?",
                        (args_blob,),
                    )
                    result = c.fetchone()
                    if result:
                        # If found in cache, return the deserialized result
                        return pickle.loads(result[0])

                # Compute the result since it's not cached
                result = f(*args, **kwargs)

                # Serialize and store the result
                result_blob = pickle.dumps(result)
                with lock:
                    c.execute(
                        "INSERT INTO completions (argument_blob, result_blob) VALUES (?, ?)",
                        (args_blob, result_blob),
                    )
                    conn.commit()
                return result
            finally:
                conn.close()

        return wrapped

    return decorator


# def establish_kg_db(db_loc):
#     conn = sqlite3.connect(db_loc)
#     c = conn.cursor()
#     # Create table
#     c.execute('''
#     CREATE TABLE IF NOT EXISTS completions
#     (argument_blob BLOB, result_blob BLOB)
#     ''')
#     conn.commit()
#     return c, conn


# def cache_completion(c, conn):
#     def decorator(f):
#         @functools.wraps(f)
#         def wrapped(*args, **kwargs):
#             # Serialize arguments to a blob and the name of the function
#             args_blob = pickle.dumps((args, kwargs))

#             # Check the cache
#             c.execute('SELECT result_blob FROM completions WHERE argument_blob = ?', (args_blob,))
#             result = c.fetchone()
#             if result:
#                 # If found in cache, return the deserialized result
#                 return pickle.loads(result[0])

#             # Compute the result since it's not cached
#             result = f(*args, **kwargs)

#             # Serialize and store the result
#             result_blob = pickle.dumps(result)
#             c.execute('INSERT INTO completions (argument_blob, result_blob) VALUES (?, ?)', (args_blob, result_blob))
#             conn.commit()
#             return result
#         return wrapped
#     return decorator


def get_embedding(attribute):
    embedding_resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=attribute,
    )
    embedding = np.array(embedding_resp.data[0].embedding)
    return embedding


def gpt_parse_suggestions(suggestions):
    if not len(suggestions):
        return {}
    try:
        suggestions = json.loads(suggestions)
        return suggestions
    except json.JSONDecodeError:
        # find the ```json block```
        if match := find_json_block.search(suggestions):
            suggestions = match.group(1)
            try:
                suggestions = json.loads(suggestions)
                return suggestions
            except json.JSONDecodeError:
                try:
                    suggestions = ast.literal_eval(suggestions)
                    return suggestions
                except Exception:
                    print(
                        f"Could not decode JSON w/Python literal eval : {suggestions}"
                    )
                print(f"Could not decode JSON : {suggestions}")
                return None
        print(f"Could not find JSON : {suggestions}")
        return None


def message_builder(cls):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that returns output as a list of JSON objects with the feature label (make sure to give enough visual details), and the probability of visually identifying the object based on this feature.",
        },
        {
            "role": "user",
            "content": 'What visual features can be used to answer the question: "is this an image of a bald eagle? Answer yes or no." A visual feature is a characteristic of an object that can be observed visually.',
        },
        {
            "role": "assistant",
            "content": """{
  "feature_label": "bald eagle",
  "visual_features": [
    {
      "feature_label": "white head",
      "importance": 0.9
    },
    {
      "feature_label": "curved yellow beak",
      "importance": 0.8
    },
    {
      "feature_label": "white tail feathers",
      "importance": 0.7
    },
    {
      "feature_label": "yellow talons",
      "importance": 0.7
    }
  ]
}""",
        },
        {
            "role": "user",
            "content": f'What visual features can be used to answer the question "Is this an image of a {cls}? Answer yes or no." A visual feature is a characteristic of an object that can be observed visually.',
        },
    ]


def construct_confusion_resolution_message(competing_classes):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Generate new and distinct visual feature labels to accurately answer the given visual reasoning questions.",
                }
            ],
        }
    ]

    user_content = "Given the confusion experienced by a visual reasoning model in accurately answering the questions due to the existing feature sets, your task is to generate new and distinct visual feature labels for each question. These new labels should specifically target aspects that reduce misclassification and enhance detection accuracy.\n"

    question_feature_dict = {}
    for i, (question, features) in enumerate(competing_classes, start=1):
        user_content += f"1. **Question: '{question}'**\n- Current Feature Labels: {features}.\n- **Objective:** Propose at least three innovative feature labels that could enhance the accuracy of answering this question. Focus on features that are specifically observable and identifiable in images where this question might be relevant.\n"
        question_feature_dict[question] = []

    user_content += "For each new feature label, explain its relevance to the specific question, its potential in reducing confusion, and how it could improve the model's performance in identifying these items accurately in images. Consider the common points of confusion in current model responses while suggesting these new labels. At the end, also output a JSON object in this format:\n```json\n"
    json_content = "{\n"
    for question in question_feature_dict.keys():
        json_content += f'"{question}": ["feature1", "feature2", ..., "featureN"],\n'
    json_content = json_content.rstrip(",\n") + "\n}"
    user_content += json_content + "\n```"
    messages.append(
        {"role": "user", "content": [{"type": "text", "text": user_content}]}
    )
    return messages

# @cache_completion("/mnt/sdd1/atharvas/lmms-eval/lmms_eval/_common/data/cached.db")
def resolve_class_confusion(competing_classes, questiontype="iteration0"):
    messages = construct_confusion_resolution_message(competing_classes)
    # messages = [
    #     {"role": "system", "content": [{"type": "text", "text": "Generate new and distinct visual feature labels to accurately answer the given visual reasoning questions."}]},
    #     {"role": "user", "content": [{"type": "text", "text": "Given the confusion experienced by a visual reasoning model in accurately answering the questions '{question1}' and '{question2}' due to the existing feature sets, your task is to generate new and distinct visual feature labels for each question. These new labels should specifically target aspects that reduce misclassification and enhance detection accuracy.\n1. **Question: '{question1}'**\n- Current Feature Labels: {question1_features}.\n- **Objective:** Propose at least three innovative feature labels that could enhance the accuracy of answering this question. Focus on features that are specifically observable and identifiable in images where this question might be relevant.\n2. **Question: '{question2}'**\n- Current Feature Labels: {question2_features}.\n- **Objective:** Identify at least three new feature labels that could improve the precision of answering this question. Concentrate on visible, distinguishable characteristics in images that would help in accurately responding, focusing solely on observable features in the image.\nFor each new feature label, explain its relevance to the specific question, its potential in reducing confusion, and how it could improve the model's performance in identifying these items accurately in images. Consider the common points of confusion in current model responses while suggesting these new labels. At the end, also output a JSON object in this format:\n```json\n{{\n\"{question1}\": [\"feature1\", \"feature2\", ..., \"featureN\"],\n\"{question2}\": [\"feature1\", \"feature2\", ..., \"featureN\"]\n}}\n```".format(question1=question1, question2=question2, question1_features=question1_features, question2_features=question2_features) }]},
    # ]
    completion = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=OPENAI_TEMP,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return completion.choices[0].message.content


def process_class(cls1, class_correlation, max_classes_in_request, concept_graph):
    raw_suggestions = {}
    suggestions = {}

    # concept_graph, visual_features_cls = get_completion(cls1, concept_graph)
    visual_features_cls = list(concept_graph.neighbors(cls1))
    cls1_features = ", ".join(visual_features_cls)
    competing_classes = []
    for other_cls in class_correlation[cls1]:
        if other_cls == cls1:
            continue

        visual_features_other_cls = list(concept_graph.neighbors(other_cls))
        other_cls_features = ", ".join(visual_features_other_cls)
        competing_classes.append((other_cls, other_cls_features))

    if len(competing_classes) == 0:
        return cls1, raw_suggestions, suggestions

    if len(competing_classes) <= max_classes_in_request:
        cclasses = [(cls1, cls1_features)] + competing_classes
        raw_suggestion = resolve_class_confusion(cclasses, questiontype="iteration0")
        suggestion = gpt_parse_suggestions(raw_suggestion)
        raw_suggestions[cls1] = raw_suggestion
        suggestions[cls1] = suggestion
    else:
        splits = [
            competing_classes[i : i + max_classes_in_request]
            for i in range(0, len(competing_classes), max_classes_in_request)
        ]
        agg_suggestion = {}
        agg_raw_suggestion = ""
        for split in splits:
            cclasses = set([(cls1, cls1_features)] + split)
            raw_suggestion = resolve_class_confusion(
                cclasses, questiontype="iteration0"
            )
            suggestion = gpt_parse_suggestions(raw_suggestion)
            agg_raw_suggestion += raw_suggestion
            if not suggestion:
                break
            for c, suggestion in suggestion.items():
                if c not in agg_suggestion:
                    agg_suggestion[c] = set()
                agg_suggestion[c].update(suggestion)
        raw_suggestions[cls1] = raw_suggestion
        suggestions[cls1] = agg_suggestion

    return cls1, raw_suggestions, suggestions


def batchify_run(process_fn, data_lst, res, batch_size, use_tqdm=False):
    data_lst_len = len(data_lst)
    num_batch = np.ceil(data_lst_len / batch_size).astype(int)
    iterator = range(num_batch)
    if use_tqdm:
        iterator = tqdm(iterator)
    for i in iterator:
        batch_data = data_lst[i * batch_size : (i + 1) * batch_size]
        batch_res = process_fn(batch_data)
        res[i * batch_size : (i + 1) * batch_size] = batch_res
        del batch_res


def get_clip_embeddings(prompts, clip_model, device, latent_dim, use_tqdm=False):
    res = torch.empty((len(prompts), latent_dim), device=device)

    def process_text(prompts):
        token = torch.cat(
            [
                clip.tokenize(
                    f"An image of a {prompt}" if "?" not in prompt else prompt
                )
                for prompt in prompts
            ]
        ).to(device)
        with torch.no_grad():
            txt_feat = clip_model.encode_text(token)
        return txt_feat.to(torch.float32).to(device)

    batchify_run(process_text, prompts, res, 2048, use_tqdm=use_tqdm)
    return res


def get_clip_image_embeddings(
    images, clip_model, device, latent_dim, preprocess, use_tqdm=False
):
    res = torch.empty((len(images), latent_dim), device=device)

    def process_image(images):
        image_embeddings = []
        preprocess_images = torch.stack([preprocess(b) for b in images]).to(device)
        with torch.no_grad():
            embeds = clip_model.encode_image(preprocess_images).cpu().numpy()
        image_embeddings.extend(embeds)
        image_embeddings = np.array(image_embeddings)
        return torch.tensor(image_embeddings, device=device)

    batchify_run(process_image, images, res, 64, use_tqdm=use_tqdm)
    return res


def schemafeatures2list(feature):
    # {
    #     "feature_label": "spiky leaves",
    #     "relation_to_object": "attribute_of",
    #     "importance": 0.9
    # }
    # becomes
    # "has spiky leaves"
    # because
    verb = "has" if feature["relation_to_object"] == "attribute_of" else "is a"
    return f"{verb} {feature['feature_label']}"


def schema_to_conceptdict(schema):
    concept_dict = {}
    for k, v in schema.items():
        list(map(schemafeatures2list, v["visual_features"]))

    for feature in schema:
        feature_label = feature["feature_label"]
        concept_dict[feature_label] = feature
    return concept_dict
