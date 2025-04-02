import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .gpt_utils import get_completion

system_prompt_gpt_subselector = """You are a helpful assistant that helps identify visual features that are useful for identifying a '{cls}' from a picture. Please select the top {k} visual features that are useful for identifying a '{cls}' from a picture.
Our algorithm runs for many iterations. In each iteration, it subselects a list of visual features for a class. For each iteration, you will be given the following information:
1) The class name.
2) A set of visual descriptors that needs to be subsampled.
3) The top 3 classes that were misidentified as the class.

Your goal is to output a list of descriptors for '{cls}' that will ensure that a '{cls}' is not misidentified for any other class. Use the feedback from the previous iterations.
Output a list of descriptors for '{cls}' that will help the classifier distinguish the original class from the ambiguous class. Each feature must be a bullet point, one per line. Make sure each bullet point itself isn't ambiguous, and keep each bullet point under 100 words.
Please please please do not output anything else! Here is an example output:

- Descriptor 1
- Descriptor 2
- Descriptor 3
- Descriptor 4
- ...
- Descriptor {k}
"""


def subselect_with_llm_history_message_builder(
    clsA,
    clsA_descriptors,
    k,
    subsampling_history,
    confusion_matrix_history,
    classes,
    distance_type,
):
    classes = classes.tolist() if isinstance(classes, np.ndarray) else classes
    bar = "=" * 20
    per_iteration_format = """iteration {i}:

{clsA} descriptors:
{clsA_descriptors}

Top 3 misidentified classes: {confusion_classes}"""
    per_iteration_format = bar + "\n" + per_iteration_format + "\n"

    clsA_subsample_history = subsampling_history[clsA]
    prompt = ""
    iter_reported = 0
    same_set_later = []

    all_descriptors = [set(d["descriptors"]) for d in clsA_subsample_history]
    for idx, d in enumerate(clsA_subsample_history):
        p = set(d["descriptors"])
        if p in all_descriptors[idx + 1 :]:
            same_set_later.append(1)
        else:
            same_set_later.append(0)

    for idx in range(len(clsA_subsample_history)):
        if same_set_later[idx]:
            continue
        i = clsA_subsample_history[idx]["iteration"]
        descriptors = clsA_subsample_history[idx]["descriptors"]
        confusion_matrix = confusion_matrix_history[i + 1]
        clsA_predictions = confusion_matrix[classes.index(clsA)]
        # import IPython; IPython.embed()
        # if distance_type == "pearson":
        #     error_rate = clsA_predictions[classes.index(clsA)]
        # elif distance_type == 'pca' or distance_type == 'emd':
        #     error_rate = (1 - clsA_predictions[classes.index(clsA)] / clsA_predictions.max())
        # else:
        #     error_rate = (clsA_predictions.sum() - clsA_predictions[classes.index(clsA)]) / clsA_predictions.sum()
        top3_classes = clsA_predictions[clsA_predictions > 0].argsort()[-4:-1]
        top3_classes = [classes[c] for c in top3_classes]
        idx_descriptors = "\n".join(["- " + x for x in descriptors])
        assert len(idx_descriptors) > 0
        prompt += per_iteration_format.format(
            i=iter_reported,  # This is a reinforement learning loop so it's better to keep the iterations continuous as it is more indiciative of the learning process
            clsA=clsA,
            clsA_descriptors=idx_descriptors,
            # error_rate=error_rate,
            confusion_classes=", ".join(top3_classes),
        )
        iter_reported += 1
    prompt += bar + "\n"

    descriptor_list = "\n".join(f"- {desc}" for desc in clsA_descriptors)
    # prompt += f"Please output a list of descriptors for {clsA} that will decrease the proportion of real instances of {clsA} that were mistaken for other classes."
    # prompt += f"Please select the top {k} visual features that are useful for identifying a '{clsA}' from a picture. "
    prompt += (
        f"From the following list of visual features, please select the top {k} visual features that are useful for identifying a '{clsA}' from a picture. Use the feedback from the previous iterations to reduce the proportion of real instances of {clsA} that were mistaken for other classes.\n\n"
        f"Features:\n{descriptor_list}\n"
        "Each feature must be a bullet point, one per line. Make sure each bullet point itself isn't ambiguous, and keep each bullet point under 100 words. For example:\n"
        # "Each feature must be a bullet point, one per line. You are encouraged to repair descripttors by making slight grammatical changes and removing extraneous information (such as references to other classes) if necessary. Make sure each bullet point itself isn't ambiguous, and keep each bullet point under 100 words. For example:\n"
        "- Descriptor 1\n"
        "- Descriptor 2\n"
        "...\n"
        "- Descriptor {k}\n".format(k=k)
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt_gpt_subselector.format(cls=clsA, k=k),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    return messages


def subselect_with_llm_single_message_builder(cls, cls_descriptors, k):
    descriptor_list = "\n".join(f"- {desc}" for desc in cls_descriptors)
    user_content = (
        f"From the following list of visual features, please select the top {k} visual features that are useful for identifying a '{cls}' from a picture.\n\n"
        f"Visual Features:\n{descriptor_list}\n\n"
        "Each feature must be a bullet point, one per line. Make sure each bullet point itself isn't ambiguous, and keep each bullet point under 100 words. For example:\n"
        # "Each feature must be a bullet point, one per line. You are encouraged to repair descripttors by making slight grammatical changes and removing extraneous information (such as references to other classes) if necessary. Make sure each bullet point itself isn't ambiguous, and keep each bullet point under 100 words. For example:\n"
        "- Descriptor 1\n"
        "- Descriptor 2\n"
        "...\n"
        "- Descriptor {k}\n".format(k=k)
    )

    messages = [
        {
            "role": "system",
            "content": "You are an expert in ranking visual features that are useful for identifying a '{cls}' from a picture. Suggest novel, nontrivial, and insightful visual features and always output atleast {k} descriptors!".format(
                cls=cls,
                k=k,
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return messages


def subselect_with_llm_single(clsA, descriptors, k, openai_model, openai_temp):
    completion = get_completion(
        model=openai_model,
        messages=subselect_with_llm_single_message_builder(clsA, descriptors, k),
        temperature=openai_temp,
    )
    content = completion.choices[0].message.content.strip().split("\n")
    content = [line.strip("- ").strip("-").strip() for line in content]
    content = [descriptor for descriptor in content if len(descriptor) > 0]

    if len(content) != k:
        print(f"Warning: {clsA} has {len(content)} descriptors. Requested {k}")
        # import IPython; IPython.embed()
        # with open("gpt_debug_logs/subselect.jsonl", "a") as f:
        #     json.dump(
        #         dict(
        #             cls=clsA,
        #             descriptors=descriptors,
        #             k=k,
        #             output=completion.choices[0].message.content,
        #         ),
        #         f,
        #     )
        #     f.write("\n")
    return content


def subselect_with_llm_history(
    clsA,
    descriptors,
    k,
    subsampling_history,
    confusion_matrix_history,
    classes,
    openai_model,
    openai_temp,
    distance_type,
):
    completion = get_completion(
        model=openai_model,
        messages=subselect_with_llm_history_message_builder(
            clsA,
            descriptors,
            k,
            subsampling_history,
            confusion_matrix_history,
            classes,
            distance_type,
        ),
        temperature=openai_temp,
    )
    content = completion.choices[0].message.content.strip().split("\n")
    content = [line.strip("- ").strip("-").strip() for line in content]
    content = [descriptor for descriptor in content if len(descriptor) > 0]

    if len(content) != k:
        print(f"Warning: {clsA} has {len(content)} descriptors. Requested {k}")
        # import IPython; IPython.embed()

        # with open("gpt_debug_logs/subselect_with_history.jsonl", "a") as f:
        #     json.dump(
        #         dict(
        #             cls=clsA,
        #             descriptors=descriptors,
        #             k=k,
        #             subsampling_history=subsampling_history,
        #             confusion_matrix_history=confusion_matrix_history,
        #             output=completion.choices[0].message.content,
        #         ),
        #         f,
        #     )
        #     f.write("\n")
    return content


def _parallel_get_llm_output(choice, self_ref):
    """
    A helper function to be run in parallel.

    Returns:
        (clsA, clsB, clsA_descriptors, clsB_descriptors, err)
        If there's an exception, returns the exception in 'err'.
    """
    clsA_idx, clsB_idx, cnt = choice
    clsA = self_ref.library.classes[clsA_idx]
    clsB = self_ref.library.classes[clsB_idx]

    try:
        # clsA_descriptors, clsB_descriptors = get_llm_output(
        #     clsA,
        #     clsB,
        #     self_ref.cls2concepts_history,
        #     self_ref.confusion_matrix_history,
        #     np.array(self_ref.classes),
        #     openai_temp=self_ref.openai_temp,
        #     openai_model=self_ref.openai_model,
        #     distance_type=self_ref.distance_type,
        # )

        clsA_descriptors, clsB_descriptors = (
            self_ref.library.get_llm_output_history_conditioned(
                clsA=clsA,
                clsB=clsB,
                distance_type=self_ref.distance_type,
                openai_model=self_ref.openai_model,
                openai_temp=self_ref.openai_temp,
            )
        )
        return (clsA, clsB, clsA_descriptors, clsB_descriptors, None)
    except Exception as e:
        return (clsA, clsB, None, None, e)


def parallel_update(self, choices):
    # 1. Submit get_llm_output calls in parallel
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Kick off parallel jobs; store futures
        futures = {
            executor.submit(_parallel_get_llm_output, choice, self): choice
            for choice in choices
        }

        # 2. Collect results as they come in
        for future in concurrent.futures.as_completed(futures):
            # Retrieve results from the future
            clsA, clsB, clsA_descs, clsB_descs, err = future.result()

            if err is not None:
                print(f"exception for {clsA} and {clsB}")
                print(err)
            else:
                # 3. Update shared data structures
                self.library.update_class(clsA, clsA_descs)
                self.library.update_class(clsB, clsB_descs)

    # 4. Post-process descriptors
    if self.subselect > 0:
        for cls in self.library.classes:
            self.library.subselect_class(cls, self.subselect)

    # for cls, all_descriptors in self.cls2concepts.items():
    #     # De-duplicate if needed
    #     if len(all_descriptors) != len(set(all_descriptors)):
    #         all_descriptors = list(set(all_descriptors))
    #         self.cls2concepts[cls] = all_descriptors

    #     # If there’s a limit on how many we keep, do sub-selection
    #     if len(all_descriptors) > self.subselect and (self.subselect > 0):
    #         if cls in self.subsampling_history:
    #             all_descriptors = subselect_with_llm_history(
    #                 cls,
    #                 all_descriptors,
    #                 k=self.subselect,
    #                 subsampling_history=self.subsampling_history,
    #                 confusion_matrix_history=self.confusion_matrix_history,
    #                 classes=np.array(self.classes),
    #                 openai_temp=self.openai_temp,
    #                 openai_model=self.openai_model,
    #                 distance_type=self.distance_type,
    #             )
    #         else:
    #             all_descriptors = subselect_with_llm_single(
    #                 cls,
    #                 all_descriptors,
    #                 k=self.subselect,
    #                 openai_temp=self.openai_temp,
    #                 openai_model=self.openai_model,
    #             )

    #         self.subsampling_history[cls].append(
    #             dict(descriptors=all_descriptors, iteration=self.iteration)
    #         )
    #         self.cls2concepts[cls] = all_descriptors
