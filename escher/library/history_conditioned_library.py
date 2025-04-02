import json
import os
from collections import defaultdict
from copy import deepcopy
from typing import Callable

import numpy as np

from escher.cbd_utils import get_completion, load_obj, message_builder
from escher.cbd_utils.metrics_history import (
    subselect_with_llm_history_message_builder,
    subselect_with_llm_single_message_builder,
)

from .base_library import Library


class HistoryConditionedLibrary(Library):
    def __init__(self):
        super().__init__()
        self.conversational_history = defaultdict(list)
        self.resolution_history = None
        self.confusion_matrix_history = []
        self.subsampling_history = defaultdict(list)
        self.cls2concepts_history = defaultdict(list)
        self.completions = defaultdict(list)

    def initialize(self, initial_descriptors, classes, dedup_func: Callable = None):
        super().initialize(initial_descriptors, classes, dedup_func)
        n_classes = len(self.cls2concepts)
        self.resolution_history = (
            np.zeros((n_classes, n_classes))
            if self.resolution_history is None
            else self.resolution_history
        )

    def update_conversational_history(self, it, query, response, cls1, cls2):
        """
        archived code.
        conversational_history and cls2concept_history are two different
        implementations of the same idea (history conditioning). I'm preserving
        the code for both but, in practice, self.cls2concept_history is what
        should be used.

        The implementation of conversational_history overlaps with the resolve_confusions_no_history
        branch in the model_{type}.py class.
        """
        self.conversational_history[it].append(
            dict(
                query=query,
                response=response,
                classes=(cls1, cls2),
                feedback=None,
            )
        )

    def subsample_with_history(self, cls, all_descriptors, **kwargs):
        """
        archived code.
        """

        if cls in self.subsampling_history:
            messages = subselect_with_llm_history_message_builder(
                clsA=cls,
                clsA_descriptors=all_descriptors,
                k=kwargs["k"],
                subsampling_history=kwargs["subsampling_history"],
                confusion_matrix_history=kwargs["confusion_matrix_history"],
                classes=kwargs["classes"],
                distance_type=kwargs["distance_type"],
            )
        else:
            messages = subselect_with_llm_single_message_builder(
                cls=cls,
                cls_descriptors=all_descriptors,
                k=kwargs["k"],
            )

        completion = get_completion(
            model=kwargs["openai_model"],
            messages=messages,
            temperature=kwargs["openai_temp"],
        )
        content = completion.choices[0].message.content.strip().split("\n")
        content = [line.strip("- ").strip("-").strip() for line in content]
        content = [descriptor for descriptor in content if len(descriptor) > 0]

        if len(content) != kwargs["k"]:
            print(
                f"Warning: {cls} has {len(content)} descriptors. Requested {kwargs['k']} descriptors."
            )

        return content

    def subsample_class(self, cls, n_subselect, **kwargs):
        """
        Generally, subsampling was kinda ineffective for most of the experiments
        we tried. Our final implementation of the library doesn't use this method.
        """

        all_descriptors = self.cls2concepts[cls]

        if len(all_descriptors) > n_subselect:
            subsampled_descriptors = self.subsample_with_history(
                cls,
                all_descriptors,
                k=n_subselect,
                subsampling_history=self.subsampling_history,
                confusion_matrix_history=self.confusion_matrix_history,
                classes=np.array(self.classes),
                openai_temp=kwargs["openai_temp"],
                openai_model=kwargs["openai_model"],
                distance_type=kwargs["distance_type"],
            )
            self.subsampling_history[cls].append(
                dict(descriptors=subsampled_descriptors, iteration=kwargs["iteration"])
            )
            self.update_class(cls, subsampled_descriptors)

    def populate_conversational_history(
        self, iteration, correlation_matrix, distance_type
    ):
        """
        archived code.
        This updated the "feedback" from the previous iteration that we
        initially set to None.
        """
        for hi, history in enumerate(self.conversational_history[iteration]):
            clsA, clsB = history["classes"]
            clsA_idx = self.classes.index(clsA)
            clsA_predictions = correlation_matrix[clsA_idx]
            if distance_type == "pearson":
                error_rate = clsA_predictions[self.classes.index(clsB)]
            elif distance_type == "pca" or distance_type == "emd":
                error_rate = (
                    1
                    - clsA_predictions[self.classes.index(clsB)]
                    / clsA_predictions.max()
                )
            else:
                error_rate = (
                    clsA_predictions[self.classes.index(clsB)] / clsA_predictions.sum()
                )
            self.conversational_history[iteration][hi]["feedback"] = error_rate

    def update_history(self, correlation_matrix):
        """
        active code.
        """
        for cls_name in self.cls2concepts:
            self.cls2concepts_history[cls_name].append(
                deepcopy(self.cls2concepts[cls_name])
            )
        self.confusion_matrix_history.append(correlation_matrix.copy())

    def get_llm_output_history_conditioned(
        self, clsA, clsB, distance_type, openai_model, openai_temp
    ):
        """
        active code.
        """
        messages = self.construct_history_conditioned_prompt(clsA, clsB, distance_type)
        completion = get_completion(
            model=openai_model,
            messages=messages,
            temperature=openai_temp,
        )
        # log the prompt
        print(
            "prompt-tokens/generated-tokens [{}|{}]".format(
                completion.usage.prompt_tokens, completion.usage.completion_tokens
            )
        )
        # log the completion per iteration
        iteration = len(self.cls2concepts_history) - 1
        messages.append(
            {
                "role": completion.choices[0].message.role,
                "content": completion.choices[0].message.content,
            }
        )
        self.completions[iteration].append(messages)

        content = completion.choices[0].message.content
        # This will error out if the output is not in the expected format.
        content = content.strip().split("\n")
        content = [line.strip("- ").strip("-").strip() for line in content]
        if content.count("") > 1:
            all_empty_idxs = [i for i, x in enumerate(content) if x == ""]
            second_last_empty = all_empty_idxs[-2]
            content = content[second_last_empty + 1 :]
        elif content.count("") < 1:
            raise ValueError(
                f"Expected two empty lines in the output. Got {content.count('')} instead."
            )
        idx = content.index("")
        return content[1:idx], content[idx + 2 :]

    def construct_history_conditioned_prompt(self, clsA, clsB, distance_type):
        """
        active code.
        """
        return message_builder(
            clsA=clsA,
            clsB=clsB,
            cls2concepts_history=self.cls2concepts_history,
            confusion_matrix_history=self.confusion_matrix_history,
            classes=self.classes,
            distance_type=distance_type,
        )

    @classmethod
    def resume(cls, clip_name, dataset_name, iteration):
        lib = cls()
        if iteration == 0:
            return lib
        try:
            conversational_history = load_obj(
                f"descriptors/{clip_name}/{dataset_name}/history/iter{iteration}_conversational_history.json"
            )
            lib.conversational_history = conversational_history

            resolution_history = np.load(
                f"descriptors/{clip_name}/{dataset_name}/history/iter{iteration}_resolution_history.npy"
            )
            lib.resolution_history = resolution_history
            lib.completions = load_obj(
                f"descriptors/{clip_name}/{dataset_name}/history/iter{iteration}_completions.json"
            )
        except FileNotFoundError:
            print(
                "WARNING: No conversational history found. Resuming without conversational history."
            )
            lib.conversational_history = defaultdict(list)
            lib.resolution_history = None

        other_stuff = load_obj(
            f"descriptors/{clip_name}/{dataset_name}/history/iter{iteration}.json"
        )
        lib.cls2concepts = other_stuff.get("cls2concepts", lib.cls2concepts)
        lib.cls2concepts_history = other_stuff.get("cls2concepts_history", lib.cls2concepts)
        lib.subsampling_history = other_stuff.get("subsampling_history", lib.subsampling_history)
        lib.initialized = True
        return lib

    @classmethod
    def from_library(cls, library):
        if isinstance(library, cls):
            return library
        elif isinstance(library, Library):
            c = cls()
            c.cls2concepts = deepcopy(library.cls2concepts)
            c.classes = library.classes
            return c
        else:
            raise ValueError("Invalid library type")

    def create_history_obj(self):
        return dict(
            cls2concepts=self.get_library(should_process_descriptors=False),
            cls2concepts_history=self.cls2concepts_history,
            subsampling_history=self.subsampling_history,
        )

    def dump_library(
        self, history_obj, correlation_matrix, descriptions_dir, iteration
    ):
        with open(
            os.path.join(
                descriptions_dir,
                "history",
                f"iter{iteration}.json",
            ),
            "w",
        ) as f:
            json.dump(
                {**history_obj, **{"correlation_matrix": correlation_matrix.tolist()}},
                f,
            )

        with open(
            os.path.join(
                descriptions_dir,
                "history",
                f"iter{iteration}_conversational_history.json",
            ),
            "w",
        ) as f:
            json.dump(self.conversational_history, f)

        # dump resolution history as numpy array
        with open(
            os.path.join(
                descriptions_dir,
                "history",
                f"iter{iteration}_resolution_history.npy",
            ),
            "wb",
        ) as f:
            np.save(f, self.resolution_history)

        # dump completions
        with open(
            os.path.join(
                descriptions_dir,
                "history",
                f"iter{iteration}_completions.json",
            ),
            "w",
        ) as f:
            json.dump(self.completions, f)
