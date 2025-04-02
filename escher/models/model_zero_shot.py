import re
from functools import partial
from typing import Dict, List

import numpy as np
import torch

from escher.cbd_utils import (
    choose_confusion_candidates,
    construct_gpt_queries,
    get_gpt_completions,
    map_cls2descriptions_class_names,
    parse_gpt_output,
    reduce_to_class_scores_by_mean,
)
from escher.cbd_utils.metrics import calculate_class_correlation
from escher.cbd_utils.metrics_history import parallel_update
from escher.library import HistoryConditionedLibrary, Library
from escher.models.model import Model


class ZeroShotModel(Model):
    def __init__(
        self,
        openai_model,
        openai_temp,
        correlation_matrix_threshold: float,
        selection_proportion: float,
        initial_descriptors: Dict[str, List[str]],
        prompt_type: str = "confound",
        classwise_topk: int = 3,
        distance_type: str = "emd",
        subselect: int = -1,
        topk: int = 50,
        # confusion_matrix_history=[],
        # subsampling_history=defaultdict(list),
        # cls2concepts_history=defaultdict(list),
        # conversational_history=defaultdict(list),
        # resolution_history=None,
        library: Library = None,
        iteration=0,
        decay_factor=0,
        salt="",
        shots=-1,
    ):
        self.openai_model = openai_model
        self.openai_temp = openai_temp
        self.correlation_matrix_threshold = correlation_matrix_threshold
        self.selection_proportion = selection_proportion
        self.initial_descriptors = initial_descriptors
        self.prompt_type = prompt_type
        self.classwise_topk = classwise_topk
        self.distance_type = distance_type
        self.subselect = subselect
        self.topk = topk
        self.iteration = iteration
        self.library = library
        # self.confusion_matrix_history = confusion_matrix_history
        # self.subsampling_history = subsampling_history
        # self.cls2concepts_history = cls2concepts_history
        self.use_conversational_history = "conversational_history" in prompt_type
        self.salt = salt
        # self.resolution_history = resolution_history

        if self.use_conversational_history:
            print("Using conversational history")
            # self.conversational_history = conversational_history
            self.prompt_type = re.sub(
                r"_with_conversational_history", "", self.prompt_type
            )
            self.library = HistoryConditionedLibrary.from_library(library)

        self.decay_factor = decay_factor

    def initialize(
        self,
        clip_model_name: str,
        scoring_clip_model_name: str,
        open_clip: bool,
        device: str,
        cls2index: Dict[str, int],
        classes: List[str],
        deduplicate_descriptors: bool = False,
    ):
        super().initialize(
            clip_model_name,
            scoring_clip_model_name,
            open_clip,
            device,
            cls2index,
            classes,
            deduplicate_descriptors=deduplicate_descriptors,
        )
        # initialize the knowledge graph from the initial set of descriptors
        self.initial_descriptors = map_cls2descriptions_class_names(
            self.initial_descriptors, self.classes
        )

        if deduplicate_descriptors:
            self.initial_descriptors = self.deduplicate_descriptors(
                self.initial_descriptors,
                use_score_model=(len(scoring_clip_model_name) > 0),
            )
            dedup_func = partial(
                self.deduplicate_descriptors,
                use_score_model=(len(scoring_clip_model_name) > 0),
            )
        else:
            dedup_func = None

        if not self.library.initialized:
            self.library.initialize(
                self.initial_descriptors, classes=classes, dedup_func=dedup_func
            )
        # self.resolution_history = (
        #     np.zeros((len(self.initial_descriptors), len(self.initial_descriptors)))
        #     if self.resolution_history is None
        #     else self.resolution_history
        # )
        # self.kg = nx.DiGraph()
        # for class_name in self.initial_descriptors:
        #     self.kg.add_node(class_name)
        #     for attribute in self.initial_descriptors[class_name]:
        #         self.kg.add_edge(class_name, attribute)

        # check that the initial set of descriptors is not missing anything
        assert len(cls2index) == len(self.initial_descriptors)
        assert all(self.initial_descriptors.values())

    @classmethod
    def model_name(cls, clip_model_name, scoring_clip_model_name, open_clip, salt):
        model_name = (
            clip_model_name.replace("/", "-")
            + ("_score" if len(scoring_clip_model_name) else "")
            + scoring_clip_model_name.replace("/", "-")
            + ("_open" if open_clip else "")
        )
        name = f"zero_shot_{model_name}"
        if salt:
            name += f"_{salt}"
        return name

    def name(self):
        return self.model_name(
            self.clip_model_name,
            self.scoring_clip_model_name,
            self.open_clip,
            self.salt,
        )

    def get_descriptors(self, should_process_descriptors=True) -> Dict[str, List[str]]:
        return self.library.get_library(
            should_process_descriptors=should_process_descriptors
        )

    def predict(
        self, image_embeddings: torch.Tensor, clip_scores: List[np.ndarray]
    ) -> List[int]:
        class_scores = reduce_to_class_scores_by_mean(
            clip_scores
        )  # (n_images, n_classes)
        return class_scores.argmax(1)  # (n_images)

    def train_descriptors(
        self,
        train_image_embeddings: torch.Tensor,
        train_clip_scores: torch.Tensor,
        masked_labels: List[int],
        use_interpretability_critic: bool = False,
        val_confusion_matrix=None,
    ):
        class_scores = reduce_to_class_scores_by_mean(
            train_clip_scores
        )  # (n_images, n_classes)

        # import IPython; IPython.embed()
        correlation_matrix, potential_confusions = calculate_class_correlation(
            classes=self.classes,
            logits=class_scores,
            report=None,  # In zero shot, we don't have any information.
            topk=self.classwise_topk,
            distance_type=self.distance_type,
        )
        if val_confusion_matrix is not None:
            correlation_matrix = val_confusion_matrix

        # if self.use_conversational_history:
        #     # for each class modified in the previous iteration, update the "feedback"
        #     self.library.populate_conversational_history(
        #         iteration=self.iteration - 1,
        #         correlation_matrix=correlation_matrix,
        #         distance_type=self.distance_type,
        #     )
        #     self.conversational_history = populate_conversational_history(
        #         iteration=it,
        #         conversational_history=self.conversational_history,
        #         correlation_matrix=correlation_matrix,
        #         classes=self.classes,
        #         distance_type=self.distance_type,
        #     )

        descriptors = self.get_descriptors()
        unprocessed_descriptors = self.get_descriptors(should_process_descriptors=False)
        if not self.use_conversational_history:
            self.resolve_confusions_no_history(
                correlation_matrix, descriptors, unprocessed_descriptors
            )
        else:
            self.resolve_confusions_with_history(
                correlation_matrix, descriptors, potential_confusions
            )

        # self.cls2concepts = self.deduplicate_descriptors(self.cls2concepts, use_score_model=(len(self.scoring_clip_model_name) > 0))
        self.library.dedup_across_classes()
        return correlation_matrix

    def resolve_confusions_no_history(
        self, correlation_matrix, descriptors, unprocessed_descriptors
    ):
        potential_confusions = []
        for i, j in zip(
            *np.where(correlation_matrix > self.correlation_matrix_threshold)
        ):
            if correlation_matrix[i, j] > 0 and i != j:
                potential_confusions.append((i, j, correlation_matrix[i, j]))

        choices = choose_confusion_candidates(potential_confusions, self.topk)
        for clsA, clsB, cnt in sorted(choices, key=lambda x: -1 * x[-1])[:10]:
            clsA = self.classes[clsA]
            clsB = self.classes[clsB]
            print(f"{cnt:.2f} : {clsA} confused with {clsB}")

        for i in range(len(choices)):
            curr, candidate = choices[i][:2]
            self.library.resolution_history[curr, candidate] += 1

        gpt_queries, class_indices = construct_gpt_queries(
            choices=choices,
            prompt_type=self.prompt_type,
            classes=self.classes,
            descriptors=descriptors,
            raw_descriptors=unprocessed_descriptors,
            correlation_matrix=correlation_matrix,
            conversational_history=None,
            # conversational_history=self.library.conversational_history
            # if self.use_conversational_history
            # else None,
        )

        # query GPT for suggested new descriptors
        gpt_output = get_gpt_completions(
            gpt_queries, self.openai_model, self.openai_temp, output_is_json=True
        )

        # if self.use_conversational_history:
        #     for i in range(len(gpt_output)):
        #         curr, candidate = choices[i][:2]
        #         self.library.update_conversational_history(
        #             it=self.iteration,
        #             query=gpt_queries[i],
        #             response=gpt_output[i],
        #             cls1=self.classes[curr],
        #             cls2=self.classes[candidate],
        #         )

        parsed_output = parse_gpt_output(self.prompt_type, gpt_output)
        # add the new descriptors to the knowledge graph
        for class_index, suggested_descriptors in zip(class_indices, parsed_output):
            self.library.update_class(self.classes[class_index], suggested_descriptors)

    def resolve_confusions_with_history(
        self, correlation_matrix, descriptors, potential_confusions
    ):
        self.library.update_history(correlation_matrix)

        choices = choose_confusion_candidates(
            potential_confusions=potential_confusions,
            topk=self.topk,
            resolution_history=self.library.resolution_history,
            decay_factor=self.decay_factor,
        )
        parallel_update(self, choices=choices)

        # for clsA_idx, clsB_idx, cnt in choices:
        #     clsA, clsB = self.classes[clsA_idx], self.classes[clsB_idx]
        #     try:
        #         clsA_descriptors, clsB_descriptors = get_llm_output(
        #             clsA,
        #             clsB,
        #             self.cls2concepts_history,
        #             self.confusion_matrix_history,
        #             np.array(self.classes),
        #             openai_temp=self.openai_temp,
        #             openai_model=self.openai_model,
        #             distance_type=self.distance_type,
        #         )
        #         self.cls2concepts[clsA] = list(
        #             set(self.cls2concepts[clsA] + clsA_descriptors)
        #         )
        #         self.cls2concepts[clsB] = list(
        #             set(self.cls2concepts[clsB] + clsB_descriptors)
        #         )
        #     except Exception as e:
        #         print(f"exception for {clsA} and {clsB}")
        #         print(e)

        # for cls, all_descriptors in self.cls2concepts.items():
        #     if len(all_descriptors) != list(set(all_descriptors)):
        #         all_descriptors = list(set(all_descriptors))
        #         self.cls2concepts[cls] = all_descriptors
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
