import re
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import TensorDataset

from escher import lm4cv
from escher.cbd_utils import (
    choose_confusion_candidates,
    construct_gpt_queries,
    get_gpt_completions,
    map_cls2descriptions_class_names,
    parse_gpt_output,
)
from escher.cbd_utils.metrics import calculate_class_correlation
from escher.cbd_utils.metrics_history import parallel_update
from escher.library import HistoryConditionedLibrary, Library
from escher.models.model import Model


def config_lookup(dataset_name):
    config_loc = "/var/local/atharvas/f/learning_descriptors/LM4CV/configs"
    internal_dataset_name = lm4cv.get_folder_name(dataset_name)
    # yaml file
    config = lm4cv.load_yaml(f"{config_loc}/{internal_dataset_name}.yaml")
    return config


LM4CV_KWARG_SET = [
    dict(),  # no kwargs
    dict(num_attributes=100),
    dict(num_attributes=200),
]


class LM4CVModel(Model):
    def __init__(
        self,
        openai_model,
        openai_temp,
        correlation_matrix_threshold: float,
        selection_proportion: float,
        initial_descriptors: Dict[str, List[str]],
        dataset_name: str,
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
        lm4cv_kwargs={},
    ):
        self.openai_model = openai_model
        self.openai_temp = openai_temp
        self.correlation_matrix_threshold = correlation_matrix_threshold
        self.selection_proportion = selection_proportion
        self.initial_descriptors = initial_descriptors
        self.dataset_name = dataset_name
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
        self.shots = shots
        self.lm4cv_kwargs = lm4cv_kwargs
        # self.resolution_history = resolution_history

        if self.use_conversational_history:
            print("Using conversational history")
            # self.conversational_history = conversational_history
            self.prompt_type = re.sub(
                r"_with_conversational_history", "", self.prompt_type
            )
            self.library = HistoryConditionedLibrary.from_library(library)

        self.decay_factor = decay_factor

    def get_attributes(self):
        return self.library.get_attributes()

    def get_attributes_clip_scores(self, clip_scores, device):
        attr2tensor = {}
        for ci, c in enumerate(self.library.cls2concepts):
            for ai, a in enumerate(self.library.cls2concepts[c]):
                if a not in attr2tensor:
                    attr2tensor[a] = clip_scores[ci][:, ai]

        attributes = torch.stack(list(attr2tensor.values()), dim=1).to(device)
        return attributes

    def update_model(self):
        self.config["attributes"] = self.get_attributes()

        train_dataset = TensorDataset(
            self.train_dataset.images.cpu(), self.train_dataset.labels.cpu()
        )
        val_dataset = TensorDataset(
            self.val_dataset.images.cpu(), self.val_dataset.labels.cpu()
        )

        self.model, selected_attributes, attr_embed = lm4cv.get_model_only(
            self.config, train_dataset, val_dataset
        )
        self.prefix = lm4cv.get_prefix(self.config)

        attr2idx = {attr: idx for idx, attr in enumerate(selected_attributes)}

        cls2concepts = dict()
        for c, dlist in self.library.cls2concepts.items():
            for attr in dlist:
                if attr not in attr2idx:
                    if attr not in self.filtered_descriptors[c]:
                        self.filtered_descriptors[c][attr] = 0
                    self.filtered_descriptors[c][attr] += 1
                else:
                    if c not in cls2concepts:
                        cls2concepts[c] = []
                    cls2concepts[c].append(attr)
        # Don't update cls2concepts
        # self.best_model, _, _, self.attributes, (_, _) = lm4cv.model(config)
        self.model = self.model.to(self.device)
        for c, filt_attrs in self.filtered_descriptors.items():
            for attr in filt_attrs:
                print(f"{c:20}: {self.filtered_descriptors[c][attr]:5} : {attr:20}")

    def initialize(
        self,
        clip_model_name: str,
        open_clip: bool,
        device: int,
        cls2index: Dict[str, int],
        classes: List[str],
        train_dataset=None,
        val_dataset=None,
        deduplicate_descriptors=False,
    ):
        super().initialize(
            clip_model_name=clip_model_name,
            scoring_clip_model_name="",
            open_clip=open_clip,
            device=device,
            cls2index=cls2index,
            classes=classes,
            deduplicate_descriptors=deduplicate_descriptors,
        )
        # initialize the knowledge graph from the initial set of descriptors
        self.initial_descriptors = map_cls2descriptions_class_names(
            self.initial_descriptors, self.classes
        )
        self.initial_descriptors = self.deduplicate_descriptors(
            self.initial_descriptors
        )
        dedup_func = self.deduplicate_descriptors

        self.library.initialize(
            self.initial_descriptors, classes=classes, dedup_func=dedup_func
        )
        # self.cls2concepts = deepcopy(self.initial_descriptors)
        self.device = device
        self.filtered_descriptors = defaultdict(dict)

        config = config_lookup(self.dataset_name)
        config["model_size"] = clip_model_name
        config["num_labels"] = len(self.classes)
        # config["epochs"] = -2000
        config["batch_size"] = 4096
        config["no_metrics"] = True
        config.update(self.lm4cv_kwargs)
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.update_model()

        self.feed_features = config["reinit"] and config["num_attributes"] != "full"

        if self.shots > 0:
            train_labels = train_dataset.labels
            self.cls2idx_train = dict()
            for idx, label in enumerate(train_labels.tolist()):
                label = int(label)
                if label not in self.cls2idx_train:
                    self.cls2idx_train[label] = []

                if len(self.cls2idx_train[label]) >= self.shots:
                    continue
                self.cls2idx_train[label].append(idx)

            assert all(
                [len(v) == self.shots for v in self.cls2idx_train.values()]
            ), "Number of shots must be equal for all classes"

        # check that the initial set of descriptors is not missing anything
        assert len(cls2index) == len(self.initial_descriptors)
        assert all(self.initial_descriptors.values())

    @classmethod
    def model_name(cls, clip_model_name, open_clip, salt, shots=-1):
        model_name = clip_model_name.replace("/", "-") + ("_open" if open_clip else "")
        name = f"lm4cv_{model_name}"
        if shots > 0:
            name += f"_{shots}shot"
        if salt:
            name += f"_{salt}"
        return name

    def name(self):
        return self.model_name(self.clip_model_name, self.open_clip, self.salt)

    def process_descriptions(self, cls2descriptions):
        result = {}
        for clsname, ds in cls2descriptions.items():
            result[clsname] = [self.prefix + d for d in ds]
            if len(result[clsname]) == 0:
                result[clsname] = [clsname]
        return result

    def get_descriptors(self, should_process_descriptors=True) -> Dict[str, List[str]]:
        # In LM4CV, we keep track of descriptors internally, get_descriptors
        # should return just the {cls_name : cls_name} as the descriptors.
        # if should_process_descriptors:
        #     return self.process_descriptions(self.cls2concepts)
        # return self.cls2concepts
        return self.library.get_library(
            should_process_descriptors=should_process_descriptors,
            custom_proc_func=self.process_descriptions,
        )

    def call_model(self, image_embeddings, clip_scores):
        if self.feed_features:
            class_scores = self.model(image_embeddings.to(self.device)).cpu()
        else:
            attributes = self.get_attributes_clip_scores(clip_scores, self.device)
            class_scores = self.model(attributes).cpu()

        return class_scores

    def predict(
        self, image_embeddings: torch.Tensor, clip_scores: List[np.ndarray]
    ) -> List[int]:
        class_scores = self.call_model(image_embeddings, clip_scores)
        return class_scores.argmax(1)  # (n_images)

    def reveal_train_labels(
        self, train_image_embeddings, train_clip_scores, masked_labels
    ):
        if self.shots > 0:
            idxs_to_reveal = sorted(
                [
                    idx
                    for classwise_idxs in self.cls2idx_train.values()
                    for idx in classwise_idxs
                ]
            )
        else:
            idxs_to_reveal = list(range(len(train_image_embeddings)))
        return idxs_to_reveal

    def train_classifier(
        self,
        train_image_embeddings,
        val_image_embeddings,
        masked_labels,
        val_labels,
        test_accuracy,
    ):
        print("Accuracy before training classifier", test_accuracy())

        if self.feed_features:
            train_features = train_image_embeddings
            val_features = val_image_embeddings
        else:
            clip_scores = self.calculate_clip_scores(train_image_embeddings)
            train_features = self.get_attributes_clip_scores(clip_scores, self.device)

            val_clip_scores = self.calculate_clip_scores(val_image_embeddings)
            val_features = self.get_attributes_clip_scores(val_clip_scores, self.device)

        self.model, best_acc, best_metrics = lm4cv.train_model_only(
            cfg=self.config,
            model=self.model,
            train_labels=masked_labels,
            train_features=train_features,
            val_features=val_features,
            val_labels=val_labels,
        )
        print("Accuracy after training classifier", test_accuracy())

    def train_descriptors(
        self,
        train_image_embeddings: torch.Tensor,
        train_clip_scores: torch.Tensor,
        masked_labels: List[int],
        use_interpretability_critic: bool = False,
        val_confusion_matrix=None,
    ):
        # class_scores = reduce_to_class_scores_by_mean(
        #     train_clip_scores
        # )  # (n_images, n_classes)

        class_scores = self.call_model(train_image_embeddings, train_clip_scores)

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
        self.update_model()
        return correlation_matrix

    def resolve_confusions_no_history(
        self, correlation_matrix, descriptors, unprocessed_descriptors
    ):
        # N = len(self.classes)
        # class_indices = []
        # gpt_queries = []
        # # self.topk
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
