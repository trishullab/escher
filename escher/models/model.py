from functools import partial
from typing import Callable, Dict, List

import numpy as np
import torch
from transformers import AutoModelForPreTraining, AutoProcessor, PreTrainedModel

from escher.cbd_utils import (
    calculate_scores,
    calculate_text_embeddings,
    cosine_similarity,
    load_clip_model,
)


class Model:
    def initialize(
        self,
        clip_model_name: str,
        scoring_clip_model_name: str,
        open_clip: bool,
        device: str,
        cls2index: Dict[str, int],
        classes: List[str],
        use_likelihood: bool = False,
        start_iteration: int = 0,
        deduplicate_descriptors: bool = False,
    ):
        """
        Initializes the clip model and initial set of descriptors.
        """
        self.should_deduplicate_descriptors = deduplicate_descriptors
        self.use_likelihood = use_likelihood
        if use_likelihood:
            self.processor = AutoProcessor.from_pretrained(clip_model_name)
            self.model: PreTrainedModel = AutoModelForPreTraining.from_pretrained(
                clip_model_name
            )
            self.prompt = "This is a photo of a {}."
            self.PAD_TOKEN = self.processor.tokenizer.pad_token_id
            self.EOS_TOKEN = self.processor.tokenizer.eos_token_id
        else:
            self.clip_model, self.tokenizer = load_clip_model(
                clip_model_name, open_clip, device
            )
        if len(scoring_clip_model_name):
            if use_likelihood:
                self.scoring_processor = AutoProcessor.from_pretrained(
                    scoring_clip_model_name
                )
                self.scoring_model: PreTrainedModel = (
                    AutoModelForPreTraining.from_pretrained(scoring_clip_model_name)
                )
            else:
                self.scoring_clip_model, self.scoring_tokenizer = load_clip_model(
                    scoring_clip_model_name, open_clip, device
                )
        self.device = device
        self.cls2index = cls2index
        self.classes = classes
        self.clip_model_name = clip_model_name
        self.scoring_clip_model_name = scoring_clip_model_name
        self.open_clip = open_clip

        assert all(self.classes)

    def name(self) -> str:
        """returns the name of the model. used for logging."""
        raise NotImplementedError

    def get_descriptors(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary from class name to descriptors for that class.
        """
        raise NotImplementedError

    def calculate_clip_scores(
        self, image_embeddings: torch.Tensor, use_score_model: bool = False
    ) -> List[np.ndarray]:
        """
        Returns the clip scores
        """
        cls2descriptions = self.get_descriptors()
        if self.scoring_clip_model_name and use_score_model:
            text_embeddings = calculate_text_embeddings(
                cls2descriptions,
                self.cls2index,
                self.scoring_clip_model,
                self.scoring_tokenizer,
                self.device,
            )
            assert (
                text_embeddings[0].shape[-1] == image_embeddings.shape[-1]
            ), f"Text and image embeddings have different dimensions: {text_embeddings.shape[-1]} != {image_embeddings.shape[-1]}"
        else:
            text_embeddings = calculate_text_embeddings(
                cls2descriptions,
                self.cls2index,
                self.clip_model,
                self.tokenizer,
                self.device,
            )
        if "siglip" in self.clip_model_name.lower():
            model_scaling = (
                self.clip_model.logit_scale.detach().cpu(),
                self.clip_model.logit_bias.detach().cpu(),
            )
        else:
            model_scaling = None
        return calculate_scores(image_embeddings, text_embeddings, model_scaling)

    @staticmethod
    def prepare_ids_for_inference(self, tokens):
        """
        [[a, b, c], [d], [e, f]] -> [[a, d, e], [b, 0, f], [c, 0, 1]]
        # 1 is the EOS token
        # 0 is the padding token
        """
        max_len = max(len(t) for t in tokens) + 1
        seq = [t + [self.PAD_TOKEN] * (max_len - len(t)) for t in tokens]
        # replace the first padding token with EOS
        for i in range(len(seq)):
            seq[i][seq[i].index(self.PAD_TOKEN)] = self.EOS_TOKEN
        seq_transposed = list(map(torch.tensor, zip(*seq)))
        return seq_transposed

    def guided_sampler(self, next_token_scores, iterator):
        ids_to_select = next(iterator)
        ids_to_select = torch.tensor(ids_to_select, device=next_token_scores.device)
        return ids_to_select

    def get_generation_scores(self, generation, tokens_to_generate):
        transition_scores = self.model.compute_transition_scores(
            generation.sequences, generation.scores, normalize_logits=True
        )

        seq_len = [
            len(sent) for sent in self.prepare_ids_for_inference(tokens_to_generate)
        ]
        description_scores = []
        for i in range(transition_scores.shape[0]):
            tokens = generation.sequences[i, -1 * seq_len[i] :]
            total_score, n_tokens = 0.0, 0
            for tok, score in zip(tokens, transition_scores[i]):
                if tok not in self.processor.tokenizer.all_special_ids:
                    total_score += score.item()
                    n_tokens += 1
            total_score /= n_tokens
            description_scores.append(total_score)
        return description_scores

    def calculate_likelihood_scores(self, images: list):
        """
        Returns the likelihood scores
        """
        assert self.use_likelihood, "This model does not support likelihood scores"
        all_descriptors = list(set(self.get_descriptors().values()))
        txt_inputs = [self.prompt.format(desc) for desc in all_descriptors] * len(
            images
        )
        img_inputs = [[image] * len(all_descriptors) for image in images]
        img_inputs = [item for sublist in img_inputs for item in sublist]
        model_inputs = self.processor(
            dict(text=txt_inputs, image=img_inputs, return_tensors="pt", padding=True)
        )

        txt_to_generate = [desc for desc in all_descriptors for _ in range(len(images))]
        tokens_to_generate = self.processor.tokenizer(txt_to_generate, padding=False)[
            "input_ids"
        ]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=False,
                # next_token_finder=None,
                next_token_finder=partial(
                    self.guided_sampler,
                    iterator=iter(self.prepare_ids_for_inference(tokens_to_generate)),
                ),
                return_dict_in_generate=True,
                output_scores=True,
            )
            scores = self.get_generation_scores(generation, tokens_to_generate)  # N * D

        scores = torch.tensor(scores).reshape(len(all_descriptors), len(images)).T
        return scores

    def deduplicate_descriptors(
        self,
        cls2descriptions: List[str],
        use_score_model: bool = False,
        similarity_threshold: float = 0.9,
    ) -> List[str]:
        if not self.should_deduplicate_descriptors:
            return cls2descriptions
        if self.scoring_clip_model_name and use_score_model:
            text_embeddings = calculate_text_embeddings(
                cls2descriptions,
                self.cls2index,
                self.scoring_clip_model,
                self.scoring_tokenizer,
                self.device,
            )
        else:
            text_embeddings = calculate_text_embeddings(
                cls2descriptions,
                self.cls2index,
                self.clip_model,
                self.tokenizer,
                self.device,
            )

        descriptor_to_embedding = {}
        for cii, (class_name, descriptors) in enumerate(cls2descriptions.items()):
            for ci, desc in enumerate(descriptors):
                # Make sure we only store each descriptor once, and skip empty embeddings
                if (
                    desc not in descriptor_to_embedding
                    and len(text_embeddings[cii][ci]) > 0
                ):
                    descriptor_to_embedding[desc] = text_embeddings[cii][ci]

        if not descriptor_to_embedding:
            # If for some reason no embeddings were collected, return as-is
            return cls2descriptions

        all_descriptors = list(descriptor_to_embedding.keys())
        stacked_embeddings = torch.stack(list(descriptor_to_embedding.values())).numpy()

        all_cosine_sim = cosine_similarity(stacked_embeddings, stacked_embeddings)

        # Deduplicate
        visited = set()
        descriptor_representative = {}  # Map each descriptor -> chosen representative

        for i in range(len(all_descriptors)):
            descriptor_representative[all_descriptors[i]] = all_descriptors[
                i
            ]  # default to itself

        for i in range(len(all_descriptors)):
            if i not in visited:
                cluster = []
                stack = [i]
                visited.add(i)

                # Perform DFS to find all descriptors connected to i above threshold
                while stack:
                    current = stack.pop()
                    cluster.append(current)
                    # Check potential neighbors
                    for j in range(len(all_descriptors)):
                        if (
                            j not in visited
                            and all_cosine_sim[current][j] >= similarity_threshold
                        ):
                            visited.add(j)
                            stack.append(j)

                # Choose the cluster[0] descriptor as the canonical representative
                representative_idx = cluster[0]
                representative_desc = all_descriptors[representative_idx]
                for idx in cluster:
                    dup_desc = all_descriptors[idx]
                    descriptor_representative[dup_desc] = representative_desc

        new_cls2descriptions = {}
        for class_name, descriptors in cls2descriptions.items():
            deduped = []
            used = set()
            for desc in descriptors:
                rep = descriptor_representative[desc]
                if rep not in used:
                    deduped.append(rep)
                    used.add(rep)
            new_cls2descriptions[class_name] = deduped

        return new_cls2descriptions

    def predict(
        self, image_embeddings: torch.Tensor, clip_scores: List[np.ndarray]
    ) -> List[int]:
        """
        Returns class predictions for each image.
        Image embeddings are passed as well if model wants to do something more complicated.

        [image_embeddings]: tensor of shape (n_images, model_depth).
        [clip_scores]: scores returned by calculate_clip_scores(image_embeddings)
        """
        pass

    def reveal_train_labels(
        self,
        train_image_embeddings: torch.Tensor,
        train_clip_scores: List[np.ndarray],
        masked_labels: List[int],
    ) -> List[int]:
        """
        Returns a list of indices on [train_image_embeddings]. Labels for the images at these indices will be written into masked_labels
        and provided to train_descriptions() and train_classifier()

        [train_image_embeddings]: image embeddings of the train set. shape (n_images, model_depth)
        [train_clip_scores]: scores returned by calculate_clip_scores(train_images)
        [masked_labels]: labels of the train_images. Label is -1 if not revealed.
        """
        return []

    def train_descriptors(
        self,
        train_image_embeddings: torch.Tensor,
        train_clip_scores: List[np.ndarray],
        masked_labels: List[int],
        use_interpretability_critic: bool,
    ):
        """
        Trains the descriptors.
        [train_image_embeddings]: image embeddings of the train set. shape (n_images, model_depth)
        [train_clip_scores]: scores returned by calculate_clip_scores(train_images)
        [masked_labels]: labels of the train_images. Label is -1 if not revealed.
        """
        pass

    def train_classifier(
        self,
        train_image_embeddings: torch.Tensor,
        val_image_embeddings: torch.Tensor,
        masked_labels: List[int],
        val_labels: List[int],
        test_accuracy: Callable[[], float],
    ):
        """
        Trains the classifier, if any. This function will be called exactly once per iteration.
        [train_image_embeddings]: image embeddings of the train set
        [masked_labels]: labels of the train_images. Label is -1 if not revealed.
        [test_accuracy]: when invoked, returns the accuracy over test_images using predictions from model.predict().
        """
        pass
