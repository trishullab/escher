import json
import os

import numpy as np
import torch

from escher.cbd_utils import metrics
from escher.models.utils import ZeroShotModel
from escher.utils.dataset_loader import ProcessedDataset, get_processed_dataset

from .models.model import Model


def get_test_accuracy_func(model: Model, test_dataset: ProcessedDataset):
    def f():
        test_clip_scores = model.calculate_clip_scores(test_dataset.images)
        predictions = torch.tensor(model.predict(test_dataset.images, test_clip_scores))
        return (predictions == test_dataset.labels).to(torch.float32).mean()

    return f


def run_experiment(
    model: Model,
    dataset_name: str,
    num_iters: int,
    clip_model_name: str,
    scoring_clip_model_name: str,
    device: int,
    use_open_clip: bool,
    image_size: int = 224,
    use_interpretability_critic: bool = False,
    start_iteration: int = 0,
    perc_labels=0,
    ablate_vlm=False,
    deduplicate_descriptors=False,
):
    # create the dataset
    train_dataset, test_dataset, val_dataset = get_processed_dataset(
        clip_model_name,
        dataset_name,
        device=device,
        use_open_clip=use_open_clip,
        image_size=image_size,
    )
    if len(scoring_clip_model_name):
        score_train_dataset, score_test_dataset, score_val_dataset = (
            get_processed_dataset(
                scoring_clip_model_name,
                dataset_name,
                device=device,
                use_open_clip=use_open_clip,
                image_size=image_size,
            )
        )
    if isinstance(model, ZeroShotModel):
        model.initialize(
            clip_model_name,
            scoring_clip_model_name,
            use_open_clip,
            f"cuda:{device}",
            train_dataset.cls2index,
            train_dataset.classes,
            deduplicate_descriptors=deduplicate_descriptors,
        )
    else:
        assert (
            len(scoring_clip_model_name) == 0
        ), "Scoring model only supported for ZeroShotModel"
        model.initialize(
            clip_model_name,
            use_open_clip,
            f"cuda:{device}",
            train_dataset.cls2index,
            train_dataset.classes,
            train_dataset,
            val_dataset,
            deduplicate_descriptors=deduplicate_descriptors,
        )
    masked_train_labels = torch.tensor([-1 for _ in range(len(train_dataset.labels))])

    logdir = os.path.join("logs", model.name())
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # create the log file
    # if logfile already exists, add _1, _2, etc. to the filename
    logfile = os.path.join(logdir, f"{dataset_name}.log")
    if os.path.exists(logfile):
        i = 1
        while os.path.exists(os.path.join(logdir, f"{dataset_name}_{i}.log")):
            i += 1
        logfile = os.path.join(logdir, f"{dataset_name}_{i}.log")

    mode = "w" if start_iteration <= 0 else "a"
    with open(logfile, mode) as log_file:
        for iteration in range(start_iteration, num_iters):
            model.iteration = iteration
            # first, calculate and log the validation accuracy
            if len(scoring_clip_model_name):
                val_clip_scores = model.calculate_clip_scores(
                    score_val_dataset.images, use_score_model=True
                )
                test_clip_scores = model.calculate_clip_scores(
                    score_test_dataset.images, use_score_model=True
                )
            else:
                val_clip_scores = model.calculate_clip_scores(val_dataset.images)
                test_clip_scores = model.calculate_clip_scores(test_dataset.images)
            val_predictions = model.predict(val_dataset.images, val_clip_scores)
            test_predictions = model.predict(test_dataset.images, test_clip_scores)
            val_acc = (
                (torch.tensor(val_predictions) == val_dataset.labels)
                .to(torch.float32)
                .mean()
            )
            test_acc = (
                (torch.tensor(test_predictions) == test_dataset.labels)
                .to(torch.float32)
                .mean()
            )

            valstr = f"iteration {iteration} validation accuracy: {val_acc}\n"
            valstr += f"iteration {iteration} test accuracy: {test_acc}\n"

            if perc_labels > 0:
                val_confusion_matrix = metrics.get_true_confusion_matrix(
                    predictions=val_predictions,
                    classes=val_dataset.classes,
                    labels=val_dataset.labels,
                    perc_labels=perc_labels,
                )
            else:
                val_confusion_matrix = None

            if ablate_vlm:
                val_confusion_matrix = np.random.rand(
                    len(val_dataset.classes), len(val_dataset.classes)
                )

            if iteration >= 0:
                log_file.write(valstr)
                log_file.flush()
                print(valstr)

                # next, log the descriptors
                descriptions_dir = os.path.join(
                    "descriptors", model.name(), dataset_name
                )
                if not os.path.exists(descriptions_dir):
                    os.makedirs(descriptions_dir)
                with open(
                    os.path.join(descriptions_dir, f"descriptors_{iteration}.json"), "w"
                ) as descriptors_file:
                    json.dump(model.get_descriptors(), descriptors_file)

                # also log the history
                history_obj = model.library.create_history_obj()

            # next, do a train loop
            train_clip_scores = model.calculate_clip_scores(train_dataset.images)
            labels_to_reveal = model.reveal_train_labels(
                train_dataset.images, train_clip_scores, masked_train_labels
            )
            for label_index in labels_to_reveal:
                masked_train_labels[label_index] = train_dataset.labels[label_index]

            train_labels_revealed = (masked_train_labels >= 0).sum()
            if train_labels_revealed.sum() > 0:
                log_file.write(
                    f"iteration {iteration} number of train labels revealed: {train_labels_revealed}\n"
                )
                log_file.flush()

            # Only use the interpretability critic every 5 iterations to speed up training.
            if iteration >= 0:
                correlation_matrix = model.train_descriptors(
                    train_dataset.images,
                    train_clip_scores,
                    masked_train_labels,
                    use_interpretability_critic=(
                        use_interpretability_critic and (iteration % 5 == 0)
                    ),
                    val_confusion_matrix=val_confusion_matrix,
                )

                os.makedirs(os.path.join(descriptions_dir, "history"), exist_ok=True)
                with open(
                    os.path.join(
                        descriptions_dir,
                        "history",
                        f"iter{iteration}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(
                        {
                            **history_obj,
                            **{"correlation_matrix": correlation_matrix.tolist()},
                        },
                        f,
                    )

                if model.use_conversational_history:
                    model.library.dump_library(
                        history_obj=history_obj,
                        correlation_matrix=correlation_matrix,
                        descriptions_dir=descriptions_dir,
                        iteration=iteration,
                    )
                    # # dump model.conversational_history as json
                    # with open(
                    #     os.path.join(
                    #         descriptions_dir,
                    #         "history",
                    #         f"iter{iteration}_conversational_history.json",
                    #     ),
                    #     "w",
                    # ) as f:
                    #     json.dump(model.conversational_history, f)

                    # # dump resolution history as numpy array
                    # with open(
                    #     os.path.join(
                    #         descriptions_dir,
                    #         "history",
                    #         f"iter{iteration}_resolution_history.npy",
                    #     ),
                    #     "wb",
                    # ) as f:
                    #     np.save(f, model.resolution_history)

            model.train_classifier(
                train_dataset.images,
                val_dataset.images,
                masked_train_labels,
                val_dataset.labels,
                get_test_accuracy_func(model, test_dataset),
            )
