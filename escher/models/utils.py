import glob
import os

from escher.library import HistoryConditionedLibrary, Library
from escher.models.model_lm4cv import LM4CV_KWARG_SET, LM4CVModel
from escher.models.model_zero_shot import ZeroShotModel


def resume_from(iteration, model_name, prompt_type, clip_model_name, dataset_name):
    if iteration == -1:
        # load last iteration
        clip_model_name.replace("/", "-")
        logs = glob.glob(
            f"descriptors/{model_name}/{dataset_name.lower()}/descriptors_*.json"
        )
        if len(logs) > 0:
            log_int = [
                int(os.path.basename(log).split("_")[1].split(".")[0]) for log in logs
            ]
            log_iteration = max(log_int)
            iteration = log_iteration + 1
            print(f"Resuming from iteration {iteration}")
            library = HistoryConditionedLibrary.resume(
                clip_name=model_name,
                dataset_name=dataset_name.lower(),
                iteration=log_iteration,
            )
            # conversational_history = load_obj(f"descriptors/zero_shot_{clip_name}/{dataset_name.lower()}/history/iter{log_iteration}_conversational_history.json")
            # conversational_history = defaultdict(list, {int(k): v for k, v in conversational_history.items()})
            # resolution_history = np.load(f"descriptors/zero_shot_{clip_name}/{dataset_name.lower()}/history/iter{log_iteration}_resolution_history.npy")
            # other_stuff = load_obj(f"descriptors/zero_shot_{clip_name}/{dataset_name.lower()}/history/iter{log_iteration}.json")
            # initial_descriptors = other_stuff['cls2concepts']
            # cls2concepts_history = other_stuff['cls2concepts_history']
            # subsampling_history = other_stuff['subsampling_history']
        else:
            raise ValueError("Cannot automatically find last iteration.")
    else:
        iteration = iteration
        library = (
            HistoryConditionedLibrary.resume(
                clip_name=model_name,
                dataset_name=dataset_name.lower(),
                iteration=iteration,
            )
            if "conversational_history" in prompt_type
            else Library()
        )
    return iteration, library


def get_model(
    initial_descriptors: dict,
    clip_model_name: str,
    scoring_clip_model_name: str,
    use_open_clip: bool,
    openai_model: str,
    openai_temp: float,
    correlation_matrix_threshold: float,
    selection_proportion: float,
    prompt_type: str,
    classwise_topk: int,
    distance_type: str,
    topk: int,
    subselect: int,
    decay_factor: float,
    shots: int,
    salt: str,
    algorithm: str,
    dataset_name: str,
    iteration: int,
    lm4cv_kwarg_set: int,
):
    if algorithm == "zero-shot":
        # hack. I should calculate model name directly instead of this.
        model_name = ZeroShotModel.model_name(
            clip_model_name=clip_model_name,
            scoring_clip_model_name=scoring_clip_model_name,
            open_clip=use_open_clip,
            salt=salt,
        )
        iteration, library = resume_from(
            iteration=iteration,
            model_name=model_name,
            prompt_type=prompt_type,
            clip_model_name=clip_model_name,
            dataset_name=dataset_name,
        )
        model = ZeroShotModel(
            openai_model=openai_model,
            openai_temp=openai_temp,
            correlation_matrix_threshold=correlation_matrix_threshold,
            selection_proportion=selection_proportion,
            initial_descriptors=initial_descriptors,
            prompt_type=prompt_type,
            classwise_topk=classwise_topk,
            distance_type=distance_type,
            topk=topk,
            subselect=subselect,
            library=library,
            # confusion_matrix_history=confusion_matrix_history,
            # subsampling_history=subsampling_history,
            # cls2concepts_history=cls2concepts_history,
            decay_factor=decay_factor,
            # conversational_history=conversational_history,
            # resolution_history=resolution_history,
            salt=salt,
            shots=shots,
        )
    elif algorithm == "lm4cv":
        model_name = LM4CVModel.model_name(
            clip_model_name=clip_model_name,
            open_clip=use_open_clip,
            salt=salt,
            shots=shots,
        )
        iteration, library = resume_from(
            iteration=iteration,
            model_name=model_name,
            prompt_type=prompt_type,
            clip_model_name=clip_model_name,
            dataset_name=dataset_name,
        )
        model = LM4CVModel(
            openai_model=openai_model,
            openai_temp=openai_temp,
            correlation_matrix_threshold=correlation_matrix_threshold,
            selection_proportion=selection_proportion,
            initial_descriptors=initial_descriptors,
            prompt_type=prompt_type,
            classwise_topk=classwise_topk,
            distance_type=distance_type,
            topk=topk,
            subselect=subselect,
            library=library,
            # confusion_matrix_history=confusion_matrix_history,
            # subsampling_history=subsampling_history,
            # cls2concepts_history=cls2concepts_history,
            decay_factor=decay_factor,
            # conversational_history=conversational_history,
            # resolution_history=resolution_history,
            salt=salt,
            shots=shots,
            dataset_name=dataset_name,
            lm4cv_kwargs=LM4CV_KWARG_SET[lm4cv_kwarg_set],
        )
    else:
        raise ValueError("Invalid algorithm")

    return model, library, iteration
