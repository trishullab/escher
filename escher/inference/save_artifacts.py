import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from escher.utils.dataset_loader import get_dataset as get_image_dataset
from escher.inference.data_utils import save_obj
from escher.inference.model_ops import get_dataset as get_clip_dataset
from escher.inference.model_ops import get_model, get_model_attrs, get_scores, getfinaliter
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="zero_shot_ViT-L-14_scoreViT-L-14_2.1.rebuttal",
        help="Name (or sub-directory) of the target model.",
    )
    parser.add_argument(
        "--clip_model_name",
        default="ViT-L/14",
        help="Which CLIP model variant to use (e.g., 'ViT-L/14').",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["food101", "cub", "nabirds", "flowers"],
        help="Datasets to process.",
    )
    parser.add_argument(
        "--base_dir",
        default="descriptors",
        help="Base directory where descriptor iteration folders are located.",
    )
    parser.add_argument(
        "--artifacts_dir",
        default="artifacts",
        help="Directory into which artifacts are saved.",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Image size for dataset loading."
    )
    parser.add_argument(
        "--val_only",
        action="store_true",
        help="Whether to only load the validation set.",
    )
    parser.add_argument(
        "--max_artifacts",
        type=int,
        default=10,
        help="Max number of new solved artifacts to store per iteration.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=5,
        help="Number of iterations for final multi-iteration analysis.",
    )

    return parser.parse_args()


def find_valid_iteration_range(descriptors_dir):
    """
    Find the valid start and end iteration indices from a descriptor directory.
    Assumes getfinaliter(descriptors_dir) returns a dict { iteration_number: [files,...] }.
    """
    iteration_map = getfinaliter(descriptors_dir)
    if not iteration_map:
        raise ValueError(f"No iteration files found under {descriptors_dir}")

    sorted_iters = sorted(iteration_map.keys())
    start_iteration = sorted_iters[0]
    end_iteration = sorted_iters[-1]

    # Move start_iteration forward until it has 'descriptors.json'
    while "descriptors.json" not in iteration_map[start_iteration]:
        start_iteration += 1
        if start_iteration > end_iteration:
            raise ValueError("No valid descriptors.json found in any iteration range.")

    # Move end_iteration backward until it has 'descriptors.json'
    while "descriptors.json" not in iteration_map[end_iteration]:
        end_iteration -= 1
        if end_iteration < start_iteration:
            raise ValueError("No valid descriptors.json found in any iteration range.")

    return start_iteration, end_iteration


def solved_at_iteration(model, val_clip_dataset, test_clip_dataset):
    """
    Return a boolean array indicating which images in the validation set
    are correctly solved at the given iteration.
    """
    val_labels, _, fin_val_scores, _, _, _ = get_scores(
        model=model,
        val_dataset=val_clip_dataset,
        test_dataset=test_clip_dataset,
        verbose_output=True,
    )
    pred_labels = fin_val_scores.argmax(axis=1)
    return pred_labels == val_labels


def infer_params_from_name(model_name, clip_model_name):
    kwargs = {}
    try:
        if "score" in model_name:
            # zero_shot_ViT-B-16_scoreViT-L-14_2.rebuttal
            algorithm, salt = model_name.split(clip_model_name.replace("/", "-"))
            clip_model_name, scoring_clip_model_name = clip_model_name.split("_score")
            kwargs["scoring_clip_model_name"] = scoring_clip_model_name
        else:
            algorithm, salt = model_name.split(clip_model_name.replace("/", "-"))
    except Exception as e:
        print("Trouble parsing {}. Using default values.".format(model_name))
        print("Error: ", e)
        try:
            algorithm = "zero-shot"
            rem_str = model_name.replace(algorithm, "").strip("_ ")
            if rem_str.startswith(clip_model_name.replace("/", "-")):
                rem_str = rem_str.replace(clip_model_name.replace("/", "-"), "").strip(
                    "_ "
                )

            if rem_str.startswith("score"):
                rem_str = rem_str.split("_")[1]
            salt = rem_str
        except Exception as e2:
            raise e2

    algorithm = algorithm.strip("_ ")
    salt = salt.strip("_ ")
    return algorithm, salt, clip_model_name, kwargs


def analyze_dataset(
    dataset_name,
    model_name,
    clip_model_name,
    base_dir="descriptors",
    artifacts_dir="artifacts",
    image_size=224,
    val_only=False,
    max_artifacts=10,
    N=5,
    device="cpu",
    **kwargs,
):
    """
    A function that encapsulates all the logic needed to:
      1) Find valid start & end iterations,
      2) Identify newly solved samples,
      3) Save artifacts of interest,
      4) Repeat iteration checks for up to a certain window,
      5) Analyze how many images are solved exactly by iteration N.

    A typical Python user might call it directly:

        from analyze_artifacts import analyze_dataset

        analyze_dataset("cars", "my_model", "ViT-L/14")

    without having to run the CLI.
    """
    # 1. Locate iteration range
    descriptors_dir = os.path.join(base_dir, model_name, dataset_name)
    start_iteration, end_iteration = find_valid_iteration_range(descriptors_dir)
    algorithm, salt, clip_model_name, kwargs = infer_params_from_name(
        model_name=model_name, clip_model_name=clip_model_name
    )

    # 2. Load the dataset & classes
    #    - get_image_dataset is presumably a function that returns (train_dataset, val_dataset)
    #    - transform/val_only based on user preference
    _, val_dataset = get_image_dataset(
        clip_model_name,
        dataset_name,
        image_size=image_size,
        transform=False,
        val_only=val_only,
    )
    # Load dataset
    train_clip_dataset, test_clip_dataset, val_clip_dataset = get_clip_dataset(
        dataset_name,
        clip_model_name,
        device=device,
        use_open_clip=kwargs.get("use_open_clip", False),
    )
    classes = train_clip_dataset.classes

    baseline_model, baseline_descriptors = get_model(
        dataset_name=dataset_name,
        device=device,
        log_iteration=start_iteration,
        salt=salt,
        return_descriptors=True,
        clip_model_name=clip_model_name,
        dataset=val_clip_dataset,
        **kwargs,
    )
    model, descriptors = get_model(
        dataset_name=dataset_name,
        device=device,
        log_iteration=end_iteration,
        salt=salt,
        return_descriptors=True,
        clip_model_name=clip_model_name,
        dataset=val_clip_dataset,
        **kwargs,
    )
    #    * get_scores returns:
    #      (val_labels, val_scores, fin_val_scores, test_labels, test_scores, fin_test_scores)
    b_val_lbl, b_val_scr, b_fin_val_scr, _, _, _ = get_scores(
        model=baseline_model,
        val_dataset=val_clip_dataset,
        test_dataset=test_clip_dataset,
        verbose_output=True,
    )
    val_lbl, val_scr, fin_val_scr, _, _, _ = get_scores(
        model=model,
        val_dataset=val_clip_dataset,
        test_dataset=test_clip_dataset,
        verbose_output=True,
    )

    baseline_solved = b_fin_val_scr.argmax(axis=1) == b_val_lbl
    solved = fin_val_scr.argmax(axis=1) == val_lbl
    new_solved = solved & (~baseline_solved)
    prob_diff = (
        fin_val_scr[solved].max(axis=1).values
        - b_fin_val_scr[solved].max(axis=1).values
    )
    assert new_solved.sum().item() > 0, "No new images solved at the final iteration."

    # 4. Gather solved artifacts
    solved_idxs = torch.where(new_solved)[0].tolist()
    solved_artifacts = []
    cnt = 0
    for idx in tqdm(solved_idxs, desc="Gathering solved artifacts"):
        image, label = val_dataset[idx]
        # Retrieve attribute explanations (baseline and ours)
        baseline_attrs = get_model_attrs(
            idx, label, b_fin_val_scr[idx], baseline_descriptors, b_val_scr, classes
        )
        attrs = get_model_attrs(
            idx, label, fin_val_scr[idx], descriptors, val_scr, classes
        )
        d = {
            "idx": idx,
            "image": image,
            "true_label": classes[label],
            "baseline": baseline_attrs,
            "ours": attrs,
            "prob_diff": prob_diff[cnt],
        }
        cnt += 1
        solved_artifacts.append(d)

    # Save these "full" newly solved artifacts
    os.makedirs(
        f"{artifacts_dir}/iter{end_iteration}vsbaseline_{dataset_name}", exist_ok=True
    )
    save_obj(
        solved_artifacts,
        f"{artifacts_dir}/iter{end_iteration}vsbaseline_{dataset_name}/solved.pkl",
    )

    # 5. For each iteration from start_iteration to start_iteration+10 (bounded by end_iteration),
    #    find newly solved images and save some artifacts
    model, descriptors = get_model(
        dataset_name=dataset_name,
        device=device,
        log_iteration=start_iteration,
        salt=salt,
        return_descriptors=True,
        clip_model_name=clip_model_name,
        dataset=val_clip_dataset,
        **kwargs,
    )
    max_iter = min(end_iteration, start_iteration + 10)
    for curr_iter in range(start_iteration, max_iter):
        # Load current iteration models
        baseline_model, baseline_descriptors = model, descriptors
        model, descriptors = get_model(
            dataset_name=dataset_name,
            device=device,
            log_iteration=curr_iter + 1,
            salt=salt,
            return_descriptors=True,
            clip_model_name=clip_model_name,
            dataset=val_clip_dataset,
            **kwargs,
        )

        # Retrieve scores
        (
            val_lbl_curr,
            val_scr_curr,
            fin_val_scr_curr,
            _,
            _,
            _,
        ) = get_scores(
            model=model,
            val_dataset=val_clip_dataset,
            test_dataset=test_clip_dataset,
            verbose_output=True,
        )

        (
            b_val_lbl_curr,
            b_val_scr_curr,
            b_fin_val_scr_curr,
            _,
            _,
            _,
        ) = get_scores(
            model=baseline_model,
            val_dataset=val_clip_dataset,
            test_dataset=test_clip_dataset,
            verbose_output=True,
        )

        baseline_solved = b_fin_val_scr_curr.argmax(axis=1) == b_val_lbl_curr
        solved = fin_val_scr_curr.argmax(axis=1) == val_lbl_curr
        new_solved = solved & (~baseline_solved)
        # for the newly solved images, what is the probability difference b/w baseline and ours?
        prob_diff = (
            fin_val_scr_curr[solved].max(axis=1).values
            - b_fin_val_scr_curr[solved].max(axis=1).values
        )
        iteration_artifacts = []
        solved_idxs = torch.where(new_solved)[0].tolist()
        cnt = 0
        for idx in tqdm(
            solved_idxs, desc=f"Iteration {curr_iter} / Total {len(solved_idxs)}"
        ):
            perc_diff = prob_diff[cnt]
            image, label = val_dataset[idx]
            # Gather attribute data
            baseline_attrs = get_model_attrs(
                idx,
                label,
                b_fin_val_scr_curr[idx],
                baseline_descriptors,
                b_val_scr_curr,
                classes,
            )
            # Provide a baseline_prediction_label if you want it
            attrs = get_model_attrs(
                idx,
                label,
                fin_val_scr_curr[idx],
                descriptors,
                val_scr_curr,
                classes,
                baseline_prediction_label=baseline_attrs["pred_y"],
            )
            d = {
                "idx": idx,
                "image": image,
                "true_label": classes[label],
                "baseline": baseline_attrs,
                "ours": attrs,
                "perc_diff": perc_diff,
            }
            iteration_artifacts.append(d)
            cnt += 1

        os.makedirs(
            f"{artifacts_dir}/solved_at_iter{curr_iter}_{dataset_name}", exist_ok=True
        )
        save_obj(
            iteration_artifacts,
            f"{artifacts_dir}/solved_at_iter{curr_iter}_{dataset_name}/solved.pkl",
        )

    # 6. Check how many images are solved exactly after N iterations
    arr_solved_at_iter = []
    for i in range(start_iteration, start_iteration + N):
        model = get_model(
            dataset_name=dataset_name,
            device=device,
            log_iteration=i,
            salt=salt,
            return_descriptors=False,
            clip_model_name=clip_model_name,
            dataset=val_clip_dataset,
            **kwargs,
        )
        arr_solved_at_iter.append(
            solved_at_iteration(model, val_clip_dataset, test_clip_dataset)
        )

    arr_solved_at_iter = np.stack(arr_solved_at_iter, axis=0)  # shape: (N, num_images)

    # not solved by baseline, not solved up to iteration N-1, solved at iteration N
    solved_by_our_model = (~arr_solved_at_iter[:-1]).all(0) & arr_solved_at_iter[-1]
    num_solved = solved_by_our_model.sum()
    print(
        f"[{dataset_name}] Number of images solved by our model after {N} iterations: {num_solved}"
    )

    # 7. Save some examples of iteration artifacts that get solved exactly at iteration N
    iteration_artifacts_w_baseline = []
    cnt = 0
    for idx, is_solved in enumerate(solved_by_our_model):
        if is_solved:
            image, label = val_dataset[idx]
            d = {
                "idx": idx,
                "image": image,
                "true_label": classes[label],
                "attrs": [],
            }
            prev_label = None
            # Gather info from each iteration up to N
            for i in range(start_iteration, start_iteration + N):
                model_i, desc_i = get_model(
                    dataset_name=dataset_name,
                    device=device,
                    log_iteration=i,
                    salt=salt,
                    return_descriptors=True,
                    clip_model_name=clip_model_name,
                    dataset=val_clip_dataset,
                    **kwargs,
                )
                (
                    val_lbl_i,
                    val_scr_i,
                    fin_val_scr_i,
                    _,
                    _,
                    _,
                ) = get_scores(
                    model=model_i,
                    val_dataset=val_clip_dataset,
                    test_dataset=test_clip_dataset,
                    verbose_output=True,
                )

                if i == (start_iteration + N - 1):
                    # Provide baseline prediction label if needed
                    attrs = get_model_attrs(
                        idx,
                        label,
                        fin_val_scr_i[idx],
                        desc_i,
                        val_scr_i,
                        classes,
                        baseline_prediction_label=prev_label,
                    )
                    attrs["iteration"] = i
                else:
                    attrs = get_model_attrs(
                        idx,
                        label,
                        fin_val_scr_i[idx],
                        desc_i,
                        val_scr_i,
                        classes,
                    )
                    attrs["iteration"] = i
                    prev_label = attrs["pred_y"]
                d["attrs"].append(attrs)

            iteration_artifacts_w_baseline.append(d)
            cnt += 1
            if cnt > max_artifacts:
                break

    os.makedirs(
        f"{artifacts_dir}/iteration_artifacts_w_baseline_{dataset_name}", exist_ok=True
    )
    save_obj(
        iteration_artifacts_w_baseline,
        f"{artifacts_dir}/iteration_artifacts_w_baseline_{dataset_name}/to_{N}.pkl",
    )


def analyze_dataset_wrapper(ds, args):
    print(f"Analyzing dataset: {ds}")
    analyze_dataset(
        dataset_name=ds,
        model_name=args.model_name,
        clip_model_name=args.clip_model_name,
        base_dir=args.base_dir,
        artifacts_dir=args.artifacts_dir,
        image_size=args.image_size,
        val_only=args.val_only,
        max_artifacts=args.max_artifacts,
        N=args.N,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )


def parallel_main(args):
    import multiprocessing as mp

    mp.freeze_support()
    mp.set_start_method("spawn")

    with ProcessPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(analyze_dataset_wrapper, ds, args) for ds in args.datasets
        ]
        for f in futures:
            f.result()


def main(args):
    for ds in args.datasets:
        print(f"Analyzing dataset: {ds}")
        analyze_dataset(
            dataset_name=ds,
            model_name=args.model_name,
            clip_model_name=args.clip_model_name,
            base_dir=args.base_dir,
            artifacts_dir=args.artifacts_dir,
            image_size=args.image_size,
            val_only=args.val_only,
            max_artifacts=args.max_artifacts,
            N=args.N,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )


if __name__ == "__main__":
    args = parse_args()
    if len(args.datasets) == 1:
        main(args)
    else:
        parallel_main(args)

# Sample invocation:
# python inference/save_visualization_artifacts.py --model_name zero_shot_ViT-L-14_a.local_run --artifacts_dir artifacts_local_run --datasets cub nabirds flowers
