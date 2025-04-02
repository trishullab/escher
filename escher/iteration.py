import argparse

from escher.cbd_utils import load_obj
from escher.models.utils import get_model
from escher.run_experiment import run_experiment


def default_args() -> dict:
    """
    These are just random numbers I came up with.
    """
    return dict(
        openai_model="gpt-3.5-turbo",
        openai_temp=0.7,
        correlation_matrix_threshold=0.7,
        selection_proportion=0.05,
        algorithm="zero-shot",
        clip_model_name="ViT-L/14",
        scoring_clip_model_name="",
        image_size=224,
        use_open_clip=False,
        device=0,
        num_iters=60,
        prompt_type="confound_w_descriptors_with_conversational_history",
        use_interpretability_critic=False,
        classwise_topk=5,
        distance_type="pearson",
        topk=50,
        subselect=-1,
        decay_factor=0,
        start_iteration=0,
        salt="",
        perc_labels=0,
        perc_initial_descriptors=1.0,
        shots=-1,
        lm4cv_kwarg_set=0,
        ablate_vlm=False,
        deduplicate_descriptors=False,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        description="Configuration for the image descriptor model"
    )

    _default_args = default_args()

    # Dataset and model parameters
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument(
        "--openai_model",
        type=str,
        default=_default_args["openai_model"],
        help="OpenAI model name",
    )
    parser.add_argument(
        "--openai_temp",
        type=float,
        default=_default_args["openai_temp"],
        help="Temperature for OpenAI model",
    )
    parser.add_argument(
        "--correlation_matrix_threshold",
        type=float,
        default=_default_args["correlation_matrix_threshold"],
        help="Threshold for correlation matrix",
    )
    parser.add_argument(
        "--selection_proportion",
        type=float,
        default=_default_args["selection_proportion"],
        help="Proportion of selection",
    )

    # Type of algorithm (zero-shot, few-shot, finetune)
    parser.add_argument(
        "--algorithm",
        type=str,
        default=_default_args["algorithm"],
        choices=["zero-shot", "lm4cv"],
        help="Type of algorithm",
    )

    # CLIP model configuration
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default=_default_args["clip_model_name"],
        choices=[
            "ViT-B/32",
            "ViT-B/16",
            "ViT-L/14",
            "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
            "google/siglip-so400m-patch14-384",
        ],
        help="CLIP model name for critic. If no scoring model provided, also used for scoring.",
    )
    parser.add_argument(
        "--scoring_clip_model_name",
        type=str,
        default=_default_args["scoring_clip_model_name"],
        help="Optional CLIP model name for scoring.",
        choices=[
            "ViT-B/32",
            "ViT-B/16",
            "ViT-L/14",
            "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
            "google/siglip-so400m-patch14-384",
            "",
        ],
    )

    # Image size
    parser.add_argument(
        "--image_size",
        type=int,
        default=_default_args["image_size"],
        help="Size of input images",
    )

    # OpenCLIP related argument
    parser.add_argument(
        "--use_open_clip",
        type=bool,
        default=_default_args["use_open_clip"],
        help="Flag to use OpenCLIP",
    )

    # Device setup
    parser.add_argument(
        "--device",
        type=int,
        default=_default_args["device"],
        help="Device to use (e.g., GPU number)",
    )

    # Iterations and other parameters
    parser.add_argument(
        "--num_iters",
        type=int,
        default=_default_args["num_iters"],
        help="Number of iterations",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default=_default_args["prompt_type"],
        choices=[
            "confound_with_history",
            "confound_w_descriptors_with_conversational_history",
            "confound_w_descriptors",
            "confound",
            "programmatic",
        ],
        help="Type of prompt",
    )
    parser.add_argument(
        "--use_interpretability_critic",
        type=bool,
        default=_default_args["use_interpretability_critic"],
        help="Flag to use interpretability critic",
    )

    # Class-wise parameters
    parser.add_argument(
        "--classwise_topk",
        type=int,
        default=_default_args["classwise_topk"],
        help="Top k classes for the confusion matrix",
    )
    parser.add_argument(
        "--distance_type",
        type=str,
        default=_default_args["distance_type"],
        help="Distance type for confusion matrix",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=_default_args["topk"],
        help="Top k to consider in the analysis",
    )
    parser.add_argument(
        "--subselect",
        type=int,
        default=_default_args["subselect"],
        help="Subselection index, -1 means no subselect",
    )
    parser.add_argument(
        "--decay_factor",
        type=float,
        default=_default_args["decay_factor"],
        help="Decay factor for penalizing repeated programs",
    )
    parser.add_argument(
        "--start_iteration",
        type=int,
        default=_default_args["start_iteration"],
        help="What iteration to start experiments, -1 means look for last run.",
    )
    parser.add_argument("--salt", type=str, default="", help="Salt for the experiment")
    parser.add_argument(
        "--perc_labels",
        type=float,
        default=_default_args["perc_labels"],
        help="Percentage of labels to use in confusion matrix, 0 = use unsupervised heuristics instead.",
    )
    parser.add_argument(
        "--perc_initial_descriptors",
        type=float,
        default=_default_args["perc_initial_descriptors"],
        help="Percentage of initial descriptors to use.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=_default_args["shots"],
        help="Number of shots per class to use to train model (only used in fewshot_model).",
    )
    parser.add_argument(
        "--lm4cv_kwarg_set",
        type=int,
        default=_default_args["lm4cv_kwarg_set"],
        help="Which set of LM4CV hyperparameters to use.",
    )
    parser.add_argument(
        "--ablate_vlm", action="store_true", help="Ablate VLM from the model."
    )
    parser.add_argument(
        "--deduplicate_descriptors",
        action="store_true",
        help="Whether to deduplicate descriptors.",
    )
    # Parse arguments
    return parser


def get_initial_descriptors(
    dataset_name: str,
    perc_initial_descriptors: float,
    salt: str,
):
    # Load descriptors based on the argument
    initial_descriptors = load_obj(
        f"descriptors/cbd_descriptors/descriptors_{dataset_name}.json"
    )

    # subsample initial descriptors
    if perc_initial_descriptors < 1.0:
        assert (
            salt != ""
        ), "Salt must be provided for subsampling so we don't overwrite the initial descriptors."
        for c, descriptors in initial_descriptors.items():
            initial_descriptors[c] = descriptors[
                : int(len(descriptors) * perc_initial_descriptors)
            ]
    return initial_descriptors


def main():
    parser = get_parser()
    args = parser.parse_args()

    initial_descriptors = get_initial_descriptors(
        dataset_name=args.dataset_name,
        perc_initial_descriptors=args.perc_initial_descriptors,
        salt=args.salt,
    )

    print(f"Dataset: {args.dataset_name}")
    print(f"Using CLIP model: {args.clip_model_name}")
    if args.scoring_clip_model_name != "":
        print(f"Using scoring CLIP model: {args.scoring_clip_model_name}")
    print(f"Number of iterations: {args.start_iteration}/{args.num_iters}")

    model, library, start_iteration = get_model(
        initial_descriptors=initial_descriptors,
        clip_model_name=args.clip_model_name,
        scoring_clip_model_name=args.scoring_clip_model_name,
        use_open_clip=args.use_open_clip,
        openai_model=args.openai_model,
        openai_temp=args.openai_temp,
        correlation_matrix_threshold=args.correlation_matrix_threshold,
        selection_proportion=args.selection_proportion,
        prompt_type=args.prompt_type,
        classwise_topk=args.classwise_topk,
        distance_type=args.distance_type,
        topk=args.topk,
        subselect=args.subselect,
        decay_factor=args.decay_factor,
        shots=args.shots,
        salt=args.salt,
        algorithm=args.algorithm,
        dataset_name=args.dataset_name,
        iteration=args.start_iteration,
        lm4cv_kwarg_set=args.lm4cv_kwarg_set,
    )

    run_experiment(
        model=model,
        dataset_name=args.dataset_name,
        num_iters=args.num_iters,
        clip_model_name=args.clip_model_name,
        scoring_clip_model_name=args.scoring_clip_model_name,
        device=args.device,
        use_open_clip=args.use_open_clip,
        image_size=args.image_size,
        use_interpretability_critic=args.use_interpretability_critic,
        start_iteration=start_iteration,
        perc_labels=args.perc_labels,
        ablate_vlm=args.ablate_vlm,
        deduplicate_descriptors=args.deduplicate_descriptors,
    )


if __name__ == "__main__":
    main()
