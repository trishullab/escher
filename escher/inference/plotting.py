import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize

sns.set_style("whitegrid")


def clean_text(text):
    """
    Undo make_descriptor_sentence logic or handle parentheses
    """
    if "which is " in text:
        return text.split("which is ")[1].strip()
    elif "which has " in text:
        return text.split("which has ")[1].strip()
    if "(" in text:
        return text.split("(")[0].strip()
    return text


def plot_histogram(
    pairs,
    label,
    dataset="",
    iteration="",
    instance_idx="",
    low_confidence=-1,
    high_confidence=-1,
    pth="debug",
    title="",
    filename=None,
    is_correct=True,
    averages=None,
):
    """
    Create a bar plot of attribute scores, optionally color-coded by confidence.
    """
    attr, scores = zip(*pairs)

    # Add a row for "Average"
    if averages is not None:
        avg_val = averages if isinstance(averages, float) else float(averages)
    else:
        avg_val = sum(scores) / len(scores)

    attr = ["Average"] + list(attr)
    scores = [avg_val] + list(scores)

    np_scores = []
    # De-dupe the (attr, score) pairs
    seen = set()
    final_attr = []
    final_scores = []

    for a, s in zip(attr, scores):
        if a not in seen:
            seen.add(a)
            final_attr.append(a)
            final_scores.append(s)

    # Optionally, you can remove or keep them all
    attr = final_attr
    scores = final_scores

    np_scores = [float(s) for s in scores]

    # Set default low/high confidence if not given
    if low_confidence < 0 or high_confidence < 0:
        mu = sum(np_scores) / len(np_scores)
        sigma = (sum([(v - mu) ** 2 for v in np_scores]) / len(np_scores)) ** 0.5
        low_confidence = mu - 3 * sigma
        high_confidence = mu + 3 * sigma

    # Clean up attribute text
    attr = [clean_text(a).lower() if a != "Average" else a for a in attr]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plt.subplots_adjust(left=0.4)  # Increase left margin if attribute text is long

    # Decide on color map
    norm = Normalize(vmin=low_confidence, vmax=high_confidence)
    cmap = cm.Blues if is_correct else cm.Reds
    colors = [cmap(norm(s)) for s in np_scores]

    sns.barplot(x=np_scores, y=attr, palette=colors, width=0.6, ax=ax)

    ax.set_xlim([low_confidence, high_confidence])

    # optionally hide x ticks
    ax.set_xticks([])
    for tick in ax.get_yticklabels():
        tick.set_fontsize(14)

    # Show numeric values inside the bars
    offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
    for i, patch in enumerate(ax.patches):
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(
            width - offset,
            y,
            f"{width:.4f}",
            ha="right",
            va="center",
            color="white",
            fontsize=12,
        )

    # Title or no title
    if title:
        ax.set_title(title, pad=20, fontweight="bold")

    sns.despine(left=True, bottom=True)

    # Save figure
    if filename:
        plt.savefig(f"{filename}.pdf", bbox_inches="tight")
        plt.savefig(f"{filename}.png", bbox_inches="tight")
    else:
        # fallback path
        dir_path = f"clean_plots/{dataset}/{iteration}_{instance_idx}"
        os.makedirs(dir_path, exist_ok=True)
        plt.savefig(os.path.join(dir_path, f"{pth}.pdf"), bbox_inches="tight")
        plt.savefig(os.path.join(dir_path, f"{pth}.png"), bbox_inches="tight")

    plt.close()
