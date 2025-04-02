import numpy as np


def expected_calibration_error(samples, true_labels, M=10):
    """
    Calculate the Expected Calibration Error (ECE) with M-bin uniform binning.
    """
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(samples, axis=1)  # maximum probability for each sample
    predicted_label = np.argmax(samples, axis=1)
    accuracies = predicted_label == true_labels

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prob_in_bin = np.mean(in_bin)

        if prob_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return ece


# ece_val = expected_calibration_error(val_scores.numpy(), val_labels.numpy())
# ece_test = expected_calibration_error(test_scores.numpy(), test_labels.numpy())
# results = {"dataset": dataset, "iteration": last_iter, "ece_val": ece_val.item(), "ece_test": ece_test.item()}
# # add to jsonl file of all ece values
# print(results)
# with open("ece_results.jsonl", "a") as f:
#     f.write(json.dumps(results) + "\n")
