
import numpy as np


def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error (RMSE).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have the same length."
        )

    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_target_score_change(before_score, after_score):
    """
    Calculate the change in a target product's score.
    """
    return after_score - before_score


def evaluate_predictions(
    actual,
    predicted,
    target_before=None,
    target_after=None
):
    """
    Calculate RMSE and optionally target score change.
    """

    rmse = calculate_rmse(actual, predicted)

    result = {
        "rmse": rmse
    }

    if target_before is not None and target_after is not None:
        result["target_score_change"] = (
            calculate_target_score_change(
                target_before,
                target_after
            )
        )

    return result
