import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores import RAPS


def calibrate_predictor(
    calibration_set: Dataset,
    best_model: nn.Module,
    alpha: float,
    penalty: float,
    batch_size: int,
) -> SplitPredictor:
    """
    Calibrate a SplitPredictor using the calibration set.

    Args:
        calibration_set: The calibration set.
        best_model: The best trained model to be used for calibration.
        alpha: The significance level (e.g., 0.05 for 95% confidence).
        penalty: The penalty parameter for the RAPS score.
        batch_size: The batch size for the DataLoader.

    Returns:
        The calibrated SplitPredictor.
    """
    model = best_model

    predictor = SplitPredictor(score_function=RAPS(penalty=penalty), model=model)

    cal_loader = DataLoader(
        calibration_set,
        batch_size=batch_size,
        shuffle=True,
    )
    predictor.calibrate(cal_loader, alpha=alpha)

    return predictor


def evaluate_predictor(
    predictor: SplitPredictor, test_set: Dataset, batch_size: int
) -> dict:
    """
    Evaluate a SplitPredictor using the test set.

    Args:
        predictor: The SplitPredictor to evaluate.
        test_set: The test set.
        batch_size: The batch size for the DataLoader.

    Returns:
        A dictionary containing the evaluation metrics.
    """
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True
    )

    metrics = predictor.evaluate(test_loader)

    return metrics
