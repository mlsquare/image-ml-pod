import logging
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from torchcp.classification.predictors import SplitPredictor


def conformal_prediction(
    output: torch.Tensor, predictor: SplitPredictor
) -> List[float]:
    """
    Perform conformal prediction on the output tensor.

    Args:
        output: The output tensor.
        predictor: The SplitPredictor.

    Returns:
        The prediction.
    """
    if output.dim() == 1:
        output = output.unsqueeze(0)
    return predictor.predict_with_logits(output)[0]


def integrated_gradients(
    best_model: torch.nn.Module,
    input_processed_img: torch.Tensor,
    input_img: torch.Tensor,
    predictions: List[int],
    device: str,
) -> Tuple[List[plt.Figure], List[int]]:
    """
    Perform integrated gradients on the input image.

    Args:
        best_model: The best model.
        input_processed_img: The processed input image.
        input_img: The input image.
        predictions: The predictions.
        device: Device

    Returns:
        The visualizations.
    """

    model = best_model.to(device)
    model.eval()
    IMG_DIM = 3
    if input_processed_img.dim() == IMG_DIM:
        input_processed_img = input_processed_img.unsqueeze(0)
    input_processed_img.requires_grad = True

    ig = IntegratedGradients(model)
    input_img = np.array(input_img).transpose((1, 2, 0))
    visualizations = []
    for target in predictions:
        attribution = ig.attribute(input_processed_img.to(device), target=target)
        attr_ig = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))
        visualization = viz.visualize_image_attr(
            attr_ig,
            input_img,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Overlayed Integrated Gradients",
            use_pyplot=False,
        )
        visualizations.append(visualization[0])

    return visualizations, predictions


def log_prediction(prediction: List[int]) -> None:
    """
    Log the prediction.

    Args:
        prediction: The prediction.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Prediction: {prediction}")
