from typing import Tuple

import torch
from PIL import Image
from pytorch_ood.detector import MultiMahalanobis
from pytorch_ood.model import WideResNet
from ..data_preprocessing.nodes import apply_transforms


class OutOfDistributionError(Exception):
    """
    Custom exception for handling out-of-distribution (OOD) errors.

    Args:
        message: The error message to be displayed.
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def prepare_image(img: Image, parameters: dict) -> torch.Tensor:
    """
    Placeholder function for preparing an image, potentially applying additional transforms.

    Args:
        img: The input image to prepare.
        parameters: A dictiionary containing specific parameters for the transformation (e.g., size).

    Returns:
        torch.Tensor: The prepared image in tensor format.
    """
    # Possible Customizations:
    # 1. Use `apply_transforms(img, parameters)` to apply domain-specific preprocessing.
    # 2. Resize the image before returning.
    # 3. Add normalization based on the dataset being used.

    # Uncomment the line below if `apply_transforms` is implemented and needed.
    # img = apply_transforms(img, parameters)
    return img


def prepare_data_for_ood(img: Image) -> torch.Tensor:
    """
    Prepares an image for out-of-distribution (OOD) detection by applying the WideResNet transformations.

    Args:
        img: The input image to prepare.

    Returns:
        torch.Tensor: The image transformed and ready for OOD detection.
    """
    transform = WideResNet.transform_for("cifar10-pt")  # Pre-trained transformations for CIFAR10.

    # Possible Customizations:
    # 1. Replace "cifar10-pt" with a different dataset transformation if using another pre-trained model.
    # 2. Add additional transforms (e.g., cropping, flipping) before or after `transform(img)`.

    img = transform(img)  # Apply the CIFAR10-specific transform.
    return img


def ood_detection(
    img: torch.Tensor, detector: MultiMahalanobis, threshold: float, device: str
) -> float:
    """
    Perform out-of-distribution (OOD) detection on a given image.

    Args:
        img: A tensor representing the image to be checked for OOD.
        detector: An instance of the MultiMahalanobis detector.
        threshold: The score threshold for classifying the image as OOD.
        device: The device to use for computation (e.g., 'cpu' or 'cuda').

    Returns:
        float: The OOD score of the image.

    Raises:
        OutOfDistributionError: If the OOD score exceeds the threshold.
    """
    img = img.unsqueeze(0)  # Add a batch dimension to the image tensor.

    # Move the detector's components to the specified device.
    detector.model = [model.to(device) for model in detector.model]
    detector.mu = [mu.to(device) for mu in detector.mu]
    detector.precision = [precision.to(device) for precision in detector.precision]

    # Compute the OOD score for the image.
    score = detector(img.to(device)).item()

    # Possible Customizations:
    # 1. Adjust the exception message to include additional information, such as the model or dataset used.
    # 2. Log the OOD score (e.g., for debugging purposes) before returning or raising an exception.
    # 3. Introduce soft thresholds or confidence intervals for borderline cases.

    if score > threshold:
        raise OutOfDistributionError(
            f"Image detected as OOD with score {score} which is above threshold {threshold}"
        )

    return score
