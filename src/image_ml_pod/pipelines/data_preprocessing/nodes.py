"""
This is a boilerplate pipeline 'data_preprocessing' 
generated using Kedro 0.19.9.
"""

from typing import Tuple, Dict
from datasets import (
    Dataset,
    DatasetDict,
)
from torchvision import transforms


def split_data(data: DatasetDict) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Splits a `DatasetDict` into its constituent datasets: train, validation, and test.

    Args:
        data (DatasetDict): A Hugging Face `DatasetDict` containing the dataset splits.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The train, validation, and test datasets.

    Customization:
        - Ensure the `data` argument contains keys named "train", "validation", and "test".
          If your dataset uses different split names, update the function to match them.
        - You can modify this function to perform additional operations such as stratified splitting
          or splitting into additional subsets like a "holdout" set.
    """
    return data["train"], data["validation"], data["test"]


def apply_transforms(data: Dataset, transform_params: Dict) -> Dataset:
    """
    Applies image transformations to a Hugging Face Dataset.

    Args:
        data (Dataset): A Hugging Face `Dataset` containing image data.
        transform_params (Dict): A dictionary of transformation parameters. 
                                 Expected keys:
                                   - "image_size" (int or Tuple[int, int]): The size to which images should be resized.

    Returns:
        Dataset: The transformed dataset with images converted to tensors and formatted for PyTorch.

    Customization:
        - Update the `transforms.Compose` pipeline to include additional transformations 
          such as augmentation (e.g., rotation, flipping, or color jitter).
        - Modify the `transforms_fn` function to handle custom image-label structures 
          or to include other features in the dataset.
    """
    # Define the transformation pipeline
    transform = transforms.Compose(
        [
            transforms.Resize(transform_params["image_size"]),
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ]
    )

    # Define the mapping function to apply transformations
    def transforms_fn(example):
        return {"image": transform(example["image"]), "label": example["label"]}

    # Apply the transformations to the dataset
    data = data.map(transforms_fn)
    data.set_format("torch", columns=["image", "label"])
    return data
