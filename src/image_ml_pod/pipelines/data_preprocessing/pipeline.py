"""
This is a boilerplate pipeline 'data_preprocessing' 
generated using Kedro 0.19.9.
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_data, apply_transforms


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a Kedro pipeline for preprocessing image data.

    The pipeline consists of the following stages:
    1. Splitting the dataset into train, validation, and test sets.
    2. Applying transformations (e.g., resizing, tensor conversion) to each split.

    Args:
        **kwargs: Additional keyword arguments for pipeline customization.

    Returns:
        Pipeline: A Kedro pipeline object representing the data preprocessing pipeline.

    Customization:
        - Replace `"data"` in the `inputs` argument of the `split_data_node` with the name of your raw input dataset
          as defined in the Kedro data catalog.
        - Adjust the `params:transform_params` in the `apply_transforms` nodes to include specific parameters
          for your transformations, such as `image_size` or additional augmentations.
        - Add or modify nodes if additional preprocessing steps are needed, such as normalization or data augmentation.
    """
    return pipeline([
        # Node for splitting the data
        node(
            func=split_data,
            inputs="data",  # Replace with the name of your raw input dataset from the Kedro catalog
            outputs=["train", "validation", "test"],
            name="split_data_node",
        ),
        # Node for applying transformations to the training data
        node(
            func=apply_transforms,
            inputs=["train", "params:transform_params"],
            outputs="train_transformed",
            name="apply_transforms_train_node",
        ),
        # Node for applying transformations to the validation data
        node(
            func=apply_transforms,
            inputs=["validation", "params:transform_params"],
            outputs="validation_transformed",
            name="apply_transforms_validation_node",
        ),
        # Node for applying transformations to the test data
        node(
            func=apply_transforms,
            inputs=["test", "params:transform_params"],
            outputs="test_transformed",
            name="apply_transforms_test_node",
        ),
    ])
