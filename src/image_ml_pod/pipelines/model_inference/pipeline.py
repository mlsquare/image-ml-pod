"""
This is a boilerplate pipeline 'model_inference'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the 'model_inference' pipeline.

    Returns:
        Pipeline: A Kedro pipeline for performing model inference.
    """
    return pipeline(
        [
            node(
                func=predict,
                inputs=["model", "normalized_img"],
                outputs="prediction",
                name="predict",
                tags=["inference"],
            )
        ]
    )
