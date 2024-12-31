"""
This is a boilerplate pipeline 'inf_data_preprocessing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_image, ood_detection, prepare_data_for_ood


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the 'inf_data_preprocessing' pipeline for image preprocessing
    and OOD detection during inference.

    Returns:
        Pipeline: The constructed Kedro pipeline.
    """
    return pipeline(
        [
            # Node to process the input image for inference.
            node(
                func=prepare_image,
                inputs=["inference_sample", "params:transform_params"],
                outputs="preprocessed_image",
                name="prepare_image_node_inference",
                tags=["inference"],
            ),

            # Node to prepare data for OOD detection.
            node(
                func=prepare_data_for_ood,
                inputs="inference_sample",
                outputs="img_for_ood",
                name="prepare_data_for_ood_node",
                tags=["inference"],
            ),

            # Node to perform OOD detection on the prepared data.
            node(
                func=ood_detection,
                inputs=[
                    "img_for_ood",
                    "ood_detector",
                    "params:ood_threshold",
                    "params:device",
                ],
                outputs=None,  # OOD detection raises exceptions instead of outputting data.
                name="ood_detection_node",
                tags=["inference"],
            ),
        ]
    )
