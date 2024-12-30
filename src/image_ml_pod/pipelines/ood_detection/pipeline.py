"""
This is a boilerplate pipeline 'ood_detection'
generated using Kedro 0.19.9.

This pipeline focuses on preparing data, training a WideResNet model,
and implementing out-of-distribution (OOD) detection using various
detection techniques.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    multi_mahalanobis_detector,
    prepare_data,
    train_wide_resnet,
    rmd_detector,
    msp_detector,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the OOD detection pipeline.

    The pipeline includes the following stages:
    1. Data preparation: Prepares in-distribution and out-of-distribution datasets.
    2. Model training: Trains a WideResNet model on the in-distribution data.
    3. OOD detection: Implements OOD detection using the Multi-Mahalanobis detector.

    To customize:
    - Enable or add additional nodes for other OOD detectors such as `rmd_detector` or `msp_detector`.
    - Adjust pipeline structure based on specific dataset requirements or analysis needs.

    Args:
        kwargs: Additional keyword arguments, not currently used.

    Returns:
        Pipeline: A Kedro pipeline instance.
    """
    return pipeline(
        [
            # Data preparation node
            node(
                func=prepare_data,
                inputs=["raw", "params:ood_detection_out_ds"],
                outputs=["train_in_ds", "test_in_ds", "out_ds"],
                name="ood_prepare_data",
                tags=["model_retrained"],
            ),
            # Model training node
            node(
                func=train_wide_resnet,
                inputs=[
                    "train_in_ds",
                    "test_in_ds",
                    "params:wide_resnet_epochs",
                    "params:wide_resnet_batch_size",
                    "params:device",
                ],
                outputs="wide_resnet_model",
                name="train_wide_resnet",
                tags=["model_retrained"],
            ),
            # Multi-Mahalanobis detector node
            node(
                func=multi_mahalanobis_detector,
                inputs=[
                    "wide_resnet_model",
                    "train_in_ds",
                    "test_in_ds",
                    "out_ds",
                    "params:detector_batch_size",
                ],
                outputs=["ood_detection_metrics", "ood_detector"],
                name="multi_mahalanobis_detector",
                tags=["model_retrained"],
            ),
            # Optional: Additional detectors (commented out by default)
            # Uncomment these nodes to include RMD or MSP detectors in the pipeline.
            
            # RMD Detector node
            # node(
            #     func=rmd_detector,
            #     inputs=[
            #         "wide_resnet_model",
            #         "train_in_ds",
            #         "test_in_ds",
            #         "out_ds",
            #         "params:detector_batch_size",
            #     ],
            #     outputs=["ood_detection_metrics", "ood_detector"],
            #     name="rmd_detector",
            #     tags=["model_retrained"],
            # ),
            
            # MSP Detector node
            # node(
            #     func=msp_detector,
            #     inputs=[
            #         "wide_resnet_model",
            #         "train_in_ds",
            #         "out_ds",
            #         "params:detector_batch_size",
            #     ],
            #     outputs="ood_detection_metrics",
            #     name="msp_detector",
            #     tags=["model_retrained"],
            # ),
        ]
    )
