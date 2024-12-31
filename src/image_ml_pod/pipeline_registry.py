"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from kedro.pipeline.modular_pipeline import pipeline

from .pipelines.inf_data_preprocessing import (
    create_pipeline as create_inf_data_preprocessing_pipeline,
)
from .pipelines.inf_pred_postprocessing import (
    create_pipeline as create_inf_postprocessing_pipeline,
)
from .pipelines.model_inference import (
    create_pipeline as create_model_inference_pipeline,
)

def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())

    inf_data_preprocessing_pipeline = create_inf_data_preprocessing_pipeline()
    model_inference_pipeline = create_model_inference_pipeline()
    inf_postprocessing_pipeline = create_inf_postprocessing_pipeline()

    # Using only nodes with tag "inference"

    inf_data_preprocessing_nodes = inf_data_preprocessing_pipeline.only_nodes_with_tags(
        "inference"
    )
    model_inference_nodes = model_inference_pipeline.only_nodes_with_tags("inference")
    inf_postprocessing_nodes = inf_postprocessing_pipeline.only_nodes_with_tags(
        "inference"
    )

    inference_pipeline = pipeline(
        [inf_data_preprocessing_nodes, model_inference_nodes, inf_postprocessing_nodes],
        inputs=["cp_predictor", "model", "ood_detector"],
        parameters=[
            "ood_threshold",
            "transform_params",
            "device",
        ],
    )

    pipelines["inference"] = inference_pipeline

    return pipelines
