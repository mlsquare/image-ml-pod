"""
This is a boilerplate pipeline 'conformal_prediction'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import calibrate_predictor, evaluate_predictor


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=calibrate_predictor,
                inputs=[
                    "validation_transformed",
                    "model",
                    "params:alpha",
                    "params:penalty",
                    "params:cp_batch_size",
                ],
                outputs="cp_predictor",
                name="cp_calibrate_predictor_node",
                tags=["model_retrained"],
            ),
            node(
                func=evaluate_predictor,
                inputs=[
                    "cp_predictor",
                    "test_transformed",
                    "params:cp_batch_size",
                ],
                outputs="cp_metrics",
                name="cp_evaluate_predictor_node",
                tags=["model_retrained"],
            ),
        ]
    )
