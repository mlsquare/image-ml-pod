"""
This is a boilerplate pipeline 'inf_postprocessing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import conformal_prediction, integrated_gradients, log_prediction


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=conformal_prediction,
                inputs=["prediction", "cp_predictor"],
                outputs="conformal_prediction",
                name="conformal_prediction_node",
                tags=["inference"],
            ),
            node(
                func=integrated_gradients,
                inputs=[
                    "best_model",
                    "normalized_img",
                    "resized_img",
                    "conformal_prediction",
                    "params:device",
                ],
                outputs=["integrated_gradients", "predictions"],
                name="integrated_gradients_node",
                tags=["inference"],
            ),
            node(
                func=log_prediction,
                inputs="conformal_prediction",
                outputs=None,
                name="log_prediction_node",
                tags=["inference"],
            ),
        ]
    )
