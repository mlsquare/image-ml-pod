"""
This is a boilerplate pipeline 'model_training' 
generated using Kedro 0.19.9.
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_model, evaluate_model, log_model, set_best_model_uri


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates a Kedro pipeline for training, evaluating, and logging a machine learning model.

    The pipeline consists of the following stages:
    1. Training the model using the training and validation datasets.
    2. Evaluating the trained model on the test dataset.
    3. Logging the trained model, its hyperparameters, and evaluation metrics to MLFlow.
    4. Setting the best model URI for further use based on evaluation metrics.

    Args:
        **kwargs: Additional keyword arguments for pipeline customization.

    Returns:
        Pipeline: A Kedro pipeline object representing the model training pipeline.

    Customization:
        - Replace the dataset names (`train_dataset`, `val_dataset`, `test_dataset`) with the appropriate 
          dataset names as defined in the Kedro data catalog.
        - Modify the `train_params` and `eval_params` references to include specific hyperparameters 
          and evaluation settings suitable for your model.
        - Update the MLFlow tracking URI in `params:mlflow_tracking_uri` if using a different MLFlow server.
        - Add or replace nodes to include additional steps such as data augmentation, hyperparameter tuning,
          or advanced evaluation metrics.
    """
    return pipeline([
        # Node for training the model
        node(
            func=train_model,
            inputs=["train_dataset", "val_dataset", "params:train_params", "device"],
            outputs="model",
            name="train_model",
        ),
        # Node for evaluating the model
        node(
            func=evaluate_model,
            inputs=["model", "test_dataset", "params:eval_params", "device"],
            outputs="evaluation_metrics",
            name="evaluate_model",
        ),
        # Node for logging the model to MLFlow
        node(
            func=log_model,
            inputs=["model", "params:train_params", "evaluation_metrics", "params:mlflow_tracking_uri"],
            outputs="model_uri",
            name="log_model",
        ),
        # Node for selecting the best model URI
        node(
            func=set_best_model_uri,
            inputs=["model", "params:best_model_metric"],
            outputs=["best_model_uri", "best_model"],
            name="set_best_model_uri",
        ),
    ])
