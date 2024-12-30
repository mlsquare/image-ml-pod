"""
This is a boilerplate pipeline 'model_training' 
generated using Kedro 0.19.9.
"""

from typing import Tuple, Any
import mlflow
import numpy as np
from datasets import Dataset
from mlflow import MlflowClient


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_params: dict,
    device: str = "cpu",
) -> dict:
    """
    Trains a model using the given training and validation datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        train_params (dict): Training parameters such as learning rate, batch size, and epochs.
        device (str): The device to use for training (default: "cpu").

    Returns:
        dict: The trained model's state dictionary.

    Customization:
        - Replace the implementation with your specific training logic (e.g., loading a PyTorch or TensorFlow model).
        - Extend the function to include additional features like early stopping or logging.
    """
    pass


def evaluate_model(
    model,
    test_dataset: Dataset,
    eval_params: dict,
    device: str = "cpu",
) -> Any:
    """
    Evaluates the trained model on the test dataset.

    Args:
        model: The trained model to evaluate.
        test_dataset (Dataset): The dataset for evaluation.
        eval_params (dict): Evaluation parameters such as batch size.
        device (str): The device to use for evaluation (default: "cpu").

    Returns:
        Any: The evaluation metrics, such as accuracy or F1 score.

    Customization:
        - Replace the implementation with the evaluation logic specific to your model and task.
        - Add or modify the metrics to suit your evaluation needs (e.g., precision, recall).
    """
    pass


def log_model(
    model: Any,
    hyperparams: dict,
    metrics: dict,
    mlflow_tracking_uri: str,
) -> str:
    """
    Logs the model, hyperparameters, and metrics to MLFlow.

    Args:
        model (Any): The trained model to log.
        hyperparams (dict): The hyperparameters used during training.
        metrics (dict): The evaluation metrics of the model.
        mlflow_tracking_uri (str): The URI of the MLFlow tracking server.

    Returns:
        str: The URI of the logged model.

    Customization:
        - Update the MLflow tracking URI if not using the default local server.
        - Add additional artifacts to log, such as loss plots or model explainability data.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Start an MLflow run
    with mlflow.start_run():
        # Log model
        mlflow.pytorch.log_model(
            model,
            artifact_path=hyperparams["model_name"],
            input_example=np.random.randn(1, 3, 224, 224),
            registered_model_name=hyperparams["model_name"],
        )

        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Get the URI of the logged model
        model_uri = mlflow.get_artifact_uri(hyperparams["model_name"])

    return model_uri


def set_best_model_uri(
    model_name: str, mlflow_tracking_uri: str
) -> Tuple[str, Any]:
    """
    Selects the best model based on evaluation metrics from MLFlow and loads it.

    Args:
        model_name (str): The name of the registered model.
        mlflow_tracking_uri (str): The URI of the MLFlow tracking
        
    Returns:
        Tuple[str, Any]: The URI of the best model and the model object.

    Customization:
        - Adjust the metric used for selection (`f1` by default) to fit your use case.
        - Extend the function to handle more complex selection logic, such as multi-metric comparison.
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    best_score = 0.0
    best_model_uri = ""
    for mv in client.search_model_versions(f"name='{model_name}'"):
        run = client.get_run(mv.run_id)
        if run.data.metrics["f1"] > best_score:
            best_score = run.data.metrics["f1"]
            best_model_uri = mv.source
    model = mlflow.pytorch.load_model(best_model_uri)
    return best_model_uri, model
