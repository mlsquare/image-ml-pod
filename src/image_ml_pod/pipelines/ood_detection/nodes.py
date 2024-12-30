"""
This is a boilerplate pipeline 'ood_detection'
generated using Kedro 0.19.9.
"""

from typing import Tuple
import mlflow
import numpy as np
import torch
from datasets import DatasetDict
from pytorch_ood.detector import RMD, MaxSoftmax, MultiMahalanobis
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm


class MyLayer(nn.Module):
    """
    Custom layer for extracting intermediate features from the WideResNet model.
    Combines batch normalization and ReLU activation.

    To customize:
        - Adjust which combination of layers from the base model to include.
        - Add additional preprocessing or transformations here if needed.
    """

    def __init__(self, bn1, relu):
        super().__init__()
        self.bn1 = bn1
        self.relu = relu

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        return x


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with image-label pairs.

    Args:
        batch: A list of data samples.

    Returns:
        Tuple[Tensor, Tensor]: A tuple of images and labels.

    To customize:
        - Adjust handling if the dataset contains additional metadata or fields.
        - Add support for multi-label datasets if needed.
    """
    data, targets = [], []
    for row in batch:
        if isinstance(row, dict):
            data.append(row["image"])
            targets.append(row["label"])
        elif isinstance(row, tuple):
            data.append(row[0])
            targets.append(row[1])

    data = torch.stack(data)
    targets = torch.tensor([np.array(t).squeeze() for t in targets]).squeeze()
    return data, targets


def prepare_data(
    raw_data: DatasetDict, out_ds: str
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Prepare in-distribution and out-of-distribution datasets.

    Args:
        raw_data: A DatasetDict containing train and validation splits.
        out_ds: Name of the out-of-distribution dataset (e.g., "cifar10").

    Returns:
        Tuple of train, validation, and out-of-distribution test datasets.

    To customize:
        - Add support for more out-of-distribution datasets by extending the `if-else` block.
        - Modify transformations for specific datasets to optimize feature extraction.
    """
    transform = WideResNet.transform_for("cifar10-pt")

    def transform_fn(x):
        return {"image": transform(x["image"]), "label": x["label"]}

    train_data = raw_data["train"].map(transform_fn, writer_batch_size=200)
    val_data = raw_data["validation"].map(transform_fn, writer_batch_size=200)
    train_data.set_format("torch", columns=["image", "label"])
    val_data.set_format("torch", columns=["image", "label"])

    if out_ds == "cifar10":
        dataset_out_test = CIFAR10(
            root="~/.data",
            download=True,
            transform=transform,
            target_transform=ToUnknown(),
        )
    else:
        raise ValueError(f"Unsupported out-of-distribution dataset: {out_ds}")

    return train_data, val_data, dataset_out_test


def train_wide_resnet(
    in_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    n_epochs: int,
    batch_size: int,
    device: str,
) -> nn.Module:
    """
    Train a WideResNet model.

    Args:
        in_dataset: In-distribution dataset for training.
        test_dataset: Dataset for validation/testing.
        n_epochs: Number of training epochs.
        batch_size: Batch size for DataLoader.
        device: Device for training ("cpu" or "cuda").

    Returns:
        nn.Module: Trained WideResNet model.

    To customize:
        - Experiment with different optimizers (e.g., SGD) or learning rate schedulers.
        - Add regularization techniques such as dropout or weight decay.
        - Modify the model architecture if needed for specific datasets or tasks.
    """
    in_loader = DataLoader(in_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = WideResNet(num_classes=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(in_loader):
            x, y = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader):
                x, y = batch["image"].to(device), batch["label"].to(device)
                y_pred = model(x)
                val_running_loss += criterion(y_pred, y).item()

    return model


def multi_mahalanobis_detector(
    wide_resnet: nn.Module,
    train_in_dataset: torch.utils.data.Dataset,
    test_in_dataset: torch.utils.data.Dataset,
    out_dataset: torch.utils.data.Dataset,
    batch_size: int,
) -> Tuple[dict, MultiMahalanobis]:
    """
    Perform OOD detection using the Multi-Mahalanobis detector.

    Args:
        wide_resnet: Trained WideResNet model.
        train_in_dataset: In-distribution training dataset.
        test_in_dataset: In-distribution test dataset.
        out_dataset: Out-of-distribution test dataset.
        batch_size: Batch size for DataLoader.

    Returns:
        Tuple[dict, MultiMahalanobis]: Detection metrics and the detector.

    To customize:
        - Experiment with different feature extraction layers for the detector.
        - Adjust scoring thresholds for specific use cases or dataset characteristics.
    """
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [
        wide_resnet.conv1,
        wide_resnet.block1,
        wide_resnet.block2,
        wide_resnet.block3,
        MyLayer(wide_resnet.bn1, wide_resnet.relu),
    ]

    detector = MultiMahalanobis(layers)
    train_loader = DataLoader(
        train_in_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        ConcatDataset([test_in_dataset, out_dataset]),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )

    detector.fit(train_loader, device=device)
    metrics = OODMetrics()

    for x, y in test_loader:
        metrics.update(detector(x.to(device)), y)

    return metrics.compute(), detector

def msp_detector(
    best_model_uri: str,
    in_dataset: torch.utils.data.Dataset,
    out_dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: str,
) -> dict:
    """
    Run the Maximum Softmax Probability (MSP) detector on the given datasets.

    Args:
        best_model_uri: The URI of the best model.
        in_dataset: The in-distribution dataset.
        out_dataset: The out-of-distribution dataset.
        batch_size: The batch size to use.
        device: The device to use.

    Returns:
        The metrics of the detector."""

    in_loader = DataLoader(in_dataset, batch_size=batch_size, shuffle=True)
    out_loader = DataLoader(out_dataset, batch_size=batch_size, shuffle=True)

    model = mlflow.pytorch.load_model(best_model_uri).to(device).eval()

    detector = MaxSoftmax(model)

    metrics = OODMetrics()

    for loader in [in_loader, out_loader]:
        for _x, _y in tqdm(loader):
            x, y = _x.to(device), _y.to(device)
            metrics.update(detector(x), y.squeeze())

    return metrics.compute()


def rmd_detector(
    best_model_uri: str,
    in_dataset: torch.utils.data.Dataset,
    out_dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: str,
) -> Tuple[dict, RMD]:
    """
    Run the Relative Mahalanobis Distance (RMD) detector on the given datasets.

    Args:
        best_model_uri: The URI of the best model.
        in_dataset: The in-distribution dataset.
        out_dataset: The out-of-distribution dataset.
        batch_size: The batch size to use.
        device: The device to use.

    Returns:
        The metrics of the detector and the detector itself.
    """

    in_loader = DataLoader(
        in_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    out_loader = DataLoader(out_dataset, batch_size=batch_size, shuffle=True)

    model = mlflow.pytorch.load_model(best_model_uri).to(device).eval()

    detector = RMD(model)
    detector.fit(in_loader, device=device)

    metrics = OODMetrics()

    for loader in [in_loader, out_loader]:
        for _x, _y in tqdm(loader):
            x, y = _x.to(device), _y.to(device)
            metrics.update(detector(x), y.squeeze())

    return metrics.compute(), detector