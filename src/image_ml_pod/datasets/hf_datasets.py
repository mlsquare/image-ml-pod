import os
from typing import Any, Dict

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from kedro.io import AbstractDataset


class HFImageFolderDataSet(AbstractDataset):
    """
    A Kedro dataset for handling image datasets in the Hugging Face `imagefolder` format.

    This dataset allows you to load image datasets stored in a folder structure compatible
    with Hugging Face's `datasets` library. The dataset is read-only and does not support
    saving changes.

    Attributes:
        data_dir (str): Path to the directory containing the image dataset.
        include_metadata (bool): Whether to include metadata during the dataset loading.

    Example:
        The directory structure for an image dataset might look like this:
        ```
        data_dir/
        ├── train/
        │   ├── class_a/
        │   │   ├── image1.jpg
        │   │   ├── image2.jpg
        │   │   └── ...
        │   ├── class_b/
        │   │   ├── image3.jpg
        │   │   ├── image4.jpg
        │   │   └── ...
        │   └── ...
        ├── validation/
        │   ├── class_a/
        │   │   ├── image5.jpg
        │   │   ├── image6.jpg
        │   │   └── ...
        │   ├── class_b/
        │   │   ├── image7.jpg
        │   │   ├── image8.jpg
        │   │   └── ...
        │   └── ...
        └── test/
            ├── class_a/
            │   ├── image9.jpg
            │   ├── image10.jpg
            │   └── ...
            ├── class_b/
            │   ├── image11.jpg
            │   ├── image12.jpg
            │   └── ...
            └── ...
        ```

        Here, the dataset is organized into subdirectories for each split (e.g., "train", "validation", "test"),
        with further subdirectories for each class label (e.g., "class_a", "class_b").

        Reference: https://huggingface.co/docs/datasets/en/image_dataset#imagefolder

    Usage in Kedro Data Catalog:

    To use `HFImageFolderDataSet` in the Kedro data catalog, add an entry to the `catalog.yml` file:

        ```yaml
        my_image_dataset:
          type: image_ml_pod.datasets.HFImageFolderDataSet
          data_dir: data/01_raw/images
        ```
    """

    def __init__(self, data_dir: str):
        """
        Initializes the HFImageFolderDataSet.

        Args:
            data_dir (str): Path to the directory containing the image dataset.
            include_metadata (bool): Optional; If True, includes metadata during dataset loading.
        """
        self._data_dir = data_dir

    def _load(self) -> DatasetDict:
        """
        Loads the image dataset from the specified directory using Hugging Face's `datasets` library.

        Returns:
            DatasetDict: A dictionary of datasets, where each key represents a split (e.g., "train").
        """
        dataset = load_dataset("imagefolder", data_dir=self._data_dir)
        return dataset

    def _save(self, data: Any) -> None:
        """
        Saving is not implemented for this dataset.

        Raises:
            NotImplementedError: Always raised, as this dataset is read-only.
        """
        raise NotImplementedError(
            "Image folder datasets are read-only. Please use the "
            "`image_ml_pod.datasets.HFDatasetWrapper` to save a dataset."
        )

    def _exists(self) -> bool:
        """
        Checks whether the specified data directory exists.

        Returns:
            bool: True if the data directory exists, False otherwise.
        """
        return os.path.exists(self._data_dir)

    def _describe(self) -> Dict[str, Any]:
        """
        Provides a description of the dataset configuration.

        Returns:
            Dict[str, Any]: A dictionary describing the dataset configuration.
        """
        return {
            "data_dir": self._data_dir,
        }



class HFDatasetWrapper(AbstractDataset):
    """
    A Kedro dataset for managing Hugging Face datasets with save and load capabilities.

    This dataset allows loading and saving datasets to disk using Hugging Face's `datasets` library.
    It provides an interface for handling datasets stored in the Hugging Face `Dataset` format.

    Attributes:
        dataset_path (str): Path to the directory where the dataset is stored.

    Usage in Kedro Data Catalog:
        To use `HFDatasetWrapper` in the Kedro data catalog, add an entry to the `catalog.yml` file:

        ```yaml
        my_huggingface_dataset:
          type: image_ml_pod.datasets.HFDatasetWrapper
          dataset_path: data/processed/my_dataset
        ```
    """

    def __init__(self, dataset_path: str):
        """
        Initializes the HFDatasetWrapper.

        Args:
            dataset_path (str): Path to the directory where the dataset is stored.
        """
        self.dataset_path = dataset_path

    def _load(self) -> Dataset:
        """
        Loads the dataset from the specified directory.

        Returns:
            Dataset: A Hugging Face Dataset object loaded from the given path.
        """
        return load_from_disk(self.dataset_path)

    def _save(self, data: Dataset) -> None:
        """
        Saves the dataset to the specified directory.

        Args:
            data (Dataset): The dataset to save.

        Returns:
            None
        """
        data.save_to_disk(self.dataset_path)

    def _exists(self) -> bool:
        """
        Checks whether the dataset exists at the specified path.

        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        return os.path.exists(self.dataset_path)

    def _describe(self) -> Dict[str, Any]:
        """
        Provides a description of the dataset configuration.

        Returns:
            Dict[str, Any]: A dictionary describing the dataset configuration.
        """
        return {
            "dataset_path": self.dataset_path,
        }
