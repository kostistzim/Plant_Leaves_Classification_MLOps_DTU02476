import os
from pathlib import Path
from typing import Tuple

import kagglehub
import torch
import typer
from PIL import Image
from torchvision import transforms

from plant_leaves.config.logging_config import logger

data_typer = typer.Typer()
LOG_PREFIX = "DATA-HANDLING"


@data_typer.command()
@logger.catch()
def download_dataset(
    dataset: str = typer.Argument("csafrit2/plant-leaves-for-image-classification", help="Kaggle dataset identifier"),
    destination: str = typer.Argument("data/raw", help="Destination folder for the dataset"),
) -> None:
    logger.configure(extra={"prefix": LOG_PREFIX})
    """
    Download the dataset from Kaggle. Kaggle API must be installed and configured (https://www.kaggle.com/docs/api#authentication).
    Note: The dataset is downloaded to the kagglehub cache folder and then moved to the destination folder.

        Parameters:
            dataset (str, optional): Default value is "csafrit2/plant-leaves-for-image-classification".
            destination (str, optional): Default value is "data/raw".

        Returns:
            None
    """
    # configure_logger()
    try:
        kagglehub.whoami()
    except Exception as e:
        logger.warning(
            "Please setup Kaggle API credentials first (using kaggle.json). Check https://www.kaggle.com/docs/api#authentication"
        )
        raise
    # When download_dataset is called from another function with defaults, the arguments are ArgumentInfo objects
    if isinstance(dataset, typer.models.ArgumentInfo):
        dataset = str(dataset.default)
    if isinstance(destination, typer.models.ArgumentInfo):
        destination = str(destination.default)
    dataset_folder_name = dataset.split("/")[-1]
    destination = os.path.join(destination, dataset_folder_name)
    destination = os.path.normpath(destination)

    if os.path.exists(destination):
        logger.info(f"Dataset {dataset} already downloaded in {destination}")
        return

    try:
        path = kagglehub.dataset_download(dataset)
        path = os.path.normpath(path)
        logger.info(f"Path (kagglehub cache) to dataset files:", path)
        os.rename(path, destination)
        logger.info(f"Files moved to:", destination)
    except Exception as e:
        logger.error(f"Error downloading or moving dataset: {e}")
        raise


@data_typer.command()
@logger.catch()
def preprocess(
    raw_data_path: Path = typer.Argument(
        default=Path("data/raw/plant-leaves-for-image-classification/Plants_2"),
        help="Path to the folder containing raw data.",
    ),
    output_folder: Path = typer.Argument(
        default=Path("data/processed"), help="Path to the folder where processed data will be stored."
    ),
    dimensions: Tuple[int, int] = typer.Option(
        (240, 240), help="Target dimensions for image resizing (width, height)."
    ),
) -> None:
    """
    Preprocess the raw data and save the processed data in the output folder.

        Parameters:
            raw_data_path: Path to the folder containing raw data.
            output_folder: Path to the folder where processed data will be stored.
            dimensions: Target dimensions for image resizing (width, height).

        Returns:
            None
    """
    logger.configure(extra={"prefix": LOG_PREFIX})

    logger.info(f"Checking if raw data folder exists...")
    try:
        if not raw_data_path.exists():
            logger.info(f"The raw data folder does not exist. Downloading the dataset...")
            download_dataset()  # TODO: Add arguments properly. Issue: raw_data_path links to data folder and not to coockie cutter raw data folder.
    except Exception as e:  # If the download failed, exit.
        logger.error("Download failed. Exiting...")
        exit()

    logger.info(f"Preprocessing data...")
    for dataset_path in raw_data_path.iterdir():
        if dataset_path.name == "images to predict":
            continue
        # Check if the folder exists and if not create it
        if not os.path.exists(os.path.join(output_folder, dataset_path.name)):
            os.makedirs(os.path.join(output_folder, dataset_path.name))

        dataset_path = Path(dataset_path)
        output_path = Path(os.path.join(output_folder, dataset_path.name))
        logger.info(f"Dataset path : {dataset_path} \n Output path : {output_path}")
        if dataset_path.name == "train":
            main_preprocessing(dataset_path, output_path, dimensions=dimensions)
        else:
            main_preprocessing(dataset_path, output_path)


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images as (X - mean(X)) / std(X).

    Parameters:
    - images: Tensor of shape (N, 3, 244 or 288, 244 or 288)

    Returns:
    - Normalized images
    """
    return (images - images.mean()) / images.std()


def main_preprocessing(data_path: Path, output_path: Path, dimensions: Tuple[int, int] = (288, 288)) -> None:
    """
    Output two folders for each category in the dataset respectively.

    Parameters:
    - data_path: Path to the folder containing raw data.
    - output_path: Path to the folder where processed data will be stored.
    - dimensions: Target dimensions for image resizing (width, height).

    Returns:
    - None
    """
    logger.configure(extra={"prefix": LOG_PREFIX})

    transform = transforms.Compose(
        [
            transforms.Resize(dimensions),  # Resize to 224x224 for most CNNs
            transforms.ToTensor(),  # Convert to tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ]
    )
    datasets_pt_l, targets_pt_l = [], []
    # Extract images to output folders
    for folder_path in data_path.iterdir():
        for img_path in folder_path.iterdir():
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                category = "healthy" if "healthy" in img_path.name else "diseased"

                img = Image.open(img_path)

                # Apply transformations
                tensor = transform(img)

                if category == "healthy":
                    datasets_pt_l.append(tensor)
                    targets_pt_l.append(0)
                else:
                    datasets_pt_l.append(tensor)
                    targets_pt_l.append(1)

    # Transform lists to tensors
    datasets_pt = torch.stack(datasets_pt_l)
    targets_pt = torch.tensor(targets_pt_l)

    # Normalize the datasets
    datasets_pt = normalize(datasets_pt)

    # Save the datasets and targets as .pt files
    torch.save(datasets_pt, output_path / "datasets.pt")
    torch.save(targets_pt, output_path / "targets.pt")

    logger.info(f"Datasets saved in {output_path}")


@logger.catch()
def load_processed_data(
    processed_data_path: Path,
) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """
    Load the processed datasets and targets.

    Parameters:
    - processed_data_path: Path to the folder containing processed data.

    Returns:
    - Tuple of three torch.utils.data.TensorDataset objects: train, test, validation
    """
    logger.configure(extra={"prefix": LOG_PREFIX})

    logger.info(f"Loading processed data...")

    # Load the processed datasets and targets
    train_images = torch.load(processed_data_path / "train" / "datasets.pt", weights_only=True)
    train_target = torch.load(processed_data_path / "train" / "targets.pt", weights_only=True)
    test_images = torch.load(processed_data_path / "test" / "datasets.pt", weights_only=True)
    test_target = torch.load(processed_data_path / "test" / "targets.pt", weights_only=True)
    validation_images = torch.load(processed_data_path / "valid" / "datasets.pt", weights_only=True)
    validation_target = torch.load(processed_data_path / "valid" / "targets.pt", weights_only=True)

    # Create the joint datasets with targets
    train = torch.utils.data.TensorDataset(train_images, train_target)
    test = torch.utils.data.TensorDataset(test_images, test_target)
    validation = torch.utils.data.TensorDataset(validation_images, validation_target)

    logger.info(f"Data loaded successfully")

    return train, test, validation


if __name__ == "__main__":
    # configure_logger()
    data_typer()
