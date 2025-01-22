from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer

from plant_leaves.config.logging_config import logger
from plant_leaves.data import load_processed_data
from plant_leaves.utils import get_project_root, show_image_and_target

LOG_PREFIX = "DATA-HANDLING"


def dataset_statistics(datadir: str = "data/processed/") -> None:
    """Compute dataset statistics."""
    root_dir = Path(get_project_root())
    logger.configure(extra={"prefix": LOG_PREFIX})
    try:
        train_dataset, test_dataset, _ = load_processed_data(Path(f"{root_dir}/{datadir}"))
    except FileNotFoundError:
        logger.error(f"No processed data found at {datadir}")
        exit(1)

    print(f"------- Train dataset -------")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print(f"------- Test dataset -------")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=32)
    train_images = [train_dataset[i][0] for i in range(len(train_dataset))]
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    test_images = [test_dataset[i][0] for i in range(len(test_dataset))]
    test_labels = [test_dataset[i][1] for i in range(len(test_dataset))]

    show_image_and_target([image.permute(1, 2, 0) for image in train_images[:32]], train_labels[:32], show=False)
    plt.savefig(f"{root_dir}/outputs/images/plant_leaves.png")
    plt.close()

    train_label_distribution = torch.bincount(torch.stack(train_labels))
    test_label_distribution = torch.bincount(torch.stack(test_labels))

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
