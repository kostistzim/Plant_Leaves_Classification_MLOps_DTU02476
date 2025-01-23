import shutil
from pathlib import Path

import torch

from plant_leaves.data import load_processed_data, preprocess
from tests import _PATH_DATA, _PATH_TEST_DATA


def test_preprocess_data():
    processed_data_path = _PATH_TEST_DATA.joinpath("processed")
    # Delete processed data if exists
    if processed_data_path.exists():
        shutil.rmtree(processed_data_path)

    # Call preprocess
    preprocess(
        _PATH_TEST_DATA.joinpath("raw/plant-leaves-for-image-classification/Plants_2"), processed_data_path, (240, 240)
    )

    # Load preprocessed images
    loaded_images = []
    loaded_targets = []

    loaded_images.append(torch.load(processed_data_path / "train" / "datasets.pt", weights_only=True))
    loaded_images.append(torch.load(processed_data_path / "test" / "datasets.pt", weights_only=True))
    loaded_images.append(torch.load(processed_data_path / "valid" / "datasets.pt", weights_only=True))

    loaded_targets.append(torch.load(processed_data_path / "train" / "targets.pt", weights_only=True))
    loaded_targets.append(torch.load(processed_data_path / "valid" / "targets.pt", weights_only=True))
    loaded_targets.append(torch.load(processed_data_path / "test" / "targets.pt", weights_only=True))

    # Assert count and shape
    for image in loaded_images:
        assert image.shape == torch.Size([2, 3, 240, 240]) or image.shape == torch.Size([2, 3, 288, 288])
    for target in loaded_targets:
        assert target.shape == torch.Size([2])


def test_load_data():
    # load datset
    train_set, test_set, validation_set = load_processed_data(_PATH_TEST_DATA.joinpath("processed"))
    for dataset, target in [(train_set, 2), (test_set, 2), (validation_set, 2)]:
        assert len(dataset) == target, "Wrong number of samples in dataset"
        x, _ = dataset[0]
        assert x.shape == (3, 240, 240) or x.shape == (3, 288, 288)
