import torch

from plant_leaves.data import MyDataset, load_processed_data
from tests import _PATH_DATA


def test_data():
    # load datset
    print(_PATH_DATA)
    train_set, test_set, validation_set = load_processed_data(_PATH_DATA)
    for dataset, target in [(train_set, 4274), (test_set, 110), (validation_set, 110)]:
        assert len(dataset) == target, "Wrong number of samples in dataset"
        x, _ = dataset[0]
        assert x.shape == (3, 240, 240) or x.shape == (3, 288, 288)
