from pathlib import Path

import typer
from torch.utils.data import Dataset
import kagglehub


# Download latest version



class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)

def load_data(raw_data_path: str = "../../data/raw/tmp.zip") -> None:
    # kagglehub.login()
    # kaggle_path = kagglehub.dataset_download("arifmia/heart-attack-risk-dataset", path=raw_data_path)

    # print("Path to dataset files:", kaggle_path)
    return None

if __name__ == "__main__":
    # typer.run(preprocess)
    typer.run(load_data)