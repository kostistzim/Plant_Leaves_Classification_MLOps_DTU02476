from pathlib import Path
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import torch
from model import PlantClassifier
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from plant_leaves.data import load_processed_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DATA_PATH = Path("data/processed/")


@hydra.main(
    config_path="../../configs",
    config_name="default_config.yaml",
    version_base=None,
)
def train(cfg: DictConfig) -> None:
    """
    Takes the CNN model and performs the training process

            Parameters:
                        batch_size (int): size of training batches
                        epochs (int): number of training runs
                        lr (float): learning rate of optimizer

    """
    params = cfg.experiment
    model = PlantClassifier().to(DEVICE)
    train_set, _, _ = load_processed_data(DATA_PATH)

    train_dataloader = DataLoader(dataset=train_set, batch_size=params.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    statistics: Dict[str, list[float]] = {"train_loss": [], "train_accuracy": []}

    for epoch in range(params.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            # if i % 100 == 0:
            print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "../../models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("../../reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
