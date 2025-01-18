import os
from datetime import datetime
from pathlib import Path
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import torch
import wandb
from dotenv import load_dotenv
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader

from plant_leaves import PROJECT_ROOT
from plant_leaves.data import load_processed_data
from plant_leaves.config.logging_config import logger
from plant_leaves.model import PlantClassifier

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DATA_PATH = Path(os.getenv("DATA_PATH"))
LOG_PREFIX = "TRAINING"



@hydra.main(
    config_path=os.path.join(PROJECT_ROOT, "configs"),
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

    # Load WandB environment variables
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")

    if not all([wandb_api_key, wandb_project, wandb_entity]):
        logger.info(
            "Please set WANDB_API_KEY, WANDB_PROJECT, and WANDB_ENTITY in the environment variables"
        )  # logged as DEBUG
        mode = "disabled"
    else:
        logger.info(f"Logging in with api key to project {wandb_project} for entity {wandb_entity}")  # logged as DEBUG
        wandb.login(key=wandb_api_key)
        mode = "online"

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{current_time}_lr_{params.lr}_bs_{params.batch_size}_epochs_{params.epochs}"
    run = wandb.init(
        project=wandb_project,  # Group all experiments for this project
        entity=wandb_entity,  # Specify the team or user account
        job_type="train",  # Specify the type of job
        name=run_name,
        config={"lr": params.lr, "batch_size": params.batch_size, "epochs": params.epochs},
        mode=mode,
    )

    model = PlantClassifier().to(DEVICE)
    train_set, _, validation_set = load_processed_data(DATA_PATH)

    logger.configure(extra={"prefix": LOG_PREFIX})
    logger.info(f"Train set size: {len(train_set)}")
    logger.info(f"Validation set size: {len(validation_set)}")

    # train_dataloader = DataLoader(dataset=train_set, batch_size=params.batch_size)
    train_dataloader = DataLoader(dataset=train_set, batch_size=params.batch_size)
    val_dataloader = DataLoader(dataset=validation_set, batch_size=params.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    statistics: Dict[str, list[float]] = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(params.epochs):
        model.train()
        epoch_loss, epoch_accuracy = 0.0, 0.0
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())
            epoch_loss += loss.item()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            epoch_accuracy += accuracy
            statistics["train_accuracy"].append(accuracy)

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            if i % 100 == 0:
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        epoch_loss = epoch_loss / len(train_dataloader)
        epoch_accuracy = epoch_accuracy / len(train_set)

        logger.info(f"Epoch {epoch}, loss: {epoch_loss}, accuracy: {epoch_accuracy}")
        wandb.log({"epoch_train_loss": epoch_loss, "epoch_train_accuracy": epoch_accuracy})

        # Validation loop
        model.eval()
        val_loss, val_correct = 0.0, 0

        with torch.no_grad():
            for img, target in val_dataloader:
                img, target = img.to(DEVICE), target.to(DEVICE)
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                val_loss += loss.item()
                val_correct += (y_pred.argmax(dim=1) == target).float().sum().item()

        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / len(validation_set)
        statistics["val_loss"].append(val_loss)
        statistics["val_accuracy"].append(val_accuracy)

        wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})
        logger.info(f"Validation loss: {val_loss}, accuracy: {val_accuracy}")

    logger.info("Training complete")
    model_path = f"{PROJECT_ROOT}/models/model.pth"
    torch.save(model.state_dict(), model_path)
    onnx_model_path = f"{PROJECT_ROOT}/models/model.onnx"
    torch.onnx.export(
        model,
        (img[0].unsqueeze(0),),
        onnx_model_path,
        input_names=["input"],  # the model's input names
        output_names=["output"],
    )
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig_path = f"{PROJECT_ROOT}/reports/figures/training_statistics.png"
    fig.savefig(fig_path)
    run.log({"training_statistics_via_matplotlib": wandb.Image(fig_path)})

    # Log model to wandb
    artifact = wandb.Artifact(
        name="leaf_classifier",
        type="model",
        description="A model trained to classify healthy and diseased plant leaves",
        metadata={"accuracy": statistics["train_accuracy"][-1], "loss": statistics["train_loss"][-1]},
    )
    artifact.add_file(model_path)
    artifact.add_file(onnx_model_path)
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":
    train()
