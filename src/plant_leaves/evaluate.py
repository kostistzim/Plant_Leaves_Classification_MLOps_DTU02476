import torch
import typer
from config.logging_config import logger
from model import PlantClassifier
from torch.utils.data import DataLoader

from src.plant_leaves.data import LOG_PREFIX, load_processed_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
LOG_PREFIX = "TESTING"


def evaluate(model_checkpoint: str) -> None:
    """
    Takes a pretrained CNN model and evaluates it on test data

            Parameters:
                        model_checkpoint (str): a PathLike object containing a file name

    """
    logger.configure(extra={"prefix": LOG_PREFIX})

    print(model_checkpoint)
    model = PlantClassifier().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, _, test_set = load_processed_data()

    logger.info(f"Train set size: {len(test_set)}")

    test_dataloader = DataLoader(test_set, batch_size=64)

    model.eval()
    correct, total = 0, 0

    for img, target in test_dataloader:
        logger.info("Initiate model evaluation on test data...")
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    logger.info(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    logger.configure(extra={"prefix": LOG_PREFIX})
    logger.remove(0)
    logger.add("logging.log", format="[{extra[prefix]}] | {time:MMMM D, YYYY > HH:mm:ss} | {level} | {message}")
    typer.run(evaluate)
