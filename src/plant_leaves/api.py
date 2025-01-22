import io
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, Dict

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import transforms

from plant_leaves.model import PlantClassifier

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
LOG_PREFIX: str = "TESTING"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load and clean up the model on startup and shutdown.

    Parameters:
    - app: FastAPI application instance.

    Yields:
    - None: Context manager for the lifespan of the application.
    """
    global model, feature_extractor, tokenizer, device, gen_kwargs
    print("Loading model")
    model = PlantClassifier().to(DEVICE)
    model.load_state_dict(torch.load("models/model.pth"))
    model.to(DEVICE)
    yield

    print("Cleaning up")
    del model, feature_extractor, tokenizer, device, gen_kwargs


app: FastAPI = FastAPI(lifespan=lifespan)


@app.get("/")
def root() -> Dict[str, str | HTTPStatus]:
    """
    Health check endpoint.

    Returns:
    - response: Dictionary containing a message and status code.
    """
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize images as (X - mean(X)) / std(X).

    Parameters:
    - images: Tensor of shape (N, 3, 244 or 288, 244 or 288).

    Returns:
    - torch.Tensor: Normalized tensor of images.
    """
    return (images - images.mean()) / images.std()


async def preprocess_image(data: UploadFile) -> torch.Tensor:
    """
    Transforms an image file into a normalized tensor.

    Parameters:
    - data: Uploaded image file.

    Returns:
    - torch.Tensor: Normalized tensor representation of the image.
    """
    content = await data.read()
    img = Image.open(io.BytesIO(content))  # Use io.BytesIO for in-memory binary streams
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224 for most CNNs
            transforms.ToTensor(),  # Convert to tensor
        ]
    )
    tensor = transform(img)
    norm_tensor = normalize(tensor)
    return norm_tensor


@app.get("/predict/", response_class=HTMLResponse)
async def get_form() -> str:
    """
    Render an HTML form for image upload.

    Returns:
    - str: HTML form as a string.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Image</title>
    </head>
    <body>
        <h1>Upload Image for Prediction</h1>
        <form action="/predict/" method="post" enctype="multipart/form-data">
            <input type="file" name="data" accept="image/*">
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    """


@app.post("/predict/")
async def predict(data: UploadFile = File(...)) -> Dict[str, str | HTTPStatus]:
    """
    Predict whether the image is healthy or diseased.

    Parameters:
    - data: Uploaded image file.

    Returns:
    - Dict[str, str | HTTPStatus]: Dictionary containing the prediction label and status code.
    """
    tensor = await preprocess_image(data)

    model.eval()
    with torch.no_grad():  # Avoid tracking gradients during inference
        y_pred = model(torch.unsqueeze(tensor, 0))
    prediction_idx = y_pred.argmax(dim=1)
    label = "healthy" if prediction_idx.item() == 0 else "diseased"
    print(f"y_pred: {y_pred} label: {label}")
    response = {
        "image_label": label,
        "status-code": HTTPStatus.OK,
    }
    return response
