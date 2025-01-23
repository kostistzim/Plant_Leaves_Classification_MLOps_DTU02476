import io
import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import AsyncGenerator, Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms
from prometheus_client import Counter, Histogram, Summary, make_asgi_app, CollectorRegistry

DEVICE: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
LOG_PREFIX: str = "TESTING"

# Prometheus metrics
prometheus_registry = CollectorRegistry()
error_counter = Counter(
    "prediction_error",
    "Number of prediction errors",
    registry=prometheus_registry
)
request_counter = Counter(
    "prediction_requests",
    "Number of prediction requests",
    registry=prometheus_registry
)
request_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    registry=prometheus_registry
)
review_summary = Summary(
    "review_length_summary",
    "Review length summary",
    registry=prometheus_registry
)

class PredictionResponse(BaseModel):
    image_label: str
    confidence: float
    status_code: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load and clean up the model on startup and shutdown.

    Parameters:
    - app: FastAPI application instance.

    Yields:
    - None: Context manager for the lifespan of the application.
    """
    global model
    model = onnx.load("models/model.onnx")
    onnx.checker.check_model(model)
    yield

    print("Cleaning up")
    del model


app: FastAPI = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app(registry=prometheus_registry))


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
    - images: Tensor of shape (N, 3, 240, 240).

    Returns:
    - torch.Tensor: Normalized tensor of images.
    """
    return (images - images.mean()) / images.std()


async def preprocess_image(data: UploadFile) -> np.ndarray:
    """
    Transforms an image file into a normalized tensor.

    Parameters:
    - data: Uploaded image file.

    Returns:
    - torch.Tensor: Normalized tensor representation of the image.
    """

    contents = await data.read()
    img = Image.open(io.BytesIO(contents))

    # Use io.BytesIO for in-memory binary streams
    transform = transforms.Compose(
        [
            transforms.Resize((240, 240)),  # TODO: Resize to 240x240 after re-training the model
            transforms.ToTensor(),  # Convert to tensor
        ]
    )
    tensor = transform(img)
    norm_tensor = normalize(tensor)
    norm_tensor = torch.unsqueeze(norm_tensor, 0)

    return norm_tensor.numpy()


@app.post("/predict/", response_model=PredictionResponse)
async def predict(data: UploadFile = File(...)) -> PredictionResponse:
    """
    Predict whether the image is healthy or diseased.
    """
    request_counter.inc()
    with request_latency.time():
        try:
            image_array = await preprocess_image(data)
            review_summary.observe(image_array.size)
            model_path = os.getenv("MODEL_PATH", "models/model.onnx")
            ort_sess = ort.InferenceSession(model_path)
            y_pred = ort_sess.run(None, {"input": image_array})
            prediction_idx = y_pred[0][0].argmax(0)
            label = "healthy" if prediction_idx.item() == 0 else "diseased"
            confidence = y_pred[0][0][prediction_idx].item()
            print(y_pred, prediction_idx, label, confidence)
            return PredictionResponse(image_label=label, confidence=confidence, status_code=200)
        
        except Exception as e:
            error_counter.inc()
            print(f"Error: {e}")
            return PredictionResponse(image_label="error", confidence=0.0, status_code=500)
