import torch

from plant_leaves.model import PlantClassifier


def test_model():
    # Create the model
    model = PlantClassifier()

    # Generate a batch of test images
    test_image = torch.rand([2, 3, 300, 255])

    # Compute the result
    out = model(test_image)
    assert out.shape == (2, 2)
