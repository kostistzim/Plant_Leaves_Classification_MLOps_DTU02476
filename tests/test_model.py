import torch

from plant_leaves.model import PlantClassifier


def test_model_train():
    # Create the model
    model = PlantClassifier()

    for bs in [3, 12, 16]:
        for w, h in [(10, 10), (240, 240), (288, 240), (240, 288)]:
            # Generate a batch of test images
            test_image = torch.rand([bs, 3, w, h])

            # Compute the result
            out = model(test_image)

            assert out.shape == (bs, 2)


def test_model_predict():
    # Create the model
    model = PlantClassifier()

    for bs in [1, 3, 12, 16]:
        for w, h in [(10, 10), (240, 240), (288, 240), (240, 288)]:
            # Generate a batch of` test images
            test_image = torch.rand([bs, 3, w, h])

            model.eval()
            with torch.no_grad():
                # Compute the result
                out = model(test_image)
                assert out.shape == (bs, 2)
