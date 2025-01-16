"""Module defining a model predicting the healthiness of plant leaves."""

from typing import Any

import timm
import torch


class PlantClassifier(torch.nn.Module):
    """Neural network using a pre-trained backbone
    with a custom head for predicting 2 classes
    of plant leaves

    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b1.ra4_e3600_r240_in1k", pretrained=True, num_classes=2)

    def forward(self, x: torch.Tensor) -> Any:
        """Executes a forward pass on input data

        Args:
            x (torch.Tensor): Input with shape [batch_size n_channels width height],
                where n_channels == 3

        Returns:
            Any: predictions shape [batch_size n_classes]
        """
        hidden = self.backbone(x)
        return hidden


# A quick test of the model
if __name__ == "__main__":
    # Create the model
    model = PlantClassifier()

    # Generate a batch of test images
    test_image = torch.rand([64, 3, 300, 255])

    # Compute the result
    out = model(test_image)
    print(out.shape)
