import torch
import timm

class PlantClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b1.ra4_e3600_r240_in1k', pretrained=True, num_classes=2)

    def forward(self, x):
        hidden = self.backbone(x)
        return hidden

# A quick test of the model
if __name__ == '__main__':
    # Create the model
    model = PlantClassifier()

    # Generate a batch of test images
    test_image = torch.rand([64, 3, 300, 255])

    # Compute the result
    out = model(test_image)
    print(out.shape)