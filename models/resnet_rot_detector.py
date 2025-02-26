import torch.nn as nn
import torchvision


class ResNet(nn.Module):
    def __init__(self, dropout=0.25, weights="IMAGENET1K_V1"):
        super(ResNet, self).__init__()
        self.resnet50 = torchvision.models.resnet50(weights=weights)
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=2048, out_features=4, bias=True),# out_features=8 -> 4
            # Quitar Sigmoid
        )

    def forward(self, x):
        return self.resnet50(x)


if __name__ == "__main__":
    model = ResNet()
    for param in model.parameters():
        print(param.requires_grad)