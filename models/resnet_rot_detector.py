import torch
from tqdm import tqdm
import torch.nn.functional as F
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

def predict(model, dataloader, device):
    results = []
    processed_names = set()  # Usar un conjunto para manejar nombres únicos procesados
    with torch.no_grad():
        for images, names in tqdm(dataloader,desc="Processing images"):
            images = images.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
            for name, prediction, confidence in zip(names, predictions.cpu().numpy(), confidences.cpu().numpy()):
                
                if name not in processed_names:  # Solo añadir si no es duplicado
                    processed_names.add(name)
                    results.append((name, prediction, confidence))
                else:
                    print(f"Duplicate image name found and skipped: {name}")
    return results


if __name__ == "__main__":
    model = ResNet()
    for param in model.parameters():
        print(param.requires_grad)