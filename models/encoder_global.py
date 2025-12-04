import torch
import torch.nn as nn
import torchvision

class GlobalEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, embed_size)

    def forward(self, image):
        with torch.no_grad():
            feat = self.resnet(image).squeeze()
        return self.fc(feat)
