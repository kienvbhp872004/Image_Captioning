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
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Output: [B, 2048, 7, 7]

        # Linear projection để giảm dimension
        self.linear = nn.Linear(2048, embed_size)
        self.dropout = nn.Dropout(0.5)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, image):
        with torch.no_grad():
            feat = self.resnet(image).squeeze()
        features = self.adaptive_pool(feat)  # [B, 2048, 7, 7]
        batch_size = features.size(0)
        features = features.permute(0, 2, 3, 1)  # [B, 7, 7, 2048]
        features = features.view(batch_size, -1, 2048)  # [B, 49, 2048]

        # Project to embed_size
        features = self.dropout(self.linear(features))  # [B, 49, embed_size]

        return features

        return self.fc(feat)
