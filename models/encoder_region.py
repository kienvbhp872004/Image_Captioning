import torch
import torch.nn as nn
import torchvision

class RegionEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.fc = nn.Linear(2048, embed_size)

    def forward(self, image, max_regions=36):
        with torch.no_grad():
            output = self.detector([image])[0]

        # Top-k regions
        scores = output["scores"]
        keep = scores.topk(min(max_regions, len(scores))).indices

        # region features (custom: using pooled features)
        region_feats = output["features"][keep]    # (N, 2048)
        boxes = output["boxes"][keep]              # (N, 4)
        labels = output["labels"][keep]            # (N)

        region_feats = self.fc(region_feats)

        return region_feats, boxes, labels
