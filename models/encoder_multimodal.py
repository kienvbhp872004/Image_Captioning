import torch
import torch.nn as nn

class MultiModalEncoder(nn.Module):
    def __init__(self, embed_size, tag_vocab_size):
        super().__init__()
        from .encoder_global import GlobalEncoder
        from .encoder_region import RegionEncoder

        self.global_encoder = GlobalEncoder(embed_size)
        self.region_encoder = RegionEncoder(embed_size)
        self.tag_embed = nn.Embedding(tag_vocab_size, embed_size)

    def forward(self, image):
        global_feat = self.global_encoder(image).unsqueeze(1)

        region_feats, boxes, labels = self.region_encoder(image)

        tag_feats = self.tag_embed(labels)

        # concat all: [global | region | tags]
        memory = torch.cat([global_feat, region_feats, tag_feats], dim=0)

        return memory
