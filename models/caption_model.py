import torch.nn as nn

class CaptionModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image, captions):
        memory = self.encoder(image)
        return self.decoder(memory, captions)
