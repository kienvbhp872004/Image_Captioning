from torch import nn
from Decoder import Decoder
from Encoder import Encoder


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, device='cpu'):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size, self.encoder.dim, device=device)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs, alphas = self.decoder(features, captions)
        return outputs, alphas