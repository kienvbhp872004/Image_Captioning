import torch.nn as nn
import torch
import os
from .encoder_global import  GlobalEncoder
from .decoder_transformer import TransformerDecoder
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8,
                 num_layers=6, forward_expansion=4, dropout=0.1, max_len=50):
        super(CaptionModel, self).__init__()

        self.encoder =GlobalEncoder (embed_size=embed_size)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            num_layers=num_layers,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_len=max_len
        )

    def forward(self, images, captions, caption_mask=None):
        """
        Args:
            images: [batch_size, 3, 224, 224]
            captions: [batch_size, seq_len]
            caption_mask: Optional padding mask
        Returns:
            outputs: [batch_size, seq_len, vocab_size]
        """
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, captions, caption_mask)
        return outputs

    def generate_caption(self, image, vocab, max_len=50, device='cuda'):
        """
        Greedy decoding để generate caption
        Args:
            image: [1, 3, 224, 224]
            vocab: Vocabulary object
            max_len: Độ dài tối đa
            device: Device
        Returns:
            caption: String caption
        """
        self.eval()
        with torch.no_grad():
            # Encode image
            encoder_out = self.encoder(image)

            # Start with <SOS> token
            caption_ids = [vocab.word2idx["<start>"]]

            for _ in range(max_len):
                captions = torch.LongTensor([caption_ids]).to(device)

                # Decode
                outputs = self.decoder(encoder_out, captions)

                # Get prediction for last token
                predicted = outputs[0, -1, :].argmax().item()
                caption_ids.append(predicted)

                # Stop if <EOS> is generated
                if predicted == vocab.word2idx["<end>"]:
                    break

            # Convert ids to words
            caption_words = [vocab.idx2word[idx] for idx in caption_ids[1:-1]]  # Skip <SOS> and <EOS>
            caption = ' '.join(caption_words)
        print(caption)
        return caption

    def save(self, save_dir = "save/model", epoch = 0):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"Model_epoch_{epoch}.pth")
        torch.save(self.state_dict(), path)

