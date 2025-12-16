import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8,
                 num_layers=6, forward_expansion=4, dropout=0.1, max_len=50):
        """
        Transformer Decoder cho Image Captioning
        Args:
            vocab_size: Kích thước vocabulary
            embed_size: Kích thước embedding
            num_heads: Số attention heads
            num_layers: Số decoder layers
            forward_expansion: Expansion factor cho feedforward network
            dropout: Dropout rate
            max_len: Độ dài tối đa của sequence
        """
        super(TransformerDecoder, self).__init__()

        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len)

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        """Tạo causal mask để ngăn attention nhìn vào future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, encoder_out, captions, caption_mask=None):
        """
        Args:
            encoder_out: [batch_size, num_pixels, embed_size] từ encoder
            captions: [batch_size, seq_len] input captions
            caption_mask: Optional padding mask cho captions
        Returns:
            outputs: [batch_size, seq_len, vocab_size]
        """
        seq_len = captions.size(1)

        # Embedding và positional encoding cho captions
        embeddings = self.word_embedding(captions)  # [B, seq_len, embed_size]
        embeddings = self.position_encoding(embeddings)
        embeddings = self.dropout(embeddings)

        # Tạo causal mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(captions.device)

        # Transformer decoder
        # memory: encoder output, tgt: caption embeddings
        decoder_out = self.transformer_decoder(
            tgt=embeddings,
            memory=encoder_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=caption_mask
        )

        # Project to vocabulary
        outputs = self.fc_out(decoder_out)  # [B, seq_len, vocab_size]

        return outputs
