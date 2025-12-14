import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super().__init__()

        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(500, embed_size)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=2048,
            batch_first=True
        )

        self.transformer = nn.TransformerDecoder(layer, num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, memory, captions):
        """
        memory: Tensor (N, S, E)  -- batch-first encoder outputs (padded)
        captions: Tensor (N, T)   -- token ids, batch-first
        returns: (N, T, vocab_size)
        """
        seq_len = captions.size(1)
        positions = torch.arange(seq_len, device=captions.device, dtype=torch.long).unsqueeze(0)  # (1, T)

        tgt = self.word_embed(captions) + self.pos_embed(positions)  # (N, T, E) broadcast pos to batch
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(captions.device)  # (T, T)

        # --- sửa: truyền memory trực tiếp (đã là (N, S, E)) ---
        out = self.transformer(tgt, memory, tgt_mask=mask)

        return self.fc_out(out)  # (N, T, vocab_size)
