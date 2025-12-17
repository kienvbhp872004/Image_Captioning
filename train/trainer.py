import torch
import torch.nn as nn
class Trainer:
    def __init__(self, model, dataloader, cfg,vocab):
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg
        self.vocab = vocab
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        from tqdm import tqdm

        for i, (images, captions) in enumerate(
                tqdm(self.dataloader, desc=f"Epoch {epoch}")
        ):
            images = images.to(self.cfg.device)
            captions = captions.to(self.cfg.device)

            # SHIFT
            decoder_input = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = self.model(images, decoder_input)

            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # 🔥 DEBUG: mỗi batch in caption của ảnh cuối
            self.model.eval()
            with torch.no_grad():
                last_img = images[-1:].detach()  # shape [1, 3, H, W]
                caption = self.model.generate_caption(
                    last_img, self.vocab, max_len=20
                )
                print(f"[Epoch {epoch} | Batch {i}] {caption}")
            self.model.train()

        return total_loss / len(self.dataloader)




