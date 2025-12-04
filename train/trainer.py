import torch
import torch.nn as nn
class Trainer:
    def __init__(self, model, dataloader, cfg):
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        from tqdm import tqdm

        for images, captions in tqdm(self.dataloader, desc=f"Epoch {epoch}"):
            images = images.to(self.cfg.device)
            captions = captions.to(self.cfg.device)

            outputs = self.model(images, captions[:, :-1])
            loss = self.criterion(outputs.reshape(-1, self.cfg.vocab_size), captions[:, 1:].reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.dataloader)

