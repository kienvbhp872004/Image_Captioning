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

        for images, captions in tqdm(self.dataloader, desc=f"Epoch {epoch}"):
            images = images.to(self.cfg.device)
            captions = captions.to(self.cfg.device)
            decoder_input = captions[:, :-1]
            targets = captions[:, 1:]
            padding_mask  = (captions == self.vocab.word2idx["<pad>"])
            outputs = self.model(images, captions, padding_mask)
            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = captions.reshape(-1)

            loss = self.criterion(outputs,targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(self.dataloader)

