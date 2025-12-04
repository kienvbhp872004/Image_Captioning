import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import FlickrDataset
from model import *
import tqdm
img_root = 'dataset/Images'
caption_root = 'dataset/captions'
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 25
    hidden_size = 512
    embedding_dim = 256
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    def collate_fn(batch):
        images = []
        captions = []
        for img, caption in batch:
            images.append(img)
            while len(caption) < vocab_size:
                pad_tensor = torch.zeros(vocab_size - len(caption)).long()
                torch.cat((caption, pad_tensor), dim=0)
            captions.append(caption)
        return images, captions
    dataset = FlickrDataset(img_root, caption_root, transform=transform)
    model = CNNtoRNN(vocab_size,hidden_size,embedding_dim,1)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    loss = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    epochs  = 10
    for epoch in range(epochs):
        total_loss = 0
        for img,captions in tqdm.tqdm(dataloader):
            img = img.to(device)
            captions = captions.to(device)
            lengths = [len(cap) - 1 for cap in captions]
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True, enforce_sorted=False)[0]
            optimizer.zero_grad()
            outputs = model(img,captions)
            loss = loss(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}  Total Loss: {total_loss}")
train()


