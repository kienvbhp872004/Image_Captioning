import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import *
from dataset import FlickrDataset

img_root = 'dataset/Images'
caption_root = 'dataset/captions'
def train():

    vocab_size = 25
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
                caption.append(0)
            captions.append(caption)
        return images, captions
    dataset = FlickrDataset(img_root, caption_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    print(1)





