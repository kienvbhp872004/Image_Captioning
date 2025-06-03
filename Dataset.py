import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from Vocab import Vocabulary


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=3):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.freq_threshold = freq_threshold
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.df["caption"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.imgs[idx])
        image = Image.open(img_path)
        if self.transform is not None:
            image = self.transform(image)
        caption = self.captions[idx]
        numericalized = [self.vocab.stoi["<SOS>"]] + self.vocab.numericalize(caption) + [self.vocab.stoi["<EOS>"]]
        return image, torch.tensor(numericalized, dtype=torch.long), len(numericalized),caption
