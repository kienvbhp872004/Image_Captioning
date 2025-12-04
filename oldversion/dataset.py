import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from vocab import Vocabulary
import nltk
nltk.data.path = ['C:\\Users\\admin\\nltk_data']

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=3):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.freq_threshold = freq_threshold

        # Lưu danh sách ảnh và caption
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.df["caption"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load ảnh khi cần
        img_path = os.path.join(self.root_dir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        caption = self.captions[idx]
        numericalized = [self.vocab.stoi["<SOS>"]]
        numericalized += self.vocab.numericalize(caption)
        numericalized.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized)
