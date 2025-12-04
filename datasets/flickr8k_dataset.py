import os
from PIL import Image
from torch.utils.data import Dataset
from transforms import get_transforms
import torch

class Flickr8kDataset(Dataset):
    def __init__(self, image_root, captions_file, vocab, transform=None):
        self.image_root = image_root
        self.transform = transform or get_transforms()
        self.vocab = vocab

        # Parse captions file
        self.data = []
        with open(captions_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_id, caption = parts
                img_filename = img_id.split("#")[0]
                caption = caption.lower().strip()
                self.data.append((img_filename, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename, caption = self.data[idx]

        # Load image
        img_path = os.path.join(self.image_root, img_filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Convert caption to tokens
        tokens = ["<start>"] + caption.split(" ") + ["<end>"]
        caption_ids = [self.vocab.word2idx.get(w, self.vocab.word2idx["<unk>"]) for w in tokens]

        return image, caption_ids
def collate_fn(batch):
    images, captions = zip(*batch)

    images = torch.stack(images)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len).long()

    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = torch.tensor(cap)

    return images, padded, torch.tensor(lengths)