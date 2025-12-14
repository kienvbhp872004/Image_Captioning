# datasets/flickr8k_dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from .transforms import get_transforms
import torch

class Flickr8kDataset(Dataset):
    """
    Dataset cho Flickr8k captions.
    - Nếu feature_root được cung cấp, dataset sẽ trả (resnet_feat, region_feat, caption_ids, img_filename)
      thay vì (image_tensor, caption_ids, img_filename).
    - Nếu feature_root không cung cấp, dataset trả image tensor như trước.
    """

    def __init__(self, image_root, captions_file, vocab=None, transform=None):
        self.image_root = image_root
        self.transform = transform or get_transforms()
        self.vocab = vocab

        self.data = []
        with open(captions_file, "r", encoding="utf-8") as f:
            # cố gắng bỏ header nếu có
            first = f.readline()
            # nếu first không phải header (không chứa 'image' hoặc 'caption') ta xử lý lại
            if "image" in first.lower() and "caption" in first.lower():
                lines = f.readlines()
            else:
                # first là dòng dữ liệu
                lines = [first] + f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Hỗ trợ file với tab hoặc comma
            if "\t" in line:
                img_id, caption = line.split("\t", 1)
            else:
                parts = line.split(",", 1)
                if len(parts) < 2:
                    continue
                img_id, caption = parts

            img_filename = img_id.split("#")[0]
            caption = caption.lower().strip()

            self.data.append((img_filename, caption))

        # --- NEW: danh sách đường dẫn file ảnh đầy đủ (theo thứ tự self.data) ---
        # lưu dạng đường dẫn đầy đủ để tiện extract feature
        self.image_paths = [os.path.join(self.image_root, img_fn) for img_fn, _ in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename, caption = self.data[idx]
        # else: load image and transform (original behavior)
        img_path = os.path.join(self.image_root, img_filename)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = ["<start>"] + caption.split(" ") + ["<end>"]
        caption_ids = [
            self.vocab.word2idx.get(w, self.vocab.word2idx["<unk>"])
            for w in tokens
        ]

        # trả kèm img_filename (dùng làm key khi cache)
        return image, caption_ids


def collate_fn(batch):
    """
    Collate cho batch khi dataset trả image hoặc precomputed features.
    - Hỗ trợ 2 dạng batch element:
      a) (image_tensor, caption_ids, img_filename)
      b) (res_feat, region_feat, caption_ids, img_filename)
    """
    # detect format by length of element
    if len(batch[0]) == 2:
        # format a)
        images, captions = zip(*batch)
        images = torch.stack(images)

        lengths = [len(c) for c in captions]
        max_len = max(lengths)

        padded = torch.zeros(len(captions), max_len).long()
        for i, cap in enumerate(captions):
            padded[i, :len(cap)] = torch.tensor(cap)

        return images, padded

    else:
        raise ValueError(f"Expected batch of length 2, but got {len(batch[0])}")
