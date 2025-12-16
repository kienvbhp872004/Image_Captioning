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

    def __init__(self, image_root, captions_file, vocab=None, transform=None,max_len = 50):
        self.image_root = image_root
        self.transform = transform or get_transforms()
        self.vocab = vocab
        self.max_len = max_len

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

        numerical = [self.vocab.word2idx["<start>"]]
        numerical += self.vocab.numericalize(caption)
        numerical.append(self.vocab.word2idx["<end>"])

        if len(numerical) < self.max_len:
            numerical += [self.vocab.word2idx["<pad>"]] * (self.max_len - len(numerical))
        else:
            numerical = numerical[:self.max_len]

        return image, torch.tensor(numerical)



