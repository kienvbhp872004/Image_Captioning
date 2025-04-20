import os
from PIL import Image
from tqdm import tqdm

class ImageCaptionDatasetLoader:
    def __init__(self, caption_file, image_dir, verbose=True):
        self.caption_file = caption_file
        self.image_dir = image_dir
        self.images = []
        self.captions = []
        self.verbose = verbose

    def load_data(self):
        if self.verbose:
            print("Đang load dữ liệu từ:", self.caption_file)

        with open(self.caption_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Loading images", disable=not self.verbose):
            try:
                filename, caption = line.strip().split(',', 1)
                image_path = os.path.join(self.image_dir, filename)

                img = Image.open(image_path).convert('RGB')
                self.images.append(img)
                self.captions.append(caption)
            except Exception as e:
                print(f"Lỗi load ảnh {image_path}: {e}")

        if self.verbose:
            print(f"Đã load {len(self.images)} ảnh và {len(self.captions)} caption.")

    def get_images(self):
        return self.images

    def get_captions(self):
        return self.captions

    def __len__(self):
        return len(self.images)
