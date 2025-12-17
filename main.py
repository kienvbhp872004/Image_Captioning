from config import Config
from models.caption_model import CaptionModel
from datasets.flickr8k_dataset import Flickr8kDataset
from utils.vocabulary import Vocabulary
from train.trainer import Trainer
from torch.utils.data import DataLoader
import os

def build_vocab(captions_file, min_freq=5):
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    word_freq = {}

    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

        # Bỏ dòng đầu (header: image,caption)
        lines = lines[1:]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Chia bằng dấu phẩy đầu tiên
            filename, caption = line.split(",", 1)

            caption = caption.strip().lower()

            # Đếm tần suất từ
            for w in caption.split():
                word_freq[w] = word_freq.get(w, 0) + 1

    # Thêm từ vào vocab
    for w, c in word_freq.items():
        if c >= min_freq:
            vocab.add_word(w)

    return vocab



def create_dataloader(cfg,is_kaggle=False):
    path_dir = ""
    if is_kaggle:
        path_dir ="/kaggle/input/flickr8k"
    path_image = os.path.join(path_dir, cfg.image_dir)
    path_caption = os.path.join(path_dir, cfg.caption_path)
    vocab = build_vocab(path_caption)
    vocab.save(cfg.vocab_path)

    dataset = Flickr8kDataset(
        image_root=path_image,
        captions_file=path_caption,
        vocab=vocab,
    )
    print(dataset.__len__())
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    return loader, vocab


def main():
    cfg = Config()

    dataloader, vocab = create_dataloader(cfg)
    cfg.vocab_size = len(vocab)

    model = CaptionModel(cfg.vocab_size,cfg.embed_size,cfg.num_heads,cfg.num_layers).to(cfg.device)
    trainer = Trainer(model, dataloader,cfg,vocab)

    for epoch in range(cfg.num_epochs):
        loss = trainer.train_epoch(epoch)
        model.save("saved/model",epoch=epoch)
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

if __name__ == "__main__":
    main()
