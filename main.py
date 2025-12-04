from config import Config
from models.encoder_multimodal import MultiModalEncoder
from models.decoder_transformer import TransformerDecoder
from models.caption_model import CaptionModel
from train.trainer import Trainer
from datasets.flickr8k_dataset import Flickr8kDataset, collate_fn
from utils.vocabulary import Vocabulary


from torch.utils.data import DataLoader


def build_vocab(captions_file, min_freq=5):
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")
    vocab.add_word("<unk>")

    word_freq = {}

    with open(captions_file, "r") as f:
        for line in f:
            _, caption = line.strip().split("\t")
            for w in caption.lower().split():
                word_freq[w] = word_freq.get(w, 0) + 1

    for w, c in word_freq.items():
        if c >= min_freq:
            vocab.add_word(w)

    return vocab


def create_dataloader(cfg):
    vocab = build_vocab(cfg.caption_path)
    vocab.save(cfg.vocab_path)

    dataset = Flickr8kDataset(
        image_root=cfg.image_dir,
        captions_file=cfg.caption_path,
        vocab=vocab
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return loader, vocab



def main():
    cfg = Config()

    dataloader, vocab = create_dataloader(cfg)
    cfg.vocab_size = len(vocab)

    encoder = MultiModalEncoder(cfg.embed_size, tag_vocab_size=200)
    decoder = TransformerDecoder(cfg.vocab_size, cfg.embed_size, cfg.num_heads, cfg.num_layers)
    model = CaptionModel(encoder, decoder).to(cfg.device)

    trainer = Trainer(model, dataloader, cfg)

    for epoch in range(cfg.num_epochs):
        loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} - Loss: {loss:.4f}")


if __name__ == "__main__":
    main()