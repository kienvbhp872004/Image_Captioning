import re
from collections import Counter

class CaptionProcessor:
    def __init__(self, captions, max_vocab_size=10000, pad_token="<pad>", start_token="<start>", end_token="<end>"):
        self.raw_captions = captions
        self.max_vocab_size = max_vocab_size
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token

        self.processed_captions = [self.preprocess_caption(c) for c in captions]
        self.max_len = max(len(c.split()) for c in self.processed_captions)

        self.word2idx, self.idx2word = self.build_vocab()
        self.vectorized_captions = [self.caption_to_indices(c) for c in self.processed_captions]

    def preprocess_caption(self, caption):
        caption = caption.lower()
        caption = re.sub(r"[^a-z0-9\s]", "", caption)
        caption = re.sub(r"\s+", " ", caption).strip()
        return f"{self.start_token} {caption} {self.end_token}"

    def build_vocab(self):
        word_counts = Counter()
        for caption in self.processed_captions:
            word_counts.update(caption.split())

        most_common_words = [word for word, _ in word_counts.most_common(self.max_vocab_size - 1)]
        vocab = [self.pad_token] + most_common_words

        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        return word2idx, idx2word

    def caption_to_indices(self, caption):
        tokens = caption.split()
        indices = [self.word2idx.get(token, self.word2idx[self.pad_token]) for token in tokens]

        if len(indices) < self.max_len:
            indices += [self.word2idx[self.pad_token]] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return indices

    def get_vectorized_captions(self):
        return self.vectorized_captions

    def get_vocab(self):
        return self.word2idx, self.idx2word

    def get_max_len(self):
        return self.max_len
