from collections import Counter
import nltk
nltk.download('punkt')
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocab(self, sentences):
        frequencies = Counter()
        idx = 4

        for sentence in sentences:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        if isinstance(text, str):
            tokens = self.tokenize(text)
        else:
            tokens = text
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]
