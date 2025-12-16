import pickle
import re
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    def __len__(self):
        return len(self.word2idx)
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    def stoi(self, word):
        return self.word2idx[word]

    def tokenize(self, text):
        """Tokenize text thành words"""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()

    def numericalize(self, text):
        """Chuyển text thành sequence of indices"""
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))
