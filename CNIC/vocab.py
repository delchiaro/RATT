from typing import List
from bidict import bidict


class Vocabulary(object):
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2
    UNK_IDX = 3
    PAD = '<pad>'
    START = '<start>'
    END = '<end>'
    UNK = '<unk>'

    """Simple vocabulary wrapper."""
    def __init__(self):
        self.wordmap = bidict()
        self.wordmap[Vocabulary.PAD] = Vocabulary.PAD_IDX
        self.wordmap[Vocabulary.START] = Vocabulary.START_IDX
        self.wordmap[Vocabulary.END] = Vocabulary.END_IDX
        self.wordmap[Vocabulary.UNK] = Vocabulary.UNK_IDX
        self._idx = 4

    def add_word(self, word):
        if not word in self.wordmap:
            self.wordmap[word] = self._idx
            self._idx += 1

    def __call__(self, w):
        return self.word2idx(w) if isinstance(w, str) else self.idx2word(w)

    def __len__(self):
        return len(self.wordmap)

    def word2idx(self, word: str):
        if not word in self.wordmap:
            return self.wordmap['<unk>']
        return self.wordmap[word]

    def idx2word(self, idx: int):
        return self.wordmap.inv[idx]

    def translate(self, ids: List[int]):
        return ' '.join([self.idx2word(id) for id in ids])

    def translate_split(self, ids: List[int]):
        return [self.idx2word(id) for id in ids]

    def translate_all(self, sentences_ids: List[List[int]]):
        return [self.translate(ids) for ids in sentences_ids]

    def translate_all_split(self, sentences_ids: List[List[int]]):
        return [self.translate_split(ids) for ids in sentences_ids]

    def remap_vocab_word_ids(self, new_vocab: 'Vocabulary'):
        old2new_ids = []
        for word, idx in new_vocab.wordmap.items():
            full_idx = self.word2idx(word)  # will return <unknown> idx if not in vocab
            old2new_ids.append(full_idx)
        return old2new_ids

    @property
    def words(self):
        return list(self.wordmap.keys())
