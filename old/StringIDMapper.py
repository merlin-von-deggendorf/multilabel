class StringIDMapper:
    def __init__(self):
        self.voc2id = {}
        self.id2voc = {}
    def add_word(self, word):
        if word not in self.voc2id:
            idx = len(self.voc2id)
            self.voc2id[word] = idx
            self.id2voc[idx] = word
    def str2id(self, word):
        return self.voc2id.get(word, -1)

    def id2str(self, idx):
        return self.id2voc.get(idx, None)
    def add_words(self, words:set[str]):
        for word in words:
            self.add_word(word)
    def __len__(self):
        return len(self.voc2id)