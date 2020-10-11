from _collections import defaultdict
import json


class Data:

    def __init__(self, *args):
        if len(args) == 1:
            self.init_test(*args)
        else:
            self.init_train(*args)

    def init_test(self, path_param):
        param = json.load(open(path_param))
        self.char_id = param["char_id"]
        self.id_tag = param["id_tag"]

    def store_parameters(self, parfile):
        param = {}
        param["char_id"] = self.char_id
        param["id_tag"] = self.id_tag
        json.dump(param, open(parfile, "w"))

    def init_train(self, trainset, devset):
        self.char_freq = defaultdict(int)
        self.tag_list = set()
        self.train_sentences = self.read_data(trainset, train=True)        
        self.char_list = [char for char, f  in self.char_freq.items() if f >1]
        self.char_id = {c: i for i, c in enumerate(self.char_list, 1)}
        self.num_chars = len(self.char_list)
        self.tag_list = list(self.tag_list)
        self.tag_id = {t: i for i, t in enumerate(self.tag_list, 1)}
        self.id_tag = {i: t for i, t in enumerate(self.tag_list, 1)}
        self.num_tags = len(self.tag_list)
        self.dev_sentences = self.read_data(devset, train=False)

    def sentences(self, filename):
        sentences = []
        with open(filename) as d:
            sentence = []
            for line in d:
                if line == "\n":
                    yield sentence
                    sentence = []
                else:
                    word = line.strip()
                    sentence.append(word)
            yield sentence

    def read_data(self, dataset, train):
        sentences = []
        with open(dataset) as d:
            sentence = []
            tags = []
            for line in d:
                if line == "\n":
                    sentences.append((sentence, tags))
                    sentence = []
                    tags = []
                else:
                    word, tag = line.split()
                    if train:
                        for char in word:
                            self.char_freq[char] += 1
                        
                        self.tag_list.add(tag)
                    sentence.append(word)
                    tags.append(tag)
        return sentences

    def words2ids_vecs(self, words):
        l = 10
        suffix_list = []
        prefix_list = []
        for word in words:
            if len(word)< l:
                fill = l-len(word)
                prefix = word + (" " *fill)
                suffix = (" " *fill) + word
            else:
                prefix = word[:l]
                suffix = word[len(word)-l:]
            prefix = prefix[::-1] #reverse prefix sequence
            prefix_id = [self.char_id.get(p, 0) for p in prefix]          
            suffix_id = [self.char_id.get(s, 0) for s in suffix]
            suffix_list.append(suffix_id)
            prefix_list.append(prefix_id)

        return prefix_list, suffix_list
        
    def tags2ids(self, tags):
        return [self.tag_id.get(tag, 0) for tag in tags]

    def ids2tags(self, ids):
        return [self.id_tag.get(id, '<unk>') for id in ids]


def run_test():

    data = Data("train.tagged", "dev.tagged")
    suff, pref = data.words2ids_vecs("miwako seehundstationen".split())
    assert len(suff) == 2
    assert len(suff[0]) == 10
    assert len(pref) == 2
    assert len(pref[0]) == 10
    assert suff[0][:4] == [0, 0, 0, 0]
    tag_ids = data.tags2ids(['VAFIN.3.Pl.Pres.Ind', 'KON', 'NN.Dat.Pl.Neut', 'PIAT.Nom.Pl.Masc'])
    assert len(tag_ids) == 4
    tags = data.ids2tags(tag_ids)

    data.store_parameters("param")
    data = Data("param.json")  # test init_test()


if __name__ == "__main__":
    run_test()
