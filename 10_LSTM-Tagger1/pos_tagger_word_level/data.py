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
        self.word_id = param["word_id"]
        self.id_tag = param["id_tag"]

    def store_parameters(self, parfile):
        param = {}
        param["word_id"] = self.word_id
        param["id_tag"] = self.id_tag
        json.dump(param, open(parfile, "w"))

    def init_train(self, trainset, devset, num_words):

        self.word_freq = defaultdict(int)
        self.word_list = []
        self.tag_list = set()
        self.train_sentences = self.read_data(trainset, train=True)
        sorted_word_tag = sorted(self.word_freq, key=self.word_freq.get, reverse=True)
        self.word_id = {w: i for i, w in enumerate(sorted_word_tag[:num_words], 1)}
        self.num_words = len(self.word_list)
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
                        self.word_freq[word] += 1
                        self.tag_list.add(tag)
                    sentence.append(word)
                    tags.append(tag)
        return sentences

    def words2ids(self, words):
        return [self.word_id.get(word, 0) for word in words]

    def tags2ids(self, tags):
        return [self.tag_id.get(tag, 0) for tag in tags]

    def ids2tags(self, ids):
        return [self.id_tag.get(id, '<unk>') for id in ids]


def run_test():
    data = Data("train.txt", "dev100.txt", 100)
    word_ids = data.words2ids("der Bär isst Bambus".split())
    assert len(word_ids) == 4
    tag_ids = data.tags2ids(['VAFIN.3.Pl.Pres.Ind', 'KON', 'NN.Dat.Pl.Neut', 'PIAT.Nom.Pl.Masc'])
    assert len(tag_ids) == 4
    tags = data.ids2tags(tag_ids)
    assert tags == ['VAFIN.3.Pl.Pres.Ind', 'KON', 'NN.Dat.Pl.Neut', 'PIAT.Nom.Pl.Masc']

    g = data.sentences("testdata.txt")
    print(*g)
    data.store_parameters("param")
    data = Data("param.json")  # test init_test()


if __name__ == "__main__":
    run_test()
