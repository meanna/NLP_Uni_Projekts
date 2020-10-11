from sys import argv
from collections import defaultdict


class Parser:
    def __init__(self, grammar, lexicon):
        self.grammar, self.start_symbol = self.read_grammar(grammar)
        self.lexicon = self.read_lexicon(lexicon)

    def read_grammar(self, grammar_file):
        grammar = defaultdict(list)
        start_symbol = None
        with open(grammar_file, "r") as file:
            for line in file:
                left, *(right) = line.split()
                if not start_symbol:
                    start_symbol = left
                grammar[left].append(tuple(right))
        return grammar, start_symbol

    def read_lexicon(self, lexicon_file):
        lexicon = {}
        with open(lexicon_file, "r") as file:
            for line in file:
                word, *tags = line.split()
                lexicon[word] = tags
        return lexicon

    def add(self, item, state):
        self.chart[state].add(item)
        left, rights, dot_pos, start_pos = item
        if dot_pos == len(rights):
            self.complete(left, start_pos, state)

        elif dot_pos < len(rights) and rights[dot_pos] in self.grammar.keys():
            self.predict(rights[dot_pos], state)

        elif state < len(self.tag_list):
            self.scan(item, state)

    def predict(self, symbol, state):
        if tuple((symbol, state)) not in self.predict_set:
            self.predict_set.add(tuple((symbol, state)))
            if self.grammar[symbol]:
                for rights in self.grammar[symbol]:
                    item = (symbol, rights, 0, state)
                    self.add(item, state)

    def scan(self, item, state):
        left, right, dot_pos, ori = item
        for tag in self.tag_list[state]:
            if right[dot_pos] == tag:
                new_item = (left, right, dot_pos + 1, ori)
                self.add(new_item, state + 1)

    def complete(self, symbol, ori, state):
        for item in self.chart[ori]:
            left, rights, dot_pos, ori2 = item
            if dot_pos < len(rights) and rights[dot_pos] == symbol:
                item = (left, rights, dot_pos + 1, ori2)
                self.add(item, state)

    def parse(self, word_list):
        result = False
        self.predict_set = set()
        try:
            self.tag_list = [self.lexicon[word] for word in word_list]
        except KeyError:
            print("word in not in the lexicon")
            return result
        end = len(word_list)
        self.chart = [set() for _ in range(end + 1)]
        # start with a start rule
        for right in self.grammar[self.start_symbol]:
            start = (self.start_symbol, right, 0, 0)
            self.add(start, 0)
            finish_item = (self.start_symbol, right, len(right), 0)
            if finish_item in self.chart[end]:
                result = True
            else: # if the current start rule failed, try the next start rule
                self.predict_set = set()
        return result


if __name__ == '__main__':
    grammar = argv[1]
    lexicon = argv[2]
    sentences = argv[3]
    early = Parser(grammar, lexicon)
    with open(sentences, "r") as file:
        for line in file:
            if line != "":
                word_list = line.split()
                result = early.parse(word_list)
                print(word_list, result)
                print("---------------------------------")
