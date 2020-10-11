from sys import argv
from collections import defaultdict


class NoiseModel:
    def __init__(self, src_word_list, tgt_word_list):
        self.src_letter_prob = dict()
        self.tgt_letter_prob = dict()

        src_letter_freq = defaultdict(int)
        tgt_letter_freq = defaultdict(int)
        for c in "".join(src_word_list):
            src_letter_freq[c] += 1
        for c in "".join(tgt_word_list):
            tgt_letter_freq[c] += 1
        src_letter_total = sum(src_letter_freq.values())
        tgt_letter_total = sum(tgt_letter_freq.values())
        for c in set("".join(src_word_list)):
            self.src_letter_prob[c] = src_letter_freq[c] / src_letter_total
        for c in set("".join(tgt_word_list)):
            self.tgt_letter_prob[c] = tgt_letter_freq[c] / tgt_letter_total

    def word_pair_prob(self, src_word, tgt_word):
        src_word_prob, tgt_word_prob = 1, 1
        for c in src_word:
            src_word_prob *= self.src_letter_prob[c]

        for c in tgt_word:
            tgt_word_prob *= self.tgt_letter_prob[c]

        return src_word_prob * tgt_word_prob


class TransliterationModel:

    def __init__(self, N, M):
        self.N = N
        self.M = M
        uniform = 1 / ((N * M) - 1)
        self.trans_unit_prob = defaultdict(lambda: uniform)
        self.trans_unit_freq = defaultdict(float)

    def add_freq(self, src_letter, tgt_letter, freq):
        self.trans_unit_freq[(src_letter, tgt_letter)] += freq

    def reestimate(self):
        total_freq = sum(self.trans_unit_freq.values())
        for char_pair, freq in self.trans_unit_freq.items():
            new_prob = freq / total_freq if freq / total_freq > 1e-300 else 1e-300
            self.trans_unit_prob[char_pair] = new_prob
        self.trans_unit_freq = defaultdict(float)


class MiningModel:

    def __init__(self, src_words, tgt_words, num_iterations):
        self.noise_model = NoiseModel(src_words, tgt_words)
        N = len(self.noise_model.src_letter_prob) + 1
        M = len(self.noise_model.tgt_letter_prob) + 1
        self.trans_model = TransliterationModel(N, M)
        self.model_prior = 0.5  # lambda 
        self.num_word_pairs = len(src_words)

        for _ in range(num_iterations):
            self.model_freq = 0  # sum of p(trans|x,y)
            for x, y in zip(src_words, tgt_words):
                self.estimate_freqs(x, y)
            self.reestimate_probs()

    def forward(self, src_word, tgt_word):
        f = [[None for _ in range(len(tgt_word))] for _ in range(len(src_word))]

        for i in range(len(src_word)):
            for k in range(len(tgt_word)):
                if i == 0 and k == 0:
                    f[i][k] = 1.0

                # if i==0, components that contain f[i - 1][..] become 0, so they get canceled out
                elif i == 0:
                    f[i][k] = f[i][k - 1] * self.trans_model.trans_unit_prob[("", tgt_word[k - 1])]

                # if k == 0, components that contain f[..][k-1] become 0, so they get canceled out
                elif k == 0:
                    f[i][k] = f[i - 1][k] * self.trans_model.trans_unit_prob[(src_word[i - 1], "")]

                else:
                    f[i][k] = f[i - 1][k] * self.trans_model.trans_unit_prob[(src_word[i - 1], "")] \
                              + f[i][k - 1] * self.trans_model.trans_unit_prob[("", tgt_word[k - 1])] \
                              + f[i - 1][k - 1] * self.trans_model.trans_unit_prob[
                                  (src_word[i - 1], tgt_word[k - 1])]
        return f

    def backward(self, src_word, tgt_word):

        b = [[None for _ in range(len(tgt_word))] for _ in range(len(src_word))]
        n = len(src_word) - 1
        m = len(tgt_word) - 1

        for i in range(n, -1, -1):
            for k in range(m, -1, -1):
                if i == n and k == m:
                    b[i][k] = 1.0

                # if i == n, components that contain b[i + 1][..] become 0, so they get canceled out
                elif i == n:
                    b[i][k] = b[i][k + 1] * self.trans_model.trans_unit_prob[("", tgt_word[k])]

                # if k == m, components that contain b[..][k+1] become 0, so they get canceled out
                elif k == m:
                    b[i][k] = b[i + 1][k] * self.trans_model.trans_unit_prob[(src_word[i], "")]

                else:
                    b[i][k] = b[i + 1][k] * self.trans_model.trans_unit_prob[(src_word[i], "")] \
                              + b[i][k + 1] * self.trans_model.trans_unit_prob[("", tgt_word[k])] \
                              + b[i + 1][k + 1] * self.trans_model.trans_unit_prob[(src_word[i], tgt_word[k])]

        return b

    def estimate_freqs(self, src_word, tgt_word):

        forward = self.forward(src_word, tgt_word)
        backward = self.backward(src_word, tgt_word)
        n = len(src_word) - 1
        m = len(tgt_word) - 1
        # p_trans refers to p_trans(x,y)
        p_trans = forward[n][m] #if forward[n][m] > 1e-300 else 1e-300
        p_noise = self.noise_model.word_pair_prob(src_word, tgt_word)
        p_mining = (self.model_prior * p_trans) + ((1 - self.model_prior) * p_noise)
        # p_trans_apos refers to p(trans|x,y)
        p_trans_apos = self.model_prior * p_trans / p_mining
        self.model_freq += p_trans_apos  # accumulate expected count of transliterations

        for i in range(len(src_word)):
            src_letter = src_word[i]
            for k in range(len(tgt_word)):
                tgt_letter = tgt_word[k]

                # compute gamma for x_i:y_k
                p = self.trans_model.trans_unit_prob[(src_letter, tgt_letter)]
                try:
                    b = backward[i + 1][k + 1]
                except:
                    b = 0
                gamma = p_trans_apos * forward[i][k] * p * b / p_trans
                self.trans_model.add_freq(src_letter, tgt_letter, gamma)

                # compute gamma for x_i:_
                p = self.trans_model.trans_unit_prob[(src_letter, "")]
                try:
                    b = backward[i + 1][k]
                except:
                    b = 0
                gamma_delete = forward[i][k] * p * b
                self.trans_model.add_freq(src_letter, "", gamma_delete)

                # compute gamma for _:y_k
                p = self.trans_model.trans_unit_prob[("", tgt_letter)]
                try:
                    b = backward[i][k + 1]
                except:
                    b = 0
                gamma_insert = forward[i][k] * p * b
                self.trans_model.add_freq("", tgt_letter, gamma_insert)

    def reestimate_probs(self):
        self.model_prior = self.model_freq / self.num_word_pairs
        self.trans_model.reestimate()

    def print_transliterations(self, src_words, tgt_words):

        for src_word, tgt_word in zip(src_words, tgt_words):
            n = len(src_word) - 1
            m = len(tgt_word) - 1
            forward = self.forward(src_word, tgt_word)
            p_trans = forward[n][m]
            p_noise = self.noise_model.word_pair_prob(src_word, tgt_word)
            p_mining = (self.model_prior * p_trans) + ((1 - self.model_prior) * p_noise)
            p_trans_apos = self.model_prior * p_trans / p_mining

            if p_trans_apos > 0.5:
                print(src_word, tgt_word)


if __name__ == "__main__":
    data = argv[1]
    src_words = []
    tgt_words = []
    with open(data) as f:
        for line in f:
            src_word, tgt_word = line.split()
            src_words.append(src_word)
            tgt_words.append(tgt_word)
    num_iterations = 3
    model = MiningModel(src_words, tgt_words, num_iterations)
    model.print_transliterations(src_words, tgt_words)
