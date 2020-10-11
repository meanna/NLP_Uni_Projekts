import sys
from collections import defaultdict


def read_data(corpus):
    """
    generate a word-frequency dictionary from a given corpus
    """
    freq = defaultdict(int)
    with open(corpus, "r") as f:
        for line in f:
            items = line.split()
            word = items[0]
            tag = items[1]
            # take only words with tag NN and consist of more than 3 alphabets
            if tag == "NN" and len(word) > 3 and word.isalpha():
                word = word.lower()
                freq[word] += 1
    return freq

def split_compound(word, result, word_list):
    """
    compute all possible partitionings of a given word
    result: a list storing partial result
    word_list: the list of words from the corpus
    """

    # an element of a partitioning should have at lease 3 characters
    if len(word) < 3:
        yield
    else:
        start = 0
        end = 3
        last_index = len(word)
        # if the word contain "s", also check if it is a "Fugen-s"
        if word[0] == "s" and result != []:
            yield from split_compound(word[1:], result, word_list)
        # iterate over the characters until finding a match
        while end <= last_index:
            subword = word[start:end]
            # if successfully parse the word, yield the partitioning result, otherwise yield None
            if end == last_index:
                if subword in word_list:
                    result.append(subword)
                    yield result
                    break
                else:
                    yield
                    break
            else:
                # if a match is found while not having reached the end of the word, then call the function recursively
                temp = result[:]
                if subword in word_list:
                    result.append(subword)
                    yield from split_compound(word[end:], result, word_list)
                    end += 1
                    result = temp
                else:
                    end += 1


def cal_score(elements):
    """
    compute the geometric mean of a given partitioning
    """
    n = len(elements)
    product = 1
    for elem in elements:
        product *= freq[elem]
    score = product ** (1 / n)

    return score

if __name__ == "__main__":
    corpus = sys.argv[1]
    freq = read_data(corpus)
    word_list = sorted(freq.keys()) # word_list is sorted alphabetically
    for word in word_list:
        d = {}
        partitions = split_compound(word, [], word_list)
        for partition in partitions:
            # filter out words that are not compound
            if partition is not None and partition[0] != word:
                partition = tuple(partition)
                d[partition] = cal_score(partition)
                d[tuple([word])] = cal_score(tuple([word]))
        sorted_partitions = sorted(d.items(), key=lambda x: x[1], reverse=True)
        for partition, score in sorted_partitions:
            partition = [i.capitalize() for i in partition]
            print(word.capitalize(), round(score, 1), (*partition), sep="  ")
