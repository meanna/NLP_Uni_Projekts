import sys
from collections import defaultdict


def read_data(corpus):
    """
    read the corpus and generate a dictionary where
    keys are the sorted characters of the anagrams
    and values are lists of anagrams sorted alphabetically
    """
    anagrams = defaultdict(set)
    with open(corpus, "r") as f:
        for line in f:
            items = line.split()
            word = items[0]
            # consider only words that consist of at least 3 characters
            if len(word) >= 3 and word.isalpha():
                sorted_chars = "".join(sorted(word.lower()))
                # in case of "Ihr" and "ihr", take only one of them
                anagrams_lower = [word.lower() for word in anagrams[sorted_chars]]
                if word.lower() not in anagrams_lower:
                    anagrams[sorted_chars].add(word)
        for k, words in anagrams.items():
            anagrams[k] = sorted(words)
    return anagrams


if __name__ == "__main__":
    corpus = sys.argv[1]
    anagrams = read_data(corpus)
    for _, words in sorted(anagrams.items()):
        if len(words) > 1:
            print(*words)
