import argparse

from data import Data

import torch

parser = argparse.ArgumentParser()
parser.add_argument('path_param', type=str)
parser.add_argument('test_data', type=str)
args = parser.parse_args()

data = Data(args.path_param + ".json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(args.path_param + ".rnn")
model = model.to(device)
sentences = data.sentences(args.test_data)

for words in sentences:
    prefixes, suffixes = data.words2ids_vecs(words)
    tag_scores = model.forward(torch.LongTensor(prefixes).to(device), torch.LongTensor(suffixes).to(device))
    preds = torch.argmax(tag_scores, dim=1)
    tag_ids = [str(id) for id in (preds.tolist())]
    tags = data.ids2tags(tag_ids)
    for w, t in zip(words, tags):
        print(w + "\t" + t)
    print()
