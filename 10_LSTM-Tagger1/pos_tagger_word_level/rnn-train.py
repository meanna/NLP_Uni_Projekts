import argparse
import random

from lstm_tagger import TaggerModel
from data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#python rnn-train.py Tiger-morph/train.tagged Tiger-morph/dev.tagged param --num_epochs=10 --num_words=20000 --emb_size=300 --rnn_size=300 --dropout_rate=0.5 --learning_rate=0.001

parser = argparse.ArgumentParser()
parser.add_argument('trainfile', type=str)
parser.add_argument('devfile', type=str)
parser.add_argument('parfile', type=str)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_words', type=int)
parser.add_argument('--emb_size', type=int)
parser.add_argument('--rnn_size', type=int)
parser.add_argument('--dropout_rate', type=float)
parser.add_argument('--learning_rate', type=float)
args = parser.parse_args()

data = Data(args.trainfile, args.devfile, args.num_words)
data.store_parameters(args.parfile + ".json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TaggerModel(args.num_words, data.num_tags, args.emb_size, args.rnn_size, args.dropout_rate)
model = model.to(device)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

best_acc = 0
word_count = sum([len(s) for s, _ in data.dev_sentences])
epoch = 0
for _ in range(args.num_epochs):
    random.shuffle(data.train_sentences)
    model.train(True)
    for words, tags in data.train_sentences:
        optimizer.zero_grad()
        word_ids = data.words2ids(words)
        labels = data.tags2ids(tags)
        tag_scores = model.forward(torch.LongTensor(word_ids).to(device))
        loss = loss_function(tag_scores, torch.LongTensor(labels).to(device))
        loss.backward()
        optimizer.step()

    model.train(False)
    corrects = 0
    for words, tags in data.dev_sentences:
        word_ids = data.words2ids(words)
        labels = data.tags2ids(tags)
        tag_scores = model.forward(torch.LongTensor(word_ids).to(device))
        preds = torch.argmax(tag_scores,dim=1)
        
        corrects += sum([1 for p, t in zip(preds.tolist(), labels) if p == t])
    new_acc = corrects / word_count
    epoch += 1
    print("epoch " + str(epoch) + ",  loss " + str(float(loss)))
    if new_acc > best_acc:
        best_acc = new_acc
        torch.save(model, args.parfile + ".rnn")
        print("save model")
        print("accuracy ", best_acc)
