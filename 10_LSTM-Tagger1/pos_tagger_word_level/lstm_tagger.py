import torch
import torch.nn as nn
import torch.nn.functional as F

class TaggerModel(nn.Module):
    def __init__(self, num_words, num_tags, embedding_size, rnn_size, dropout_rate):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_words + 1, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embedding_size, rnn_size, bidirectional=True)
        self.hidden2tag = nn.Linear(rnn_size * 2, num_tags + 1)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        #print(embeds.size())
        dropped_embeds = self.dropout(embeds)
        #print(dropped_embeds.size())
        lstm_out, h = self.lstm(dropped_embeds.unsqueeze(dim=1))
        #print("squeeze ", dropped_embeds.unsqueeze(dim=1).size())
        #print("lstm out ",lstm_out.size())
        #print("lstm out hidden",h[0].size(), h[1].size())
        lstm_out = self.dropout(lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        #print(tag_space.size())
        tag_scores = F.log_softmax(tag_space, dim=1)
        #print(tag_scores.size())
        return tag_scores


def run_test():
    tagger = TaggerModel(10, 15, 30, 20, 0.2)
    word_ids = [1, 2, 3, 4, 6, 6, 3, 0]
    tag_scores = tagger.forward(torch.LongTensor(word_ids))


if __name__ == "__main__":
    run_test()
