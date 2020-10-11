import torch
import torch.nn as nn
import torch.nn.functional as F


class TaggerModel(nn.Module):
    def __init__(self, num_chars, num_tags, embedding_size, rnn_size, char_rnn_size, dropout_rate):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_chars + 1, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm_forward_char = nn.LSTM(embedding_size, char_rnn_size, batch_first=True)
        self.lstm_backward_char = nn.LSTM(embedding_size, char_rnn_size, batch_first=True)
        self.lstm = nn.LSTM(char_rnn_size*2, rnn_size, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(rnn_size * 2, num_tags + 1)

    def forward(self, prefixes,suffixes):
        prefixes_embed = self.word_embeddings(prefixes)
        suffixes_embed = self.word_embeddings(suffixes)
        prefixes_embed = self.dropout(prefixes_embed)
        suffixes_embed = self.dropout(suffixes_embed)
        forward_lstm, _ = self.lstm_forward_char(prefixes_embed)
        backward_lstm, _ = self.lstm_backward_char(suffixes_embed)
        # concatenate the last time steps of each lstm
        concat = torch.cat([forward_lstm[: , -1, :] ,backward_lstm[: , -1, :] ], dim=1)
        lstm_out, _ = self.lstm(concat.unsqueeze(dim=0))
        tag_space = self.hidden2tag(lstm_out.squeeze(dim=0))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def run_test():
    
    from data import Data
    data = Data("train.tagged", "dev.tagged")
    tagger = TaggerModel( data.num_chars, data.num_tags, embedding_size=7, rnn_size=11, char_rnn_size=12, dropout_rate=0.1)
    pre,suf = data.words2ids_vecs("miwako seehundstationen".split())
    tag_scores = tagger.forward(torch.LongTensor(pre),torch.LongTensor(suf))


if __name__ == "__main__":
    run_test()
