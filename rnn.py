import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self,
                 input_size=100,
                 num_hidden=64,
                 bidirectional=True,
                 num_layers=1,
                 dropout_rate=0.0):
        super(EncoderLSTM, self).__init__()
        self.num_encoder = num_hidden
        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=num_hidden,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=True,
                               dropout=dropout_rate)

    def forward(self, x, *args):
        if len(args) == 0:
            output, (ht, ct) = self.encoder(x)
        else:
            len_x = args[0]
            sorted_seq_lengths, indices = torch.sort(len_x, descending=True)
            _, desorted_indices = torch.sort(indices, descending=False)
            x = x[indices]
            packed_inputs = nn.utils.rnn.pack_padded_sequence(x, sorted_seq_lengths, batch_first=True)
            output, (ht, ct) = self.encoder(packed_inputs)
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = padded_output[desorted_indices]
        return output, ht

    @classmethod
    def from_config(cls, config, tokenizer):
        _, emb_dim = tokenizer.weight_matrix.shape
        return cls(emb_dim,
                   config.num_hidden,
                   config.bidirectional,
                   config.num_layer,
                   config.dropout_rate)


class EncoderGRU(nn.Module):
    def __init__(self,
                 input_size=100,
                 num_hidden=64,
                 bidirectional=True,
                 num_layers=1,
                 dropout_rate=0.0):
        super(EncoderGRU, self).__init__()
        self.num_encoder = num_hidden
        self.encoder = nn.GRU(input_size=input_size,
                              hidden_size=num_hidden,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True,
                              dropout=dropout_rate)

    def forward(self, x, *args):
        if len(args) == 0:
            output, ht = self.encoder(x)
        else:
            len_x = args[0]
            sorted_seq_lengths, indices = torch.sort(len_x, descending=True)
            _, desorted_indices = torch.sort(indices, descending=False)
            x = x[indices]
            packed_inputs = nn.utils.rnn.pack_padded_sequence(x, sorted_seq_lengths, batch_first=True)
            output, ht = self.encoder(packed_inputs)
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = padded_output[desorted_indices]
        return output, ht

    @classmethod
    def from_config(cls, config, tokenizer):
        _, emb_dim = tokenizer.weight_matrix.shape
        return cls(emb_dim,
                   config.num_hidden,
                   config.bidirectional,
                   config.num_layer,
                   config.dropout_rate)


class EmbGRU(nn.Module):
    def __init__(self, weight_mat, num_hidden, num_layers, bidirectional, dropout_emb=0.2, dropout_rnn=0.3):
        super(EmbGRU, self).__init__()
        vocab_size, embedding_dim = weight_mat.shape

        self.dropout = nn.Dropout(p=dropout_emb)
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight_mat), freeze=False)

        self.encoder = EncoderGRU(input_size=embedding_dim,
                                  num_hidden=num_hidden,
                                  num_layers=num_layers,
                                  bidirectional=bidirectional,
                                  dropout_rate=dropout_rnn)

        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input, *args):
        batch_size = input.size(0)
        embeds = self.word_embedding(input)  # embeds的维度应该是(batch_size, seq_len, embedding_dim)
        embeds = self.dropout(embeds)
        gru_out, _ = self.encoder(embeds, *args)
        gru_pool = self.maxpool(gru_out.permute(0,2,1)).view(batch_size,-1)
        return gru_out, gru_pool

    @classmethod
    def from_config(cls, config, tokenizer):
        return cls(tokenizer.weight_matrix,
                   config.num_hidden,
                   config.num_layers,
                   config.bidirectional,
                   config.dropout_emb,
                   config.dropout_rnn)


class EmbLSTM(nn.Module):
    def __init__(self, weight_mat, num_hidden, num_layers, bidirectional, dropout_emb=0.2, dropout_rnn=0.3):
        super(EmbLSTM, self).__init__()
        vocab_size, embedding_dim = weight_mat.shape

        self.dropout = nn.Dropout(p=dropout_emb)
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weight_mat), freeze=False)

        self.encoder = EncoderLSTM(input_size=embedding_dim,
                                   num_hidden=num_hidden,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional,
                                   dropout_rate=dropout_rnn)

        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input, *args):
        batch_size = input.size(0)
        embeds = self.word_embedding(input)  # embeds的维度应该是(batch_size, seq_len, embedding_dim)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.encoder(embeds, *args)
        lstm_pool = self.maxpool(lstm_out.permute(0,2,1)).view(batch_size,-1)
        return lstm_out, lstm_pool

    @classmethod
    def from_config(cls, config, tokenizer):
        return cls(tokenizer.weight_matrix,
                   config.num_hidden,
                   config.num_layers,
                   config.bidirectional,
                   config.dropout_emb,
                   config.dropout_rnn)
