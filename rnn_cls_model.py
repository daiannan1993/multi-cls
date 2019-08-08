import torch.nn as nn
import torch

from rnn import EmbLSTM


class RNNClsModel(nn.Module):
    def __init__(self, config, data_handler, is_train):
        super(RNNClsModel, self).__init__()
        self.bilstm = EmbLSTM.from_config(config, data_handler.tokenizer)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.is_train = is_train

        directions = 1
        if config.bidirectional: directions = 2

        self.hidden2label1 = nn.Linear(directions * config.num_hidden, data_handler.num_label1)
        self.hidden2label2 = nn.Linear(directions * config.num_hidden, data_handler.num_label2)

        self.criterion = nn.NLLLoss()

    def forward(self, tok_sent_padded, len_sents):
        _, gru_pooled = self.bilstm(tok_sent_padded, len_sents)  # n, t, d
        logits_label1 = self.hidden2label1(gru_pooled)
        logits_label2 = self.hidden2label2(gru_pooled)

        return logits_label1, logits_label2

    def calc_loss(self, tok_sent, len_sents, tok_label1, tok_label2):
        logits_label1, logits_label2 = self.forward(tok_sent, len_sents)
        loss_label1 = self.criterion(self.log_softmax(logits_label1), tok_label1)
        loss_label2 = self.criterion(self.log_softmax(logits_label2), tok_label2)
        loss = loss_label1 + loss_label2
        return loss

    def decode(self, tok_sent, len_sents):
        logits_label1, logits_label2 = self.forward(tok_sent, len_sents)
        label1_pred = logits_label1.max(dim=1)[1].tolist()
        label2_pred = logits_label2.max(dim=1)[1].tolist()
        return label1_pred, label2_pred

    def make_optimizer(self, lr, lr_emb=0.5):
        embedding_params = self.bilstm.word_embedding.parameters()
        id_embedding_params = list(map(id, embedding_params))
        base_params = filter(lambda p: id(p) not in id_embedding_params,
                             self.parameters())
        optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': embedding_params, 'lr': lr * lr_emb}], lr=lr)
        return optimizer
