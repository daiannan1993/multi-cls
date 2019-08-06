import time
from tqdm import tqdm

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.util import save_checkpoint


class TrainerRNN:
    def __init__(self, config):
        self.config = config

    # 训练
    def run_epochs(self, data_handler, model):
        model.cuda()
        optimizer = model.make_optimizer(self.config.lr)
        train_loader, test_loader = data_handler.prepare_data()
        best_acc = 0

        scheduler = CosineAnnealingLR(optimizer, self.config.num_epochs)

        for epoch in range(1, self.config.num_epochs + 1):
            print('-----Epoch %s ---------' % (str(epoch)))
            time.sleep(0.01)
            self.train(train_loader, model, optimizer)

            with torch.no_grad():
                acc_label1, acc_label2 = self.eval(test_loader, model)
                acc = acc_label1 + acc_label2
                if acc > best_acc:
                    save_checkpoint(self.config.paths['path_ckpt'], model, optimizer)
                    best_acc = acc

            scheduler.step()
            time.sleep(0.01)

        self.print_bad_cases(data_handler, model, test_loader)

    @staticmethod
    def train(data_loader, model, optimizer):
        loss_sum = 0

        for tok_sent, len_sent, tok_label1, tok_label2, sent in tqdm(data_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            tok_sent_padded = pad_sequence(tok_sent, batch_first=True)

            tok_sent_padded = tok_sent_padded.cuda()
            len_sent = len_sent.cuda()
            tok_label1 = tok_label1.cuda()
            tok_label2 = tok_label2.cuda()

            loss = model.calc_loss(tok_sent_padded, len_sent, tok_label1, tok_label2)
            loss_sum += loss.item()

            loss.backward()
            optimizer.step()

        print("tag train loss: %1.3f" % loss_sum)

    # 实现n句话的测试
    @staticmethod
    def eval(test_loader, model, during_train=True):
        num_total = 0
        num_wrong_1 = 0
        num_wrong_2 = 0
        bad_cases_1 = []
        bad_cases_2 = []
        for tok_sent, len_sent, tok_label1, tok_label2, sent in test_loader:
            model.eval()
            tok_sent_padded = pad_sequence(tok_sent, batch_first=True)

            tok_sent_padded = tok_sent_padded.cuda()
            len_sent = len_sent.cuda()

            label1_pred, label2_pred = model.decode(tok_sent_padded, len_sent)

            for i in range(len(tok_sent)):
                num_total += 1

                if label1_pred[i] != tok_label1[i]:
                    num_wrong_1 += 1
                    if not during_train:
                        bad_cases_1.append({'text': sent[i],
                                            'label1_true': tok_label1[i].item(),
                                            'label1_pred': label1_pred[i]})

                if label2_pred[i] != tok_label2[i]:
                    num_wrong_2 += 1
                    if not during_train:
                        bad_cases_2.append({'text': sent[i],
                                            'label2_true': tok_label2[i].item(),
                                            'label2_pred': label2_pred[i]})

        acc_label1 = 1 - (num_wrong_1/num_total)
        acc_label2 = 1 - (num_wrong_2/num_total)
        print("evaluation on test set")
        print('label1 acc: %1.4f' % acc_label1)
        print('label2 acc: %1.4f' % acc_label2)

        if not during_train:
            return bad_cases_1, bad_cases_2
        else:  # during train
            return acc_label1, acc_label2

    def print_bad_cases(self, data_handler, model, test_loader):
        bad_cases_1, bad_cases_2 = self.eval(test_loader, model, during_train=False)

        if bad_cases_1:
            df_bc_1 = pd.DataFrame(bad_cases_1)
            df_bc_1['label1_true'] = df_bc_1['label1_true'].apply(data_handler.decode_label1)
            df_bc_1['label1_pred'] = df_bc_1['label1_pred'].apply(data_handler.decode_label1)
            df_bc_1.to_csv(self.config.paths['path_log'] + 'bad_cases_1.csv', sep='\t', index=False,
                           columns=['text', 'label1_true', 'label1_pred'])

        if bad_cases_2:
            df_bc_2 = pd.DataFrame(bad_cases_2)
            df_bc_2['label2_true'] = df_bc_2['label2_true'].apply(data_handler.decode_label2)
            df_bc_2['label2_pred'] = df_bc_2['label2_pred'].apply(data_handler.decode_label2)
            df_bc_2.to_csv(self.config.paths['path_log'] + 'bad_cases_2.csv', sep='\t', index=False,
                           columns=['text', 'label2_true', 'label2_pred'])

    def predict(self, data_handler, model, sent):
        model.eval()
        tok_sent = data_handler.encode_text(sent)
        tok_sent_ts = torch.tensor(tok_sent).view(1, -1)
        label1_pred, label2_pred = model.decode(tok_sent_ts, torch.tensor([len(tok_sent)]))
        label1_pred = data_handler.decode_label1(label1_pred[0])
        label2_pred = data_handler.decode_label2(label2_pred[0])
        return label1_pred, label2_pred

    def pred_sent(self, data_handler, model, sent):
        label1_pred, label2_pred = self.predict(data_handler, model, sent)
        result_label = pd.Series({'text': sent, 'label1': label1_pred, 'label2': label2_pred})
        print(result_label)
