import torch
from torch.utils.data import DataLoader, Dataset
from torchnlp.samplers import BucketBatchSampler
from collections import defaultdict
from sklearn.model_selection import KFold

from src.util import process_csv


class DataHandler:
    def __init__(self, config, tokenizer):
        self.config = config
        data = process_csv(config.paths['path_data'])
        self.tokenizer = tokenizer

        label1_count = defaultdict(int)
        for label in data['label1']:
            label1_count[label] += 1

        label2_count = defaultdict(int)
        for label in data['label2']:
            label2_count[label] += 1

        label1_sorted = sorted(label1_count, reverse=True)
        label2_sorted = sorted(label2_count, reverse=True)

        self.label1_dict = dict()
        self.label2_dict = dict()
        self.label1_dict['<unk>'] = 0
        self.label2_dict['<unk>'] = 0
        init_len_label1 = len(self.label1_dict)
        init_len_label2 = len(self.label2_dict)

        for idx, label in enumerate(label1_sorted):
            self.label1_dict[label] = idx + init_len_label1

        for idx, label in enumerate(label2_sorted):
            self.label2_dict[label] = idx + init_len_label2

        self.num_label1 = len(self.label1_dict)
        self.num_label2 = len(self.label2_dict)

        self.tok2label1 = {value: key for key, value in self.label1_dict.items()}
        self.tok2label2 = {value: key for key, value in self.label2_dict.items()}

    def encode_label(self, data):
        data['tok_label1'] = self.label1_dict['<unk>']
        data['tok_label2'] = self.label2_dict['<unk>']
        for idx, series in data.iterrows():
            if series['label1'] in self.label1_dict:
                data.loc[idx, 'tok_label1'] = self.label1_dict[series['label1']]
            if series['label2'] in self.label2_dict:
                data.loc[idx, 'tok_label2'] = self.label2_dict[series['label2']]

    def _encode_text(self, data):
        tok_text = []
        for idx, series in data.iterrows():
            tok_text.append(self.encode_text(series['text']))
        data['tok_text'] = tok_text

    def prepare_data(self):
        path_data = self.config.paths['path_data']
        data_df = process_csv(path_data)

        kf = KFold(n_splits=self.config.n_splits, shuffle=True)
        train_index, test_index = next(kf.split(data_df['text']))

        train_df = data_df.loc[train_index, :]
        test_df = data_df.loc[test_index, :]

        self.encode_label(train_df)
        self.encode_label(test_df)
        self._encode_text(train_df)
        self._encode_text(test_df)

        train_sampler = BucketBatchSampler(train_df['tok_text'].tolist(),
                                           batch_size=self.config.batch_size_train,
                                           drop_last=False,
                                           sort_key=lambda x: len(x))

        train_loader = DataLoader(Data(train_df),
                                  batch_sampler=train_sampler,
                                  collate_fn=Data.collate_fn)

        test_sampler = BucketBatchSampler(test_df['tok_text'].tolist(),
                                          batch_size=self.config.batch_size_test,
                                          drop_last=False,
                                          sort_key=lambda x: len(x))

        test_loader = DataLoader(Data(test_df),
                                 batch_sampler=test_sampler,
                                 collate_fn=Data.collate_fn)

        return train_loader, test_loader

    def decode_label1(self, tok_label1):
        return self.tok2label1[tok_label1]

    def decode_label2(self, tok_label2):
        return self.tok2label2[tok_label2]

    def decode_text(self, tok_sent):
        return self.tokenizer.decode(tok_sent)

    def encode_text(self, sent):
        return self.tokenizer.encode(sent)


# 模型类
class Data(Dataset):
    def __init__(self, data_df):
        super(Data, self).__init__()
        self.tok_sent = data_df['tok_text'].tolist()
        self.tok_label1 = data_df['tok_label1'].tolist()
        self.tok_label2 = data_df['tok_label2'].tolist()
        self.sent = data_df['text'].tolist()

    def __len__(self):
        return len(self.tok_sent)

    def __getitem__(self, idx):
        tok_sent = torch.tensor(self.tok_sent[idx], dtype=torch.int64)
        len_sent = torch.LongTensor([len(self.tok_sent[idx])])
        tok_label1 = torch.tensor([self.tok_label1[idx]], dtype=torch.int64)
        tok_label2 = torch.tensor([self.tok_label2[idx]], dtype=torch.int64)
        sent = self.sent[idx]

        return tok_sent, len_sent, tok_label1, tok_label2, sent

    @staticmethod
    def collate_fn(batch):
        tok_sent, len_sent, tok_label1, tok_label2, sent = zip(*batch)
        len_sent = torch.cat(len_sent)
        tok_label1 = torch.cat(tok_label1)
        tok_label2 = torch.cat(tok_label2)
        return tok_sent, len_sent, tok_label1, tok_label2, sent
