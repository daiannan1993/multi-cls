import numpy as np


class TokenizerChs:
    # 读取训练数据
    def __init__(self, config):
        super(TokenizerChs, self).__init__()
        self.config = config
        self.gen_weightmat()

    def load_embedding(self):
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')
        with open(self.config.paths['path_embedding'], 'r') as file:
            info = file.readline() # 如果第一行是说明嵌入的维度和数目的话
            embedding = dict(get_coefs(*line.split(' ')) for line in file)
        vocab_size, embedding_dim = list(map(int, (info.split(' '))))

        return embedding, vocab_size, embedding_dim

    def gen_tokenizer(self, embedding):
        self.char_dict = {'<pad>': 0, '<unk>': 1}
        for idx, char in enumerate(embedding.keys()):
            self.char_dict[char] = idx + 2

    def gen_weightmat(self):
        embedding, vocab_size, embedding_dim = self.load_embedding()

        # 把所有的当前字向量的平均作为unk的字向量
        unk_emb_list = []
        for value in embedding.values():
            unk_emb_list.append(value[np.newaxis, :])
        embedding_unk = np.concatenate(unk_emb_list, axis=0).mean(0)

        self.gen_tokenizer(embedding)

        self.weight_matrix = np.zeros([vocab_size+2, embedding_dim])
        for word, index in self.char_dict.items():
            if index == 0 or index == 1:
                continue
            self.weight_matrix[index] = embedding[word]
        self.weight_matrix[1] = embedding_unk

    def encode(self, sent):
        out = []
        for char in sent:
            char = char.lower()
            try:
                out.append(self.char_dict[char])
            except KeyError:
                out.append(self.char_dict['<unk>'])
        return out

    def decode(self, tokens):
        index2char = {index: char for char, index in self.char_dict.items()}
        output_list = [index2char[token] for token in tokens]
        output_text = ''.join(output_list)
        return output_text

    def gen_sent_list(self, sent):
        return list(sent)
