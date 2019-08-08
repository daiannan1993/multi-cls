from src.tokenizer_chs import TokenizerChs
from src.data_handler import DataHandler
from src.rnn_cls_model import RNNClsModel
from src.trainer_rnn import TrainerRNN
from src.util import process_paths
from src.run_main import run

action = 'train'
# action = 'predict'
to_pred_sent = '副驾驶安全带提示功能无法使用怎么处理'

name_model = 'rnn_maxus_failure'
ext_paths = {
    'path_log': 'logs/',
    'path_embedding': 'resources/char_embedding_tencent_chs.txt',
    'path_libs': 'resources/libs/',
    'path_data': 'data/failure_maxus_0807.csv',
    'path_ckpt': 'checkpoints/'+name_model+'.tar',
}


class Config:
    n_splits = 5
    # 网络参数
    num_hidden = 128
    num_layers = 2
    bidirectional = True
    dropout_emb = 0.2
    dropout_rnn = 0.2
    lr = 0.003

    batch_size_train = 32
    batch_size_test = 16

    paths = process_paths(ext_paths)
    num_epochs = 10


components = {
    'tokenizer': TokenizerChs,
    'data_handler': DataHandler,
    'model': RNNClsModel,
    'config': Config,
    'trainer': TrainerRNN
}

if __name__ == '__main__':
    run(components, action, to_pred_sent)
