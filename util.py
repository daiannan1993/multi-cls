import pandas as pd
import os
import torch


def process_csv(path):
    data_df = pd.read_csv(path,
                          header=0,
                          sep=',',
                          usecols=['标准故障点', '标准故障描述', '标准问'])
    data_df.rename(columns={'标准故障点': 'label1',
                            '标准故障描述': 'label2',
                            '标准问': 'text'},
                   inplace=True)
    return data_df


def process_path(path):
    folder_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(folder_path.replace('/src', ''), 'ext', path)


def process_paths(in_paths):
    out_paths = {}
    for key, value in in_paths.items():
        out_paths[key] = process_path(value)
    return out_paths

# https://github.com/leigh-plt/cs231n_hw2018/blob/master/assignment2/pytorch_tutorial.ipynb
def save_checkpoint(checkpoint_path, model, optimizer):
    ckpt = {'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
             }
    torch.save(ckpt, checkpoint_path)
    print('checkpoints saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, cpu=False):
    # 不在此处load optimizer
    # 因为load model用于训练的话，需要optimizer的声明在model.cuda()之后
    # 如果不用于训练，则不需要optimizer
    if cpu:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    else:
        ckpt = torch.load(checkpoint_path)

    model.load_state_dict(ckpt['state_dict'], strict=True)
    # optimizer.load_state_dict(ckpt['optimizer'])
    print('checkpoints loaded from %s' % checkpoint_path)