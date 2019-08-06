import os

from src.util import load_checkpoint


def run(components, action, to_pred_sent):
    config = components['config']()
    tokenizer = components['tokenizer'](config)
    data_handler = components['data_handler'](config, tokenizer)
    Model = components['model']
    trainer = components['trainer'](config)

    if action == 'train':
        model = Model(config, data_handler, True)
        if os.path.exists(config.paths['path_ckpt']):
            load_checkpoint(config.paths['path_ckpt'], model, cpu=False)
        trainer.run_epochs(data_handler, model)
    elif action == 'predict':
        model = Model(config, data_handler, False)
        load_checkpoint(config.paths['path_ckpt'], model, cpu=True)
        trainer.pred_sent(data_handler, model, to_pred_sent)
    else:
        raise Exception('No such action available.')
