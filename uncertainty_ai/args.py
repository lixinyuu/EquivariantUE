import argparse
import yaml
from pathlib import Path

def parse_arguments():
    train_parser = argparse.ArgumentParser(description="Training parser for training hyper-parrameters at ech checkpoint.")
    train_parser.add_argument('--n_epoch', type=int, default=200)
    train_parser.add_argument('--n_early_stopping', type=int, default=200)
    train_parser.add_argument('--optimizer', type=str, default='Adam')
    train_parser.add_argument('--optimizer_args', type=dict, default = {})
    train_parser.add_argument("--loader_tr_args", type=dict, default = {'batch_size': 32})
    train_parser.add_argument("--loader_te_args", type=dict, default = {'batch_size': 32})
    train_parser.add_argument('--learning_rate', type=float, default=0.0002)
    train_parser.add_argument('--momentum', type=float, default=0.9)
    train_parser.add_argument('--weight_decay', type=float, default=0.0005)
    train_parser.add_argument('--lr_warmup', type=int, default=0)
    train_parser.add_argument('--lr_warmup_start', type=float, default=0.00001)
    train_parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=30)
    train_parser.add_argument('--lr_schedule', action="store_const", default=True, const=True)
    train_parser.add_argument('--loss_fn', type=str, default='MSELoss')
    train_parser.add_argument('--loss_fn_args', type=dict, default={})
    train_parser.add_argument('--train_to_end', action="store_const", default=False, const=True)
    train_parser.add_argument('--n_validation_set', type=int, default=0)
    train_parser.add_argument('--choose_best_val_model', action="store_const", default=False, const=True)
    train_parser.add_argument('--model', type=str, default='SchNet')
    train_parser.add_argument('--model_args', type=dict, default={})
    train_parser.add_argument('--fine_tune_layers', type=int, default=1)
    train_parser.add_argument('--pretrained_model', action='store_const', default=False, const=True)
    train_parser.add_argument('--initial_state_file', type=str, default='Adam')
    train_parser.add_argument('--save_first_state', action='store_const', default=False, const=True)
    train_parser.add_argument('--continue_training', action='store_const', default=False, const=True)
    train_parser.add_argument(('--amp'), action='store_const', default=False, const=True)
    train_args, _ = train_parser.parse_known_args()
    return train_args

def replace_with_config_arguments(args, config):
    assert Path(config).is_file()
    with open(config, 'r') as fp:
        config_dict = yaml.load(fp, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            # if isinstance(value, list):
            #     for v in value:
            #         arg_dict[key].append(v)
            # else:
            #     arg_dict[key] = value
            arg_dict[key] = value
        else:
            pass
    return args


