import os
import sys
import json
import torch
import numpy as np
import random
import configargparse
from efficiency.log import show_time, fwrite
from efficiency.function import shell


def get_args():
    cur_time = show_time(printout=False)
    parser = configargparse.ArgumentParser(
        description='Args for Text Classification')
    group = parser.add_argument_group('Model Hyperparameters')
    group.add_argument('-lstm_n_layer', default=2, type=int,
                       help='num of layers in LSTM')
    group.add_argument('-lstm_dropout', default=0.1, type=float,
                       help='dropout in >=1th LSTM layer')
    group.add_argument('-lstm_dim', default=256, type=int,
                       help='dimension of the lstm hidden states')
    group.add_argument('-vocab_max_size', default=100000, type=int,
                       help='max number of words in vocab')
    group.add_argument('-n_linear', default=2, type=int,
                       help='number of linear layers after lstm')
    group.add_argument('-n_classes', default=2, type=int,
                       help='number of classes to predict')
    group.add_argument('-emb_dim', default=100, type=int,
                       help='dimension of embedding vectors')

    group = parser.add_argument_group('Training Specs')
    group.add_argument('-seed', default=0, type=int, help='random seed')
    group.add_argument('-batch_size', default=64, type=int, help='batch size')
    group.add_argument('-epochs', default=20, type=int,
                       help='number of epochs to train the model')
    group.add_argument('-lr', default=0.001, type=float, help='learning rate')
    group.add_argument('-decay', default=0.001, type=float, help='weight decay')

    group = parser.add_argument_group('Files')
    group.add_argument('-data_dir', default='data/re_semeval/', type=str,
                       help='the directory for data files')
    group.add_argument('-train_fname', default='train.csv', type=str,
                       help='training file name')
    group.add_argument('-data_sizes', nargs=3, default=[None, None, None],
                       type=int,
                       help='# samples to use in train/dev/test files')
    group.add_argument('-preprocessed', action='store_false', default=True,
                       help='whether input data is preprocessed by spacy')
    group.add_argument('-lower', action='store_false', default=True,
                       help='whether to lowercase the input data')

    group.add_argument('-uid', default=cur_time, type=str,
                       help='the id of this run')
    group.add_argument('-save_dir', default='tmp/', type=str,
                       help='directory to save output files')
    group.add_argument('-save_dir_cp', default='tmp_cp/', type=str,
                       help='directory to backup output files')
    group.add_argument('-save_meta_fname', default='run_meta.txt', type=str,
                       help='file name to save arguments and model structure')
    group.add_argument('-save_log_fname', default='run_log.txt', type=str,
                       help='file name to save training logs')
    group.add_argument('-save_valid_fname', default='valid_e00.txt', type=str,
                       help='file name to save valid outputs')
    group.add_argument('-save_vis_fname', default='example.txt', type=str,
                       help='file name to save visualization outputs')
    group.add_argument('-save_model_fname', default='model', type=str,
                       help='file to torch.save(model)')
    group.add_argument('-save_vocab_fname', default='vocab.json', type=str,
                       help='file name to save vocab')

    group = parser.add_argument_group('Run specs')
    group.add_argument('-n_gpus', default=1, type=int, help='# gpus to run on')
    group.add_argument('-load_model', default='', type=str,
                       help='path to pretrained model')
    group.add_argument('-verbose', action='store_true', default=False,
                       help='whether to show pdb.set_trace() or not')

    args = parser.parse_args()
    return args


def setup():
    args = get_args()

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    else:
        shell('rm {}/*'.format(args.save_dir))
    args.save_meta_fname = os.path.join(args.save_dir, args.save_meta_fname)
    args.save_log_fname = os.path.join(args.save_dir, args.save_log_fname)
    args.save_valid_fname = os.path.join(args.save_dir, args.save_valid_fname)
    args.save_vis_fname = os.path.join(args.save_dir, args.save_vis_fname)
    args.save_model_fname = os.path.join(args.save_dir, args.save_model_fname)
    args.save_vocab_fname = os.path.join(args.save_dir, args.save_vocab_fname)

    args.data_sizes = \
        select_data(save_dir=args.save_dir, data_dir=args.data_dir,
                    train_fname=args.train_fname, data_sizes=args.data_sizes,
                    skip_header=True, verbose=True)

    if not args.verbose: import pdb; pdb.set_trace = lambda: None

    return args


def dynamic_setup(proc_id, args, dataset, model):
    def _count_parameters(model):
        return sum(
            p.numel() for p in model.parameters() if p.requires_grad)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    vocab = dataset.INPUT.vocab
    args.vocab_size = len(vocab)
    args.n_params = _count_parameters(model)

    if proc_id == 0:
        writeout = " ".join(sys.argv[1:]).replace(' -', ' \ \n-')
        writeout += '\n' * 3 + \
                    json.dumps(args.__dict__, indent=4, sort_keys=True)
        writeout += '\n' * 3 + repr(model)

        fwrite(writeout, args.save_meta_fname)

        print('[Info] Model has {} trainable parameters'.format(args.n_params))

    return args


def clean_up(args):
    cmd = 'cp -a {} {}'.format(args.save_dir, args.save_dir_cp)
    shell(cmd)


def select_data(save_dir='./tmp', data_dir='./data/wiki_person',
                train_fname='train.csv', data_sizes=[None, None, None],
                skip_header=True, verbose=True):
    files = ['train', 'valid', 'test']
    suffix = '.' + train_fname.split('.')[-1]
    n_lines = {}

    def _get_num_lines(file):
        with open(file) as f:
            data = [line.strip() for line in f if line]
        num_lines = len(data) if not skip_header else len(data) - 1
        return num_lines

    for file, data_size in zip(files, data_sizes):

        read_from = os.path.join(data_dir,
                                 train_fname.replace('train', file))
        save_to = os.path.join(save_dir, file + suffix)

        if skip_header:
            shell('head -1 {fin} > {fout}'.format(fin=read_from, fout=save_to))

        if data_size is None:
            cmd = "awk 'NR>=2' {fin} | shuf >> {fout}" \
                .format(fin=read_from, fout=save_to)
        else:
            cmd = "awk 'NR>=2&&NR<={line_num}' {fin} | shuf | head -{data_size} >> {fout}" \
                .format(line_num=1 + data_size, data_size=data_size,
                        fin=read_from, fout=save_to)

        shell(cmd)
        n_lines[file] = _get_num_lines(save_to)

    if verbose:
        writeout = ['{}: {}'.format(*item) for item in n_lines.items()]
        writeout = ', '.join(writeout)
        print('[Info] #samples in', writeout)
    return list(n_lines.values())


if __name__ == '__main__':
    args = setup()
    print(args.__dict__)
