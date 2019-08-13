from os.path import isdir
from os import mkdir
from copy import deepcopy
import torch


def setup(args):
    opts = deepcopy(args)

    device = 'cuda' if opts.gpu_id >= 0 else 'cpu'
    opts.device = torch.device(device)

    if not isdir(opts.save_dir):
        import pdb;

        pdb.set_trace()
        mkdir(opts.save_dir)
    return opts
