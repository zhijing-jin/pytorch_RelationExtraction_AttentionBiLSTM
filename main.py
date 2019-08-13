"""
This implementation works with a minibatch of size 1 only for both training and inference.
"""
from __future__ import division
import argparse
import datetime
import time
from copy import deepcopy
from tqdm import tqdm
from efficiency.log import show_time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model import SimpleBiLSTMBaseline
from dataloader import Dataset
from evaluate import Evaluator, predict


def main(opts):
    dataset = Dataset(data_dir=opts.data_dir, train_fname=opts.data_train)
    train_dl, valid_dl, test_dl = dataset.get_dataloader(
        batch_size=opts.batch_size, device=opts.device)

    model = SimpleBiLSTMBaseline(
        vocab_size=dataset.input_vocab_size,
        emb_dim=opts.emb_dim, hidden_dim=opts.hidden_dim,
        num_linear=opts.num_linear, predict_dim=1
    )

    if opts.gpu_id >= 0:
        model = model.cuda()

    if opts.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=opts.lr)
    evaluator = Evaluator(valid_dl)

    # start training
    for epoch in range(1, opts.n_epochs + 1):
        batch_loss = 0
        batch_cnt = 0

        model.train()

        for batch in tqdm(train_dl):
            inp = batch.inp
            batch_size = len(inp)

            loss = model.score(inp, batch.tgt)
            loss = loss / batch_size

            optimizer.zero_grad()
            loss.backward()

            batch_loss += loss.item()
            batch_cnt += batch_size

            if opts.clip_grad:
                clip_grad_norm_(model.parameters(), opts.clip_bound)
            optimizer.step()

            if batch_cnt % opts.log_every_n_batches < batch_size:
                avg_loss = sum(batch_loss) / len(batch_loss)
                show_time("Epoch {:03d}: averaged_loss {}"
                          .format(epoch, avg_loss))

                batch_loss = 0
        model.eval()
        avg_error = evaluator.evaluate(model)
        # import pdb;
        # pdb.set_trace()
        evaluator.get_summary("Epoch {:03d}".format(epoch))

        test_preds = predict(test_dl, model)

        torch.save(model, opts.save_dir + 'model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple Classifier")

    # dataset
    parser.add_argument('--data_dir', type=str, default="data/",
                        help="directory of datafiles")
    parser.add_argument('--data_train', type=str, default="train.csv",
                        help="name of training file")
    parser.add_argument('--save_dir', type=str, default="tmp/",
                        help="directory to save outputs")
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')

    # model specs
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='size of hidden layers in LSTM and linear layers')
    parser.add_argument('--emb_dim', type=int, default=100,
                        help="dim of embedding file")
    parser.add_argument('--num_linear', type=int, default=3,
                        help="number of linear layers")

    # optimizer specs
    parser.add_argument('--optimizer', type=str, choices=['Adam'],
                        default='Adam', help='The type of optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--clip_grad', action='store_true', default=True,
                        help='gradient clipping is required to prevent gradient explosion')
    parser.add_argument('--clip_bound', type=float, default=0.25,
                        help='constraint of gradient norm for gradient clipping')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='the number of epochs to train')

    # gpu specs
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which gpu to use, -1 if no gpu')
    # log specs
    parser.add_argument('--log_every_n_batches', type=int, default=1024,
                        help="show log every n batches")
    args = parser.parse_args()

    from utils import setup

    opts = setup(args)

    main(opts)
