from __future__ import division, print_function
from tqdm import tqdm
import torch

from efficiency.log import fwrite
from efficiency.function import shell

from get_args import setup, dynamic_setup, clean_up
from dataloader import Dataset
from model import LSTMClassifier
from evaluate import Validator, Predictor


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train(proc_id, n_gpus, model=None, train_dl=None, validator=None,
          tester=None, epochs=20, lr=0.001, log_every_n_batches=100):
    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=lr, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dl) if proc_id == 0 else train_dl
        total_loss = 0
        cnt = 0
        for batch in pbar:
            batch_size = len(batch.tgt)

            if proc_id == 0 and cnt % log_every_n_batches < batch_size:
                pbar.set_description('E{:02d}, loss:{:.4f}, lr:{}'
                                 .format(epoch, total_loss / cnt if cnt else 0,
                                         opt.param_groups[0]['lr']))
                pbar.refresh()

            loss = model.loss(batch.input, batch.tgt)
            total_loss += loss.item() * batch_size
            cnt += batch_size

            opt.zero_grad()
            loss.backward()
            clip_gradient(model, 1e-1)
            opt.step()


        print('[Info] Avg loss')

        if n_gpus > 1: torch.distributed.barrier()

        model.eval()
        validator.evaluate(model, epoch)
        tester.evaluate(model, epoch)
        if proc_id == 0:
            validator.write_summary(epoch)
            tester.write_summary(epoch)


def bookkeep(model, validator, tester, args, INPUT_field):
    tester.final_evaluate(model)

    predictor = Predictor()
    predictor.pred_sent(INPUT_field, model)

    save_model_fname = validator.save_model_fname + '.e{:02d}.loss{:.4f}.torch'.format(
        validator.best_epoch, validator.best_error)
    cmd = 'cp {} {}'.format(validator.save_model_fname, save_model_fname)
    shell(cmd)

    clean_up(args)


def run(proc_id, n_gpus, devices, args):
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port=args.tcp_port)
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=dev_id)
    device = torch.device(dev_id)

    dataset = Dataset(proc_id=proc_id, data_dir=args.save_dir,
                      train_fname=args.train_fname,
                      preprocessed=args.preprocessed, lower=args.lower,
                      vocab_max_size=args.vocab_max_size, emb_dim=args.emb_dim,
                      save_vocab_fname=args.save_vocab_fname, verbose=True, )
    train_dl, valid_dl, test_dl = \
        dataset.get_dataloader(proc_id=proc_id, n_gpus=n_gpus, device=device,
                               batch_size=args.batch_size)


    model = LSTMClassifier(emb_vectors=dataset.INPUT.vocab.vectors,
                           lstm_dim=args.lstm_dim,
                           lstm_n_layer=args.lstm_n_layer,
                           lstm_dropout=args.lstm_dropout,
                           n_linear=args.n_linear, n_classes=args.n_classes)
    model = model.to(device)
    args = dynamic_setup(proc_id, args, dataset, model)


    validator = Validator(dataloader=valid_dl, save_dir=args.save_dir,
                          save_log_fname=args.save_log_fname,
                          save_model_fname=args.save_model_fname,
                          valid_or_test='valid',
                          vocab_itos=dataset.INPUT.vocab.itos)
    tester = Validator(dataloader=test_dl, save_log_fname=args.save_log_fname,
                       save_dir=args.save_dir, valid_or_test='test',
                       vocab_itos=dataset.INPUT.vocab.itos)

    train(proc_id, n_gpus, model=model, train_dl=train_dl,
          validator=validator, tester=tester, epochs=args.epochs, lr=args.lr)

    if proc_id == 0: bookkeep(model, validator, tester, args, dataset.INPUT)


def main():
    args = setup()

    n_gpus = args.n_gpus
    devices = range(n_gpus)

    if n_gpus == 1:
        run(0, n_gpus, devices, args)
    else:
        mp = torch.multiprocessing
        mp.spawn(run, args=(n_gpus, devices, args), nprocs=n_gpus)


if __name__ == '__main__':
    main()
