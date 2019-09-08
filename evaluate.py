from __future__ import division, print_function
import os
import json
import torch
from efficiency.log import fwrite
from model import LSTMClassifier


class Validator:
    def __init__(self, dataloader=None, save_log_fname='tmp/run_log.txt',
                 save_model_fname='tmp/model.torch', save_dir='tmp/',
                 valid_or_test='valid', vocab_itos=dict()):
        self.avg_error = 0
        self.dataloader = dataloader
        self.save_log_fname = save_log_fname
        self.save_model_fname = save_model_fname
        self.valid_or_test = valid_or_test
        self.best_error = float('inf')
        self.best_epoch = 0
        self.save_dir = save_dir
        self.vocab_itos = vocab_itos

    def evaluate(self, model, epoch):
        error = 0
        count = 0
        n_correct = 0

        for batch_ix, batch in enumerate(self.dataloader):
            batch_size = len(batch.tgt)
            loss, acc = model.loss_n_acc(batch.input, batch.tgt)
            error += loss.item() * batch_size
            count += batch_size
            n_correct += acc
        avg_error = (error / count)
        self.avg_error = avg_error

        if (self.valid_or_test == 'valid') and (avg_error < self.best_error):
            self.best_error = avg_error
            self.best_epoch = epoch

            checkpoint = {
                'model': model.state_dict(),
                'settings': model.opts,
                'epoch': epoch,
            }
            torch.save(checkpoint, self.save_model_fname)

    def write_summary(self, epoch):
        def _format_value(v):
            if isinstance(v, float):
                return '{:.4f}'.format(v)
            elif isinstance(v, int):
                return '{:02d}'.format(v)
            else:
                return '{}'.format(v)

        summ = {
            'Eval': '(e{:02d},{})'.format(epoch, self.valid_or_test),
            'avg_error': self.avg_error,
        }
        summ = {k: _format_value(v) for k, v in summ.items()}
        writeout = json.dumps(summ)

        fwrite(writeout + '\n', self.save_log_fname, mode='a')
        printout = '[Info] {}'.format(writeout)
        print(printout)

        return writeout
    def reduce_lr(self, opt):
        if self.avg_error > self.best_error:
            for g in opt.param_groups:
                g['lr'] = g['lr'] / 2

    def final_evaluate(self, model):
        show_preds = []
        for batch in self.dataloader:
            pred = model.predict(batch.input)
            show_preds += [pred]


class Predictor:
    def __init__(self):
        pass

    def use_pretrained_model(self, model_fname, device=torch.device('cpu')):
        self.device = device
        checkpoint = torch.load(model_fname)
        model_opt = checkpoint['model_opt']
        model = LSTMClassifier(**model_opt)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        self.model = model

    def pred_sent(self, INPUT_field, model=None):
        ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
        device = next(model.parameters()).device
        if model is None: model = self.model
        model.eval()

        test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
        test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

        test_sen1 = INPUT_field.preprocess(test_sen1)
        test_sen1 = [[INPUT_field.vocab.stoi[x] for x in test_sen1]]

        test_sen2 = INPUT_field.preprocess(test_sen2)
        test_sen2 = [[INPUT_field.vocab.stoi[x] for x in test_sen2]]

        with torch.no_grad():
            test_batch = torch.LongTensor(test_sen1).to(device)

            output = model.predict(test_batch)
            if (output[0] == 1):
                print("Sentiment: Positive")
            else:
                print("Sentiment: Negative")


if __name__ == '__main__':
    pass
