from __future__ import division, print_function
import os
import json
import torch
from utils import fwrite, show_var, shell

from model import LSTMClassifier


class Validator:
    def __init__(self, dataloader=None, save_log_fname='tmp/run_log.txt',
                 save_model_fname='tmp/model.torch', save_dir='tmp/',
                 valid_or_test='valid', vocab_itos=dict(), label_itos=dict()):
        self.avg_loss = 0
        self.dataloader = dataloader
        self.save_log_fname = save_log_fname
        self.save_model_fname = save_model_fname
        self.valid_or_test = valid_or_test
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.save_dir = save_dir
        self.vocab_itos = vocab_itos
        self.label_itos = label_itos

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
        avg_loss = (error / count)
        self.avg_loss = avg_loss
        self.acc = (n_correct / count)

        if (self.valid_or_test == 'valid') and (avg_loss < self.best_loss):
            self.best_loss = avg_loss
            self.best_epoch = epoch

            checkpoint = {
                'model': model.state_dict(),
                'model_opt': model.opts,
                'epoch': epoch,
            }
            torch.save(checkpoint, self.save_model_fname)

    def write_summary(self, epoch=0, summ=None):
        def _format_value(v):
            if isinstance(v, float):
                return '{:.4f}'.format(v)
            elif isinstance(v, int):
                return '{:02d}'.format(v)
            else:
                return '{}'.format(v)

        summ = {
            'Eval': '(e{:02d},{})'.format(epoch, self.valid_or_test),
            'loss': self.avg_loss,
            'acc': self.acc,
        } if summ is None else summ
        summ = {k: _format_value(v) for k, v in summ.items()}
        writeout = json.dumps(summ)

        fwrite(writeout + '\n', self.save_log_fname, mode='a')
        printout = '[Info] {}'.format(writeout)
        print(printout)

        return writeout

    def reduce_lr(self, opt):
        if self.avg_loss > self.best_loss:
            for g in opt.param_groups:
                g['lr'] = g['lr'] / 2

    def final_evaluate(self, model,
                       perl_fname='eval/semeval2010_task8_scorer-v1.2.pl'):
        preds = []
        truths = []
        for batch in self.dataloader:
            pred = model.predict(batch.input)
            preds += pred

            truth = batch.tgt.view(-1).detach().cpu().numpy().tolist()
            truths += truth

        pred_fname = os.path.join(self.save_dir, 'tmp_pred.txt')
        truth_fname = os.path.join(self.save_dir, 'tmp_truth.txt')
        result_fname = os.path.join(self.save_dir, 'tmp_result.txt')

        writeout = ["{}\t{}\n".format(ix, self.label_itos[pred]) for ix, pred in
                    enumerate(preds)]
        fwrite(''.join(writeout), pred_fname)

        writeout = ["{}\t{}\n".format(ix, self.label_itos[truth]) for ix, truth
                    in enumerate(truths)]
        fwrite(''.join(writeout), truth_fname)

        cmd = 'perl {} {} {}'.format(perl_fname, pred_fname, truth_fname)
        stdout, _ = shell(cmd, stdout=True)
        fwrite(stdout, result_fname)


class Predictor:
    def __init__(self, vocab_fname):
        with open(vocab_fname) as f:
            vocab = json.load(f)
        self.tgt_itos = vocab['tgt_vocab']['itos']
        self.input_stoi = vocab['input_vocab']['stoi']

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
        if model is None: model = self.model
        model.eval()
        device = next(model.parameters()).device
        input_stoi = self.input_stoi

        test_sentence = "The most common ENT_1_START audits ENT_1_END were about ENT_2_START waste ENT_2_END and recycling ."
        test_label = 'Product-Producer(e2,e1)'

        test_sen_ixs = INPUT_field.preprocess(test_sentence)
        test_sen_ixs = [[input_stoi[x] if x in input_stoi else 0
                         for x in test_sen_ixs]]

        with torch.no_grad():
            test_batch = torch.LongTensor(test_sen_ixs).to(device)

            output = model.predict(test_batch)
            prediction = self.tgt_itos[output[0]]
            show_var(['test_sentence', 'test_label', 'prediction'])


if __name__ == '__main__':
    pass
