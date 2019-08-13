from tqdm import tqdm
import numpy as np
from efficiency.log import fwrite


class Evaluator:
    def __init__(self, dataloader, output_fname='./tmp/model_eval.txt'):
        self.dataloader = dataloader
        self.avg_error = 0
        self.output_fname = output_fname
        fwrite('', self.output_fname)

    def evaluate(self, model):
        assert not model.training, 'You need to call model.eval()'
        total_error = 0
        for batch in self.dataloader:
            batch_size = batch.inp.size(0)
            score = model.score(batch.inp, batch.tgt)
            total_error += score.item() * batch_size
        self.avg_error = total_error / len(self.dataloader)
        return self.avg_error

    def get_summary(self, header=''):
        def _format_value(v):
            if isinstance(v, float):
                return '{:.4f}'.format(v)
            elif isinstance(v, int):
                return '{:d}'.format(v)
            else:
                return '{}'.format(v)

        stats = {
            'avg_error': self.avg_error
        }
        writeout = [header] if header else []
        writeout += ['{}:{}'.format(k, _format_value(v))
                     for k, v in stats.items()]
        # import pdb;
        # pdb.set_trace()
        writeout = '\t'.join(writeout) + '\n'
        fwrite(writeout, self.output_fname, mode='a')
        print('[Info] Eval:', writeout)
        print('[Info] Saved model evaluation statistics to:', self.output_fname)
        return stats


def predict(dataloader, model):
    assert not model.training, 'You need to call model.eval()'

    test_preds = []
    for batch in dataloader:
        preds = model.predict(batch.inp)
        test_preds.append(preds)
    test_preds = np.hstack(test_preds)
    # import pdb;
    # pdb.set_trace()
    return test_preds
