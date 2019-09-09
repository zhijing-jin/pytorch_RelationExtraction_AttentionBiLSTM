import os
import configargparse
from itertools import product
from utils import show_time


def get_combinations():
    emb_dropouts = [0.3, 0.5]
    lstm_n_layers = [1, 2]
    lstm_dropouts = [0.1, 0.2, 0.3]
    lstm_dims = [128, 256, 512, 1024]
    # lstm_combines = ['add', 'concat']
    linear_dropouts = [0.3, 0.5]
    wds = [0, 1e-6, 1e-5]
    # wd

    choices = [emb_dropouts, lstm_n_layers, lstm_dropouts, lstm_dims,
               linear_dropouts, wds]
    combinations = list(product(*choices))
    return combinations


def get_results():
    folders = [f for f in os.listdir('.')
               if os.path.isdir(f) and f.startswith('090')]

    results = {}
    for dir in folders:
        file = os.path.join(dir, 'tmp_result.txt')
        if not os.path.isfile(file): continue
        with open(file) as f:
            final_line = [line.strip() for line in f][-1]
        pref = '<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = '
        suf = '% >>>'
        result = final_line.split(pref)[-1].split(suf)[0]
        result = float(result)
        results[dir] = result
    sorted_results = sorted(results.items(), key=lambda x: x[-1], reverse=True)

    # [('09081913', 71.3), ('09082311', 71.16), ('09082016', 70.58),
    # ('09090012', 70.49), ('09090140', 70.48), ('09082208', 70.47),
    # ('09082018', 70.39), ('09082315', 70.35), ('09082348', 70.23),
    # ('09081934', 70.13)]
    import pdb;
    pdb.set_trace()


def get_args():
    parser = configargparse.ArgumentParser('Options for multi-run')
    parser.add_argument('-start_ix', default=-50, type=int,
                        help='start ix of test batches')
    parser.add_argument('-end_ix', default=None, type=int,
                        help='end ix of test batch')
    parser.add_argument('-inspect_result', default=False, action='store_true', help='whether to inspect the results of grid search')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.inspect_result: get_results(); return

    combinations = get_combinations()  # 1296
    combination_set = combinations[args.start_ix:args.end_ix]

    for parameter_set in combination_set:
        uid = show_time()
        cmd = 'python train.py -save_dir {save_dir} ' \
              '-emb_dropout {} ' \
              '-lstm_n_layer {} ' \
              '-lstm_dropout {} ' \
              '-lstm_dim {} ' \
              '-linear_dropout {} ' \
              '-weight_decay {} ' \
            .format(save_dir=uid, *parameter_set)
        os.system(cmd)


if __name__ == '__main__':
    main()
