import os
import re
import json
import random

from efficiency.log import fwrite
from efficiency.function import shell
from efficiency.nlp import NLP


def load_data(path):
    ENT_1_START = '<e1>'
    ENT_1_END = '</e1>'
    ENT_2_START = '<e2>'
    ENT_2_END = '</e2>'

    nlp = NLP()
    data = []
    with open(path) as f:
        lines = [line.strip() for line in f]
    for idx in range(0, len(lines), 4):
        id = int(lines[idx].split("\t")[0])
        relation = lines[idx + 1]

        sentence = lines[idx].split("\t")[1][1:-1]
        sentence = sentence.strip()

        sentence = sentence.replace(ENT_1_START, ' ENT_1_START ')
        sentence = sentence.replace(ENT_1_END, ' ENT_1_END ')
        sentence = sentence.replace(ENT_2_START, ' ENT_2_START ')
        sentence = sentence.replace(ENT_2_END, ' ENT_2_END ')

        sentence = nlp.word_tokenize(sentence)

        ent1 = sentence.split(' ENT_1_START ')[-1].split(' ENT_1_END ')[0]
        ent2 = sentence.split(' ENT_2_START ')[-1].split(' ENT_2_END ')[0]


        data.append({
            'label': relation,
            'sentence': sentence,
            'ent1': ent1,
            'ent2': ent2,
            'id': id,
        })

    return data


def split(data, dev_size=800):
    random.shuffle(data)
    dev = data[:dev_size]
    train = data[dev_size:]
    return train, dev


def save_to_json(data, file):
    writeout = json.dumps(data, indent=4)
    fwrite(writeout, file)
    print('[Info] Saving {} data to {}'.format(len(data), file))


def download():
    cmd = 'mkdir data/re_semeval/raw 2>/dev/null \n' \
          'wget https://raw.githubusercontent.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction/master/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT -P data/re_semeval/raw \n' \
          'wget https://raw.githubusercontent.com/SeoSangwoo/Attention-Based-BiLSTM-relation-extraction/master/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT -P data/re_semeval/raw \n'
    shell(cmd)


def main():
    download()

    data = {}
    data_dir = 'data/re_semeval/'

    raw_fname = os.path.join(data_dir, 'raw', 'TRAIN_FILE.TXT')
    nontest_data = load_data(raw_fname)
    data['train'], data['valid'] = split(nontest_data)

    raw_fname = os.path.join(data_dir, 'raw', 'TEST_FILE_FULL.TXT')
    data['test'] = load_data(raw_fname)

    for key, value in data.items():
        json_fname = os.path.join(data_dir, '{}.json'.format(key))
        save_to_json(value, json_fname)


if __name__ == "__main__":
    main()
