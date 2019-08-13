import os
from efficiency.function import shell


def preproc(file):
    import csv
    import spacy
    spacy_en = spacy.load('en')

    def tokenizer(text):  # create a tokenizer function
        if text is None: return text
        text = ' '.join(text.split()).lower()
        toks = [tok.text for tok in spacy_en.tokenizer(text)]
        return ' '.join(toks)

    backup = file + '.untok'
    if not os.path.isfile(backup):
        shell('cp {} {}'.format(file, backup))

    with open(backup) as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    rows_tok = [{k: tokenizer(v) if k in ['comment_text'] else v
                 for k, v in row.items()}
                for row in rows]
    import pdb;
    pdb.set_trace()

    with open(file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=rows_tok[0].keys())
        writer.writeheader()
        writer.writerows(rows_tok)
    return rows_tok


def main():

    file_templ = 'data/{}.csv'
    # for typ in 'train valid test'.split():
    for typ in 'test'.split():
        file = file_templ.format(typ)
        preproc(file)


if __name__ == "__main__":
    main()
