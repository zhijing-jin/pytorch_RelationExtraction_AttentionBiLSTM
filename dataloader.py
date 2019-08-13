import torch
from tqdm import tqdm
from torchtext.data import Field, TabularDataset, Iterator, BucketIterator
from efficiency.log import show_time, show_var


class Dataset:
    def __init__(self, data_dir='data/', train_fname='train.csv',
                 time_log=True, vocab_min_freq=2, preprocessed=False):
        if time_log: show_time("[Info] Start data loading from {}{}".format(data_dir, train_fname))

        self.time_log = time_log

        tokenize = (lambda x: x.split()) if preprocessed else "spacy"

        TEXT = Field(sequential=True, batch_first=True, tokenize=tokenize,
                     lower=True)
        LABEL = Field(sequential=False, dtype=torch.float, batch_first=True,
                      use_vocab=False)
        fields = [
            ("id", None),
            ("target", LABEL),
            ("comment_text", TEXT),
        ]
        train_ds, valid_ds = TabularDataset.splits(
            path=data_dir,
            train=train_fname,
            validation=train_fname.replace("train", "valid"),
            format='csv',
            skip_header=True,
            fields=fields
        )
        fields_tst = [
            ("id", None),
            ("comment_text", TEXT),
        ]
        test_ds = TabularDataset(
            path=data_dir + train_fname.replace("train", "test"),
            format='csv',
            skip_header=True,
            fields=fields_tst
        )

        if time_log: show_time("[Info] Finished data loading")

        TEXT.build_vocab(train_ds, min_freq=vocab_min_freq)

        import pdb;
        pdb.set_trace()
        TEXT.vocab.freqs.most_common(10)
        train_ds[0].__dict__.keys()
        train_ds[0].comment_text[:3]
        # show_var(["train_ds[0]"])

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds
        self.input_vocab_size = len(TEXT.vocab)

    def get_dataloader(self, batch_size=64, device=torch.device('cpu')):

        if self.time_log: show_time("[Info] Start making iterators")
        train_iter, valid_iter = BucketIterator.splits(
            (self.train_ds, self.valid_ds),
            batch_sizes=(batch_size, batch_size),
            device=device,
            sort_key=lambda x: len(x.comment_text),
            sort_within_batch=False,
            repeat=False
        )

        batch = next(train_iter.__iter__())
        # import pdb;
        # pdb.set_trace()
        batch.__dict__.keys()

        test_iter = Iterator(
            self.test_ds,
            batch_size=batch_size,
            sort=False,
            device=device,
            repeat=False,
            sort_within_batch=False
        )

        train_dl = BatchWrapper(train_iter)
        valid_dl = BatchWrapper(valid_iter)
        test_dl = BatchWrapper(test_iter)

        next(train_dl.__iter__())
        if self.time_log: show_time("[Info] Finished making iterators")

        return train_dl, valid_dl, test_dl


class BatchWrapper:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            batch_dic = {
                'inp': getattr(batch, 'comment_text'),
                'tgt': getattr(batch, 'target') if hasattr(batch, 'target') else None,
            }
            s = Struct(**batch_dic)
            # import pdb; pdb.set_trace()

            yield s

    def __len__(self):
        return len(self.dataloader)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def main():
    file_dir = "data/train.csv"
    dataset = Dataset(file_dir)
    train_dl, valid_dl, test_dl = dataset.get_dataloader()
    for epoch in range(10):
        for batch in tqdm(train_dl):
            import pdb;
            pdb.set_trace()
            print(batch)


if __name__ == "__main__":
    main()
