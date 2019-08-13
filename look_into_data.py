import numpy as np
import torch


def look_into_data(train_file):
    import pandas as pd

    pd.read_csv(train_file).head(2)
    import pdb; pdb.set_trace()


def train():
    pass


def main():
    train_file = "data/train.csv"
    look_into_data(train_file)


if __name__ == "__main__":
    main()
