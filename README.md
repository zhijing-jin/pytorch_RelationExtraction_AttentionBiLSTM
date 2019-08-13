This is an example of NLP classification on Kaggle dataset challenge.

## Step 1. Take a deep breath.
Do look into the data. At least read through 100 examples. Run `python look_into_data.py`.
- Understand the Input and Output.
- Find some signals for negative and positive samples.
- When programming, don't bother to look back at the data format again.

The dataset is [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/12500/download-all).

## Step 2. Dataloading 
```bash
./prepare_env.sh
python preprocess.py
```
## Step 3. Train and Tune
```bash
python main.py --data_train train.full.csv
```
- Get some feeling of how well the model trains.