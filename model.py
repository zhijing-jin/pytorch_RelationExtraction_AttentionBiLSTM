from torch import nn
import numpy as np


class SimpleBiLSTMBaseline(nn.Module):
    def __init__(self, vocab_size=10000, emb_dim=300,
                 hidden_dim=256, lstm_dropout=0.1, num_linear=1,
                 predict_dim=1, crit=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1,
                               dropout=lstm_dropout, batch_first=True)
        self.linear_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in
                              range(num_linear - 1)]
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, predict_dim)

        self.crit = crit
        import pdb;
        pdb.set_trace()
        self

    def forward(self, inp):
        output = self.embedding(inp)
        output, _ = self.encoder(output)
        output = output[:, -1]
        for layer in self.linear_layers:
            output = layer(output)
        output = self.predictor(output)
        output = output.squeeze()  # (batch_size,)
        return output

    def score(self, inp, tgt):
        preds = self.forward(inp)
        score = self.crit(preds, tgt)
        return score

    def predict(self, inp):
        preds = self.forward(inp)
        preds = preds.data.cpu().numpy()
        preds = 1 / (1 + np.exp(-preds))  # sigmoid
        return preds
