import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size=50000, emb_dim=100, emb_vectors=None,
                 lstm_dim=256, lstm_n_layer=2, lstm_dropout=0.1,
                 bidirectional=True,
                 n_linear=2, n_classes=1, crit=nn.CrossEntropyLoss()):
        super().__init__()
        vocab_size, emb_dim = emb_vectors.shape
        self.lstm_n_layer = lstm_n_layer
        self.lstm_dim = lstm_dim
        n_dirs = bidirectional + 1
        self.n_dirs = n_dirs

        self.embedding_layer = nn.Embedding(*emb_vectors.shape)
        self.embedding_layer.from_pretrained(emb_vectors,
                                             padding_idx=1)  # pad=1 in torchtext

        self.lstm = nn.LSTM(emb_dim, lstm_dim // n_dirs, num_layers=lstm_n_layer,
                            bidirectional=bidirectional, dropout=lstm_dropout,
                            batch_first=True)
        self.linear_layers = [nn.Linear(lstm_dim, lstm_dim) for _ in
                              range(n_linear - 1)]
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.label = nn.Linear(lstm_dim, n_classes)
        self.crit = crit

        self.opts = {
            'vocab_size': vocab_size,
            'emb_dim': emb_dim,
            'emb_vectors': emb_vectors,
            'lstm_dim': lstm_dim,
            'lstm_n_layer': lstm_n_layer,
            'lstm_dropout': lstm_dropout,
            'n_linear': n_linear, 'n_classes': n_classes,
            'crit': crit,
        }

    def attention_net(self, lstm_output, final_state):
        """
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.

        Arguments
        ---------

        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM

        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.

        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)

        """
        attn_weights = torch.bmm(lstm_output, final_state.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1).unsqueeze(2) # shape = (batch_size, seq_len, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2),
                                     soft_attn_weights).squeeze(2)

        return new_hidden_state

    def forward(self, input):
        inp = self.embedding_layer(input)
        batch_size = len(inp)
        output, (final_h, final_c) = self.lstm(inp)
        final_h = final_h.view(self.lstm_n_layer, self.n_dirs, batch_size, self.lstm_dim // self.n_dirs)[-1]
        final_h = final_h.permute(1, 0, 2)  # (batch_size, 2, -1)
        final_h = final_h.contiguous().view(batch_size, self.lstm_dim)

        # final_h.size() = (batch_size, hidden_size)
        # output.size() = (batch_size, num_seq, hidden_size)

        attn_output = self.attention_net(output, final_h)

        output = output[:, -1]
        for layer in self.linear_layers:
            output = layer(output)

        logits = self.label(output)
        return logits

    def loss(self, input, target):
        logits = self.forward(input)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        loss = self.crit(logits_flat, target_flat)  # mean_score per batch
        return loss

    def predict(self, input):
        logits = self.forward(input)
        preds = logits.max(dim=-1)[1]
        preds = preds.detach().cpu().numpy()
        return preds

    def loss_n_acc(self, input, target):
        logits = self.forward(input)
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target.view(-1)
        loss = self.crit(logits_flat, target_flat)  # mean_score per batch

        pred_flat = logits_flat.max(dim=-1)[1]
        acc = (pred_flat == target).sum()
        return loss, acc