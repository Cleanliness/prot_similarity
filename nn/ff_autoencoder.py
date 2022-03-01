import torch
from torch import nn
from collections import OrderedDict

class Autoencoder(nn.Module):
    """feedforward autoencoder"""
    def __init__(self, bneck_size, enc_layers, dec_layers):
        super().__init__()
        seq_layers = []
        for i in range(0, len(enc_layers) - 1):
            seq_layers.append(("enc_ff " + str(i), nn.Linear(enc_layers[i], enc_layers[i + 1])))
            seq_layers.append(("enc_sigmoid " + str(i), nn.Sigmoid()))

        seq_layers.append(("bneck_out", nn.Linear(enc_layers[-1], bneck_size)))
        self.encoder = nn.Sequential(OrderedDict(seq_layers))

        seq_layers = [("bneck_out", nn.Linear(bneck_size, dec_layers[0]))]
        for i in range(0, len(enc_layers) - 1):
            seq_layers.append(("dec_ff " + str(i), nn.Linear(dec_layers[i], dec_layers[i + 1])))
            seq_layers.append(("dec_sigmoid " + str(i), nn.Sigmoid()))

        self.decoder = nn.Sequential(OrderedDict(seq_layers))

    def forward(self, nn_in):
        return self.decoder(self.encoder(nn_in))


h = Autoencoder(4, [7, 4], [4, 7])
k = h(torch.Tensor([5, 5, 5, 5, 5, 5, 5]))

hhh = 3
