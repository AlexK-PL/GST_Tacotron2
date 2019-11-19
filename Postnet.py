import torch
from torch import nn
from torch.nn import functional as F
from nn_layers import convolutional_module


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, tacotron_hyperparams):
        super(Postnet, self).__init__()
        #  self.dropout = nn.Dropout(0.5)
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                convolutional_module(tacotron_hyperparams['n_mel_channels'],
                                     tacotron_hyperparams['postnet_embedding_dim'],
                         kernel_size=tacotron_hyperparams['postnet_kernel_size'], stride=1,
                         padding=int((tacotron_hyperparams['postnet_kernel_size'] - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(tacotron_hyperparams['postnet_embedding_dim']))
        )

        for i in range(1, tacotron_hyperparams['postnet_n_convolutions'] - 1):
            self.convolutions.append(
                nn.Sequential(
                    convolutional_module(tacotron_hyperparams['postnet_embedding_dim'],
                             tacotron_hyperparams['postnet_embedding_dim'],
                             kernel_size=tacotron_hyperparams['postnet_kernel_size'], stride=1,
                             padding=int((tacotron_hyperparams['postnet_kernel_size'] - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(tacotron_hyperparams['postnet_embedding_dim']))
            )

        self.convolutions.append(
            nn.Sequential(
                convolutional_module(tacotron_hyperparams['postnet_embedding_dim'],
                                     tacotron_hyperparams['n_mel_channels'],
                         kernel_size=tacotron_hyperparams['postnet_kernel_size'], stride=1,
                         padding=int((tacotron_hyperparams['postnet_kernel_size'] - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(tacotron_hyperparams['n_mel_channels']))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x
