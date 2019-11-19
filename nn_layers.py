import torch
from torch import nn
from librosa.filters import mel as librosa_mel_fn
from stft import STFT

clip_val = 1e-5
C = 1


class convolutional_module(nn.Module):
    """This class defines a 1d convolutional layer and its initialization for the system we are
    replicating"""
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=None, dilation=1, bias=True,
                 w_init_gain='linear'):
        # in PyTorch you define your Models as subclasses of torch.nn.Module
        super(convolutional_module, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        # initialize the convolutional layer which is an instance of Conv1d
        # torch.nn.Conv1d calls internally the method torch.nn.functional.conv1d, which accepts the
        # input with the shape (minibatch x in_channels x input_w), and a weight of shape
        # (out_channels x (in_channels/groups) x kernel_w). In our case, we do not split into groups.
        # Then, our input shape will be (48 x 512 x 189) and the weights are set up as
        # (512 x 512 x 5)
        self.conv_layer = torch.nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation, bias=bias)

        """Useful information of Xavier initialization in:
        https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/"""
        torch.nn.init.xavier_uniform_(self.conv_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        conv_output = self.conv_layer(x)
        return conv_output


class linear_module(torch.nn.Module):
    """This class defines a linear layer and its initialization method for the system we are
    replicating. This implements a linear transformation: y = xA^t + b"""
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(linear_module, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class location_layer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(location_layer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        """We are being very restricting without training a bias"""
        """I think in_channels = 2 is k (number of vectors for every encoded stage position from prev.
        alignment)."""
        self.location_conv = convolutional_module(2, attention_n_filters, kernel_size=attention_kernel_size,
                                                  padding=padding, bias=False, stride=1, dilation=1)
        self.location_dense = linear_module(attention_n_filters, attention_dim, bias=False,
                                            w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class TacotronSTFT(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_de_normalize(self, magnitudes):
        output = torch.exp(magnitudes) / C
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = torch.log(torch.clamp(mel_output, min=clip_val) * C)
        return mel_output
