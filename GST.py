import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class GST(nn.Module):

    def __init__(self, hyper_parameters):

        super().__init__()
        self.prosody_extractor = LogMelSpecReferenceEncoder()
        self.stl = MultiSTL(hyper_parameters=hyper_parameters)

    def forward(self, logmel_spec, logmel_lengths):
        prosody_features_embedded = self.prosody_extractor(logmel_spec, logmel_lengths)  # [N, 512]
        style_embed, gst_scores = self.stl(prosody_features_embedded)

        return style_embed, gst_scores

    def inference(self, scores):  # NEED TO DEFINE SCORES TENSOR DIMENSION!!
        style_embed_inference = self.stl.inference(scores=scores)

        return style_embed_inference


class PitchContourEncoder(nn.Module):
    """

    """
    def __init__(self, hyper_parameters):

        super().__init__()

        K = len(hyper_parameters['ref_enc_out_channels'])
        filters = [1] + hyper_parameters['ref_enc_out_channels']
        kernel_sizes = hyper_parameters['seq_ref_enc_filter_size']

        convs_2d = []

        for i in range(K):
            conv2d_init = nn.Conv2d(in_channels=filters[i], out_channels=filters[i + 1],
                                    kernel_size=(kernel_sizes[i], 3), stride=(1, 1),
                                    padding=(int((kernel_sizes[i] - 1) / 2), int((3 - 1) / 2)), bias=True)

            nn.init.xavier_uniform_(conv2d_init.weight, gain=torch.nn.init.calculate_gain('linear'))

            convs_2d.append(conv2d_init)

        self.convs2D = nn.ModuleList(convs_2d)

        self.bns2D = nn.ModuleList([nn.BatchNorm2d(num_features=hyper_parameters['ref_enc_out_channels'][i])
                                    for i in range(K)])

        # WEIGHT INITIALIZATION DEFAULT:
        self.prosody_bi_lstm = nn.LSTM(input_size=int(176), hidden_size=int(512/2), num_layers=1, batch_first=True,
                                       bidirectional=True)

    def forward(self, bin_locations):  # [N, BIN_SUBAND, LEN_MELSPEC] (BIN_SUBAND = 13)
        N = bin_locations.size(0)  # Number of samples
        # Changing tensor dimensions to have 1 input channel for the first conv2D layer:
        bin_locations = bin_locations.unsqueeze(1)
        bin_locations = bin_locations.transpose(2, 3)  # [N, 1, LEN_MELSPEC, BIN_SUBAND]
        """We implement ReLU gates at the output of Conv. layers. We could check it without"""
        # For pitch tracking:
        for conv2, bn2 in zip(self.convs2D, self.bns2D):
            bin_locations = conv2(bin_locations)
            bin_locations = bn2(bin_locations)
            bin_locations = F.dropout(F.relu(bin_locations), 0.5, self.training)  # [N, Cout, LEN_MELSPEC, BIN_SUBAND]

        # Resize:
        bin_locations = bin_locations.transpose(1, 2)  # [N, LEN_MELSPEC, Cout, BIN_SUBAND]
        T = bin_locations.size(1)
        bin_locations = bin_locations.contiguous().view(N, T, -1)  # [N, LEN_MELSPEC, Cout*BIN_SUBAND]

        # Encode sequences into a bidirectional LSTM layer:
        """In our case, we do not care about the specific length of each sequence, as with the zero padding the encoder
        should be able to also encode the different lengths and see zero when its over. That is why we do not apply
        a packing padded sequence before LSTM layer."""
        _, (encoded_prosody, cell_state) = self.prosody_bi_lstm(bin_locations)

        encoded_prosody = encoded_prosody.transpose(0, 1)
        encoded_prosody = encoded_prosody.contiguous().view(N, -1)

        return encoded_prosody  # should be [N, 512]


# DENSE GST Reference Encoder:
class ProsodyEncoder(nn.Module):
    """
    This convolution class nn.Module performs two parallel convolution stacks, 1-D conv. and another 2-D conv.
    Afterwards, the output of both will be concatenated to be passed, later, through a bidirectional LSTM layer.
    """
    def __init__(self, hyper_parameters):

        super().__init__()

        K = len(hyper_parameters['ref_enc_out_channels'])
        filters = [1] + hyper_parameters['ref_enc_out_channels']
        kernel_sizes = hyper_parameters['seq_ref_enc_filter_size']

        # I NEED TO ADJUST PADDING TO NOT LOSE THE TOTAL LENGTH OF SEQUENCE!!
        convs_1d = []
        convs_2d = []

        for i in range(K):
            conv1d_init = nn.Conv1d(in_channels=filters[i], out_channels=filters[i + 1],
                                    kernel_size=kernel_sizes[i], stride=1,
                                    padding=int((kernel_sizes[i] - 1) / 2), bias=True)

            nn.init.xavier_uniform_(conv1d_init.weight, gain=torch.nn.init.calculate_gain('linear'))

            convs_1d.append(conv1d_init)

            conv2d_init = nn.Conv2d(in_channels=filters[i], out_channels=filters[i + 1],
                                    kernel_size=(kernel_sizes[i], 3), stride=(1, 1),
                                    padding=(int((kernel_sizes[i] - 1) / 2), int((3 - 1) / 2)), bias=True)

            nn.init.xavier_uniform_(conv2d_init.weight, gain=torch.nn.init.calculate_gain('linear'))

            convs_2d.append(conv2d_init)

        self.convs1D = nn.ModuleList(convs_1d)
        self.convs2D = nn.ModuleList(convs_2d)

        self.bns1D = nn.ModuleList([nn.BatchNorm1d(num_features=hyper_parameters['ref_enc_out_channels'][i])
                                    for i in range(K)])
        self.bns2D = nn.ModuleList([nn.BatchNorm2d(num_features=hyper_parameters['ref_enc_out_channels'][i])
                                    for i in range(K)])

        self.prosody_linear = nn.Linear(512, 256, bias=True)
        torch.nn.init.xavier_uniform_(self.prosody_linear.weight, gain=torch.nn.init.calculate_gain('linear'))

        # WEIGHT INITIALIZATION DEFAULT:
        self.prosody_bi_lstm = nn.LSTM(input_size=int(256), hidden_size=int(512/2), num_layers=1, batch_first=True,
                                       bidirectional=True)

    def forward(self, bin_locations, pitch_intensities):  # [N, LEN_MELSPEC, 1], [N, LEN_MELSPEC, 3]
        N = bin_locations.size(0)  # Number of samples
        num_intensities = pitch_intensities.size(2)
        # Changing tensor dimensions to have 1 input channel for the first conv2D layer:
        pitch_intensities = pitch_intensities.view(N, 1, -1, num_intensities)  # [N, 1, LEN_MELSPEC, num_intensities]
        bin_locations = bin_locations.transpose(1, 2)  # [N, 1, LEN_MELSPEC]
        """We implement ReLU gates at the output of Conv. layers. We could check it without"""
        # For pitch tracking:
        for conv, bn in zip(self.convs1D, self.bns1D):
            bin_locations = conv(bin_locations)
            bin_locations = bn(bin_locations)
            bin_locations = F.dropout(F.relu(bin_locations), 0.5, self.training)  # [N, Cout, T]

        # For pitch intensities:
        for conv2, bn2 in zip(self.convs2D, self.bns2D):
            pitch_intensities = conv2(pitch_intensities)
            pitch_intensities = bn2(pitch_intensities)
            pitch_intensities = F.dropout(F.relu(pitch_intensities), 0.5, self.training)  # [N, Cout, T, bins]

        # Resize pitch intensities
        bin_locations = bin_locations.transpose(1, 2)  # [N, T, Cout]
        pitch_intensities = pitch_intensities.transpose(1, 2)  # [N, T, Cout, bins]
        T = pitch_intensities.size(1)
        pitch_intensities = pitch_intensities.contiguous().view(N, T, -1)  # [N, T, Cout*bins]

        # Concatenate features
        pitch_convolved = torch.cat((bin_locations, pitch_intensities), 2)

        # Linear projection (IS IT NECESSARY? DOES ACTIVATION FUNCTION IMPROVE THE RESULT?)
        projection_pitch_convolved = F.dropout(F.tanh(self.prosody_linear(pitch_convolved)), 0.5, self.training)

        # Encode sequences into a bidirectional LSTM layer:
        """In our case, we do not care about the specific length of each sequence, as with the zero padding the encoder
        should be able to also encode the different lengths and see zero when its over. That is why we do not apply
        a packing padded sequence before LSTM layer."""
        _, (encoded_prosody, cell_state) = self.prosody_bi_lstm(projection_pitch_convolved)

        encoded_prosody = encoded_prosody.transpose(0, 1)
        encoded_prosody = encoded_prosody.contiguous().view(N, -1)

        return encoded_prosody  # should be [N, 512]


class LogMelSpecReferenceEncoder(nn.Module):
    """
    """
    def __init__(self):

        super().__init__()

        reference_encoder_out_channels = [32, 32, 64, 64, 128, 128]
        K = len(reference_encoder_out_channels)
        filters = [1] + reference_encoder_out_channels
        kernel_size = (3, 3)
        stride = (2, 2)
        padding = (1, 1)

        convs_2d = []

        for i in range(K):
            conv2d_init = nn.Conv2d(in_channels=filters[i], out_channels=filters[i + 1],
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=True)

            nn.init.xavier_uniform_(conv2d_init.weight, gain=torch.nn.init.calculate_gain('linear'))

            convs_2d.append(conv2d_init)

        self.convs2D = nn.ModuleList(convs_2d)
        self.bns2D = nn.ModuleList([nn.BatchNorm2d(num_features=reference_encoder_out_channels[i])
                                    for i in range(K)])

        out_channels = self.calculate_channels(80, 3, 2, 1, K)
        # self.gru = nn.GRU(input_size=reference_encoder_out_channels[-1] * out_channels, hidden_size=512,
        #                   batch_first=True, bidirectional=False)

        # WEIGHT INITIALIZATION DEFAULT:
        self.bi_lstm = nn.LSTM(input_size=reference_encoder_out_channels[-1] * out_channels,
                               hidden_size=int(512/2), num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, logmel_spec, logmel_lengths):  # [N, MEL_CHANNELS, LEN_MELSPEC]
        N = logmel_spec.size(0)  # Number of samples
        # Changing tensor dimensions to have 1 input channel for the first conv2D layer:
        logmel_spec = logmel_spec.unsqueeze(1)
        logmel_spec = logmel_spec.transpose(2, 3)  # [N, 1, LEN_MELSPEC, MEL_CHANNELS]
        """We implement ReLU gates at the output of Conv. layers. We could check it without"""
        for conv2, bn2 in zip(self.convs2D, self.bns2D):
            logmel_spec = conv2(logmel_spec)
            logmel_spec = bn2(logmel_spec)
            logmel_spec = F.dropout(F.relu(logmel_spec), 0.5, self.training)  # [N, Cout, LEN_MELSPEC, BIN_SUBAND]

        # Resize:
        logmel_spec = logmel_spec.transpose(1, 2)  # [N, LEN_MELSPEC, Cout, MEL_CHANNELS]
        T = logmel_spec.size(1)
        logmel_spec = logmel_spec.contiguous().view(N, T, -1)  # [N, LEN_MELSPEC, Cout*BIN_SUBAND]

        logmel_lengths = logmel_lengths.cpu().numpy()
        last_hidden_states = torch.zeros(N, 512)

        logmel_after_lengths = np.trunc(logmel_lengths / 2**6)
        logmel_after_lengths = logmel_after_lengths + 1
        logmel_after_lengths = logmel_after_lengths.astype(int)
        logmel_after_lengths = torch.tensor(logmel_after_lengths)
        # logmel_spec = nn.utils.rnn.pack_padded_sequence(logmel_spec, logmel_after_lengths, batch_first=True)
        self.bi_lstm.flatten_parameters()
        # memory, out = self.gru(logmel_spec)
        outputs, (hidden_states, cell_state) = self.bi_lstm(logmel_spec)
        hidden_states = hidden_states.transpose(0, 1)
        hidden_states = hidden_states.contiguous().view(N, -1)
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # for j in range(N):
        #     last_hidden_states[j, :] = outputs[j, logmel_after_lengths[j] - 1, :]

        # return last_hidden_states.cuda(non_blocking=True)
        return hidden_states

    def calculate_channels(self, L, kernel_size, stride, padding, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * padding) // stride + 1
        return L


# BASIC FORM FOR NOW. NEEDS TO BE EXPANDED TO OUR NEW PROPOSAL
class MultiSTL(nn.Module):

    """
    inputs --- [N, E]
    """

    def __init__(self, hyper_parameters):

        super().__init__()
        # E = 256 / num_heads = 8 / token_num = 10!!
        self.embed = nn.Parameter(torch.FloatTensor(hyper_parameters['token_num'],
                                                    hyper_parameters['E'] // hyper_parameters['num_heads']))
        # d_q = hyper_parameters['E'] // 2
        d_q = hyper_parameters['E']
        d_k = hyper_parameters['E'] // hyper_parameters['num_heads']

        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k,
                                            num_units=hyper_parameters['E'], num_heads=hyper_parameters['num_heads'])

        init.xavier_uniform_(self.embed, gain=init.calculate_gain('linear'))

    def forward(self, inputs):
        N = inputs.size(0)  # Number of samples in the batch
        query = inputs.unsqueeze(1)  # [N, 1, E]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed, gst_scores = self.attention(query, keys)

        return style_embed, gst_scores

    def inference(self, scores):
        keys = F.tanh(self.embed).unsqueeze(0)
        style_embed_inference = self.attention.inference(keys, scores=scores)

        return style_embed_inference


class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]  T_q = 1
        key --- [N, T_k, key_dim]  T_k = 5 (num of tokens)
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        #self.sparse_max = Sparsemax(dim=3)

        # Linear projection of data (encoder and decoder states) into a fixed number of hidden units
        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):

        querys = self.W_query(query)  # [N, T_q, num_units] the last dimension changes according to the output dim
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        # the number of units set at the initialization is the total of hidden feature units we want. Then, we will
        # assign a specific number of num_units according to the number of heads of the multi head Attention.

        # Basically, style tokens are the number of heads we configure to learn different types of attention
        #
        split_size = self.num_units // self.num_heads  # integer division, without remainder
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.33)  # cube root instead of square to prevent very small values
        scores = F.softmax(scores, dim=3)  # From dimension 3, length of Key sequences.
        # scores = self.sparse_max(scores)
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        scores = scores.squeeze()

        return out, scores

    def inference(self, key, scores):  # key [1, 5, 512/8] # [1, num_tokens]
        """Only need the keys that are already trained, and the scores that I impose"""
        scores = scores.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(self.num_heads, -1, -1, -1)
        # print(scores.shape)
        values = self.W_value(key)

        # the number of units set at the initialization is the total of hidden feature units we want. Then, we will
        # assign a specific number of num_units according to the number of heads of the multi head Attention.

        # Basically, style tokens are the number of heads we configure to learn different types of attention
        #
        split_size = self.num_units // self.num_heads  # integer division, without remainder
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))

        # out = score * V
        out = torch.matmul(scores, values)  # [h, 1, T_q = 1, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
