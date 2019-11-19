import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from nn_layers import linear_module, location_layer
from utils import get_mask_from_lengths


class AttentionNet(nn.Module):
    # 1024, 512, 128, 32, 31
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(AttentionNet, self).__init__()
        self.query_layer = linear_module(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        # Projecting inputs into 128-D hidden representation
        self.memory_layer = linear_module(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        # Projecting into 1-D scalar value
        self.v = linear_module(attention_dim, 1, bias=False)
        # Convolutional layers to obtain location features and projecting them into 128-D hidden representation
        self.location_layer = location_layer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1) # eliminates the third dimension of the tensor, which is 1.
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        # I think attention_weights is a [BxNUMENCINPUTS] so with unsequeeze(1): [Bx1xNUMENCINPUTS] and memory is
        # [BxNUMENCINPUTSx512]
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1] # all list values but the last one. The result is a list of the in_dim element
        # concatenated with sizes of layers (i.e. [80, 256])
        self.layers = nn.ModuleList(
            [linear_module(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Decoder(nn.Module):
    def __init__(self, tacotron_hyperparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = tacotron_hyperparams['n_mel_channels']
        self.n_frames_per_step = tacotron_hyperparams['number_frames_step']
        self.encoder_embedding_dim = tacotron_hyperparams['encoder_embedding_dim']
        self.attention_rnn_dim = tacotron_hyperparams['attention_rnn_dim'] # 1024
        self.decoder_rnn_dim = tacotron_hyperparams['decoder_rnn_dim'] # 1024
        self.prenet_dim = tacotron_hyperparams['prenet_dim']
        self.max_decoder_steps = tacotron_hyperparams['max_decoder_steps']
        # The threshold to decide whether stop or not stop decoding?
        self.gate_threshold = tacotron_hyperparams['gate_threshold']
        self.p_attention_dropout = tacotron_hyperparams['p_attention_dropout']
        self.p_decoder_dropout = tacotron_hyperparams['p_decoder_dropout']
        # Define the prenet: there is only one frame per step, so input dim is the number of mel channels.
        # There are two fully connected layers:
        self.prenet = Prenet(
            tacotron_hyperparams['n_mel_channels'] * tacotron_hyperparams['number_frames_step'],
            [tacotron_hyperparams['prenet_dim'], tacotron_hyperparams['prenet_dim']])
        # input_size: 1024 + 512 (output of first LSTM cell + attention_context) / hidden_size: 1024
        self.attention_rnn = nn.LSTMCell(
            tacotron_hyperparams['prenet_dim'] + tacotron_hyperparams['encoder_embedding_dim'],
            tacotron_hyperparams['attention_rnn_dim'])
        # return attention_weights and attention_context. Does the alignments.
        self.attention_layer = AttentionNet(
            tacotron_hyperparams['attention_rnn_dim'], tacotron_hyperparams['encoder_embedding_dim'],
            tacotron_hyperparams['attention_dim'], tacotron_hyperparams['attention_location_n_filters'],
            tacotron_hyperparams['attention_location_kernel_size'])
        # input_size: 256 + 512 (attention_context + prenet_info), hidden_size: 1024
        self.decoder_rnn = nn.LSTMCell(
            tacotron_hyperparams['attention_rnn_dim'] + tacotron_hyperparams['encoder_embedding_dim'],
            tacotron_hyperparams['decoder_rnn_dim'], 1)
        # (LSTM output)1024 + (attention_context)512, out_dim: number of mel channels. Last linear projection that
        # generates an output decoder spectral frame.
        self.linear_projection = linear_module(
            tacotron_hyperparams['decoder_rnn_dim'] + tacotron_hyperparams['encoder_embedding_dim'],
            tacotron_hyperparams['n_mel_channels']*tacotron_hyperparams['number_frames_step'])
        # decision whether to continue decoding.
        self.gate_layer = linear_module(
            tacotron_hyperparams['decoder_rnn_dim'] + tacotron_hyperparams['encoder_embedding_dim'], 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        # reshape decoder inputs in case we want to work with more than 1 frame per step (chunks). Otherwise, this next
        # line does not just do anything
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        # concatenates [Bx1024] and [Bx512]. All dimensions match except 1 (torch.cat -1)
        # concatenate the i-th decoder hidden state together with the i-th attention context
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        # the previous input is for the following LSTM cell, initialized with zeroes the hidden states and the cell
        # state.
        # compute the (i+1)th attention hidden state based on the i-th decoder hidden state and attention context.
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(self.attention_cell, self.p_attention_dropout, self.training)
        # concatenate the i-th state attention weights together with the cumulated from previous states to compute
        # (i+1)th state
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        # compute (i+1)th attention context and provide (i+1)th attention weights based on the (i+1)th attention hidden
        # state and (i)th and prev. weights
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        # cumulate attention_weights adding the (i+1)th to compute (i+2)th state
        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input,
                                                                  (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(self.decoder_cell, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, self.attention_weights

        """
        # the decoder_output from ith step passes through the pre-net to compute new decoder hidden state and attention_
        # context (i+1)th
        prenet_output = self.prenet(decoder_input)
        # the decoder_input now is the concatenation of the pre-net output and the new (i+1)th attention_context
        decoder_input = torch.cat((prenet_output, self.attention_context), -1)
        # another LSTM Cell to compute the decoder hidden (i+1)th state from the decoder_input
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))

        # with new attention_context we concatenate again with the new (i+1)th decoder_hidden state.
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        # the (i+1)th output is a linear projection of the decoder hidden state with a weight matrix plus bias.
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        # check whether (i+1)th state is the last of the sequence
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights"""

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []

        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            # a class list, when += means concatenation of vectors
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
            # getting the frame indexing from reference mel frames to pass it as the new input of the next decoding
            # step: Teacher Forcing!
            # Takes each time_step of sequences of all mini-batch samples (i.e. [48, 80] as the decoder_inputs is
            # parsed as [189, 48, 80]).

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments
