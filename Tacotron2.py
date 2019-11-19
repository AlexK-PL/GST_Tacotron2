from math import sqrt

import torch
from torch import nn

from Encoder import Encoder
from Decoder import Decoder
from Postnet import Postnet
from GST import GST

from utils import to_gpu, get_mask_from_lengths
from fp16_optimizer import fp32_to_fp16, fp16_to_fp32


class tacotron_2(nn.Module):
    def __init__(self, tacotron_hyperparams):
        super(tacotron_2, self).__init__()
        self.mask_padding = tacotron_hyperparams['mask_padding']
        self.fp16_run = tacotron_hyperparams['fp16_run']
        self.n_mel_channels = tacotron_hyperparams['n_mel_channels']
        self.n_frames_per_step = tacotron_hyperparams['number_frames_step']
        self.embedding = nn.Embedding(
            tacotron_hyperparams['n_symbols'], tacotron_hyperparams['symbols_embedding_length'])
        # CHECK THIS OUT!!!
        std = sqrt(2.0 / (tacotron_hyperparams['n_symbols'] + tacotron_hyperparams['symbols_embedding_length']))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(tacotron_hyperparams)
        self.decoder = Decoder(tacotron_hyperparams)
        self.postnet = Postnet(tacotron_hyperparams)
        self.gst = GST(tacotron_hyperparams)

    def parse_batch(self, batch):
        # GST I add the new tensor from prosody features to train GST tokens:
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths, prosody_padded = batch
        text_padded = to_gpu(text_padded).long()
        max_len = int(torch.max(input_lengths.data).item())  # With item() you get the pure value (not in a tensor)
        input_lengths = to_gpu(input_lengths).long()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        prosody_padded = to_gpu(prosody_padded).float()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths, prosody_padded),
            (mel_padded, gate_padded))

    def parse_input(self, inputs):
        inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
        return inputs

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs

        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, output_lengths, gst_prosody_padded = self.parse_input(inputs)
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        # GST style embedding plus embedded_inputs before entering the decoder
        # bin_locations = gst_prosody_padded[:, 0, :]
        # pitch_intensities = gst_prosody_padded[:, 1:, :]
        # bin_locations = bin_locations.unsqueeze(2)
        gst_style_embedding, gst_scores = self.gst(gst_prosody_padded, output_lengths)  # [N, 512]
        gst_style_embedding = gst_style_embedding.expand_as(encoder_outputs)

        encoder_outputs = encoder_outputs + gst_style_embedding

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, gst_scores],
            output_lengths)

    def inference(self, inputs, gst_scores):  # gst_scores must be a torch tensor
        inputs = self.parse_input(inputs)
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        # GST inference:
        gst_style_embedding = self.gst.inference(gst_scores)
        gst_style_embedding = gst_style_embedding.expand_as(encoder_outputs)

        encoder_outputs = encoder_outputs + gst_style_embedding

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
