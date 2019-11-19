import random

import numpy as np
import torch
import torch.utils.data

import nn_layers
from scipy.io.wavfile import read
from text import text_to_sequence
from hyper_parameters import tacotron_params


class DataPreparation(torch.utils.data.Dataset):

    def __init__(self, audiopaths_and_text, tacotron_hyperparams):
        self.audiopaths_and_text = audiopaths_and_text
        self.audio_text_parameters = tacotron_hyperparams
        self.stft = nn_layers.TacotronSTFT(tacotron_hyperparams['filter_length'], tacotron_hyperparams['hop_length'],
                                           tacotron_hyperparams['win_length'], tacotron_hyperparams['n_mel_channels'],
                                           self.audio_text_parameters['sampling_rate'],
                                           tacotron_hyperparams['mel_fmin'], tacotron_hyperparams['mel_fmax'])
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def load_audiowav_torch(self, audiopath, samp_rate):
        sr, data = read(audiopath)
        assert samp_rate == sr, "Sample rate does not match with the configuration"

        return torch.FloatTensor(data.astype(np.float32))

    def melspec_textSequence_pair(self, audiopath_and_text):
        wav_path, sentence = audiopath_and_text[0], audiopath_and_text[1]
        # wav to torch tensor
        wav_torch = self.load_audiowav_torch(wav_path, self.audio_text_parameters['sampling_rate'])
        wav_torch_norm = wav_torch / self.audio_text_parameters['max_wav_value']
        wav_torch_norm = wav_torch_norm.unsqueeze(0)
        wav_torch_norm = torch.autograd.Variable(wav_torch_norm, requires_grad=False)
        mel_spec = self.stft.mel_spectrogram(wav_torch_norm)
        mel_spec = torch.squeeze(mel_spec, 0)
        # text to torch integer tensor sequence
        sentence_sequence = torch.IntTensor(text_to_sequence(sentence, self.audio_text_parameters['text_cleaners']))

        return sentence_sequence, mel_spec

    def __getitem__(self, index):
        return self.melspec_textSequence_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class DataCollate:

    def __init__(self, number_frames_step):
        self.number_frames_step = number_frames_step

    def __call__(self, batch):
        inp_lengths, sorted_decreasing = torch.sort(torch.LongTensor([len(x[0]) for x in batch]),
                                                   dim=0, descending=True)
        max_length_in = inp_lengths[0]

        # padding sentences sequences for a fixed-length tensor size
        sentences_padded = torch.LongTensor(len(batch), max_length_in)
        sentences_padded.zero_()
        for i in range(len(sorted_decreasing)):
            int_seq_sentence = batch[sorted_decreasing[i]][0]
            # all slots of a line until the end of the sentence. The rest, 0's
            sentences_padded[i, :int_seq_sentence.size(0)] = int_seq_sentence

        # length of the mel filterbank used
        num_melfilters = batch[0][1].size(0)

        # longest recorded spectrogram representation + 1 space to mark the end
        max_length_target = max([x[1].size(1) for x in batch])  # THERE IS A CHANGE FROM THE ORIGINAL CODE!!!
        # add extra space if the number of frames per step is higher than 1
        if max_length_target % self.number_frames_step != 0:
            max_length_target += self.number_frames_step - max_length_target % self.number_frames_step
            assert max_length_target % self.number_frames_step == 0

        # padding mel spectrogram representations. The output is a 3D tensor
        melspec_padded = torch.FloatTensor(len(batch), num_melfilters, max_length_target)
        melspec_padded.zero_()

        # GST new prosody matrices definition with zero padding:
        prosody_padded = torch.FloatTensor(len(batch), num_melfilters, max_length_target)
        prosody_padded.zero_()

        gate_padded = torch.FloatTensor(len(batch), max_length_target)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))

        for j in range(len(sorted_decreasing)):
            melspec = batch[sorted_decreasing[j]][1]
            melspec_padded[j, :, :melspec.size(1)] = melspec

            # GST filling padded prosody matrix:
            prosody_padded[j, :, :melspec.size(1)] = melspec

            gate_padded[j, melspec.size(1) - 1:] = 1
            output_lengths[j] = melspec.size(1)

        return sentences_padded, inp_lengths, melspec_padded, gate_padded, output_lengths, prosody_padded
