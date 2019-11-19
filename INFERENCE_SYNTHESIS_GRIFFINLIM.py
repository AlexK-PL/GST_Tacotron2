import time
t1 = time.time()
import torch
import numpy as np
from scipy.io.wavfile import write

import sys

from hyper_parameters import tacotron_params
from training import load_model
from text import text_to_sequence
from audio_processing import griffin_lim
from nn_layers import TacotronSTFT

torch.manual_seed(1234)

predicted_melspec_folder = '/homedtic/apeiro/GST_Tacotron2_ORIGINAL/Predicted_melspec/'
audio_path = '/homedtic/apeiro/GST_Tacotron2_ORIGINAL/Synth_wavs/' \
             'MelScaleLength_synth_1_78000steps_softmax_3tokens_1head_exp_decay'

extension = '.wav'

hparams = tacotron_params
MAX_WAV_VALUE = 32768.0

# load trained tacotron 2 model:
checkpoint_path = "outputs/checkpoint_78000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()


# preparing inputs for synthesis:
test_text = "That is not only my accusation."  # 1
# test_text = "Yet even so early as the death of the first Sir John Paul,"  # 2 
# test_text = "This was all the police wanted to know."  # 3
# test_text = "And there may be only nine."  # 4
# test_text = "He had here completed his ascent."  # 5
# test_text = "From defection to return to Fort Worth."  # 6
# test_text = "Yet the law was seldom if ever enforced."  # 7
# test_text = "The latter too was to be laid before the House of Commons."  # 8
# test_text = "Palmer speedily found imitators."  # 9
# test_text = "There were others less successful."  # 10

# test_text = "On December twenty-six, nineteen sixty-three, the FBI circulated additional instructions to all its 
# agents,"
# test_text = "Communications in the motorcade."
# test_text = "Examination of the cartridge cases found on the sixth floor of the Depository Building"
# test_text = "each patrolman might be given a prepared booklet of instructions explaining what is expected of him. The Secret Service has expressed concern"
gst_head_scores = np.array([0.6, 0.23, 0.17])

# gst_head_scores = np.array([0.27, 0.27, 0.27])

# gst_head_scores[j] = 0.62
gst_scores = torch.from_numpy(gst_head_scores)
gst_scores = torch.autograd.Variable(gst_scores).cuda().float()

sequence = np.array(text_to_sequence(test_text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

t = time.time()

# text2mel:
mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, gst_scores)

# save the predicted outputs from tacotron2:
mel_outputs_path = predicted_melspec_folder + "output.pt"
mel_outputs_postnet_path = predicted_melspec_folder + "output_postnet.pt"
alignments_path = predicted_melspec_folder + "alignment.pt"
print(mel_outputs_postnet.size())
torch.save(mel_outputs, mel_outputs_path)
torch.save(mel_outputs_postnet, mel_outputs_postnet_path)
torch.save(alignments, alignments_path)

print("text2mel prediction successfully performed...")

save_path = audio_path + extension

# Griffin Lim vocoder synthesis:
griffin_iters = 60
taco_stft = TacotronSTFT(hparams['filter_length'], hparams['hop_length'], hparams['win_length'], sampling_rate=hparams['sampling_rate'])
mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
mel_decompress = mel_decompress.transpose(1, 2).data.cpu()

spec_from_mel_scaling = 1000
spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
spec_from_mel = spec_from_mel * spec_from_mel_scaling

audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, griffin_iters)

audio = audio.squeeze()
audio_numpy = audio.cpu().numpy()

write(save_path, 22050, audio_numpy)

elapsed = time.time() - t
total_elapsed = time.time() - t1
print(total_elapsed)
print(elapsed)
print("mel2audio synthesis successfully performed.")

