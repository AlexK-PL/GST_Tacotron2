import time
t1 = time.time()
import torch
import numpy as np
from scipy.io.wavfile import write

import sys

sys.path.append('waveglow/')

from hyper_parameters import tacotron_params
from training import load_model
from text import text_to_sequence

torch.manual_seed(1234)

predicted_melspec_folder = './Predicted_melspec/'
# audio_path = '/homedtic/apeiro/GST_Tacotron2_prosody_dense_synthesis/Synth_wavs/synth_wav_
# 40500steps_second_02_fourth_05.wav'
audio_path = '/homedtic/apeiro/GST_Tacotron2_ORIGINAL/Synth_wavs/' \
             'MelScaleLength_synth_1_78000steps_softmax_3tokens_1head_exp_decay_waveglow'

extension = '.wav'

hparams = tacotron_params
MAX_WAV_VALUE = 32768.0

# load trained tacotron 2 model:
checkpoint_path = "outputs/checkpoint_78000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
model.to('cuda')
_ = model.eval()

# load pre trained waveglow model for mel2audio:
waveglow_path = 'waveglow/waveglow_old.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
# waveglow.cuda()

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
# test_text = "each patrolman might be given a prepared booklet of instructions explaining what is expected of him.
# The Secret Service has expressed concern"
gst_head_scores = np.array([0.6, 0.23, 0.17])

# gst_head_scores = np.array([0.27, 0.27, 0.27])

# gst_head_scores[j] = 0.62
gst_scores = torch.from_numpy(gst_head_scores)
gst_scores = torch.autograd.Variable(gst_scores).cuda().float()

sequence = np.array(text_to_sequence(test_text, ['english_cleaners']))[None, :]
# sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

t = time.time()

save_path = audio_path + extension

with torch.no_grad():
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
    print("text2mel prediction generated successfully...")
    # mel2wav
    audio = MAX_WAV_VALUE * waveglow.infer(mel_outputs_postnet, sigma=0.666)[0]

# audio = audio.cpu().numpy()
# audio = audio.astype('int16')
audio_numpy = audio[0].data.cpu().numpy()
write(save_path, 22050, audio)

elapsed = time.time() - t
total_elapsed = time.time() - t1
print(total_elapsed)
print(elapsed)
print("mel2audio synthesis successfully performed.")

