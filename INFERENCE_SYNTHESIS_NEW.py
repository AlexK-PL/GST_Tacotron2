import torch
import numpy as np
from scipy.io.wavfile import write
from hyper_parameters import tacotron_params
from training import load_model
from text import text_to_sequence

import sys

sys.path.append('waveglow/')

predicted_melspec_folder = '/homedtic/apeiro/GST_Tacotron2_ORIGINAL/Predicted_melspec/'
audio_path = '/homedtic/apeiro/GST_Tacotron2_ORIGINAL/Synth_wavs/'
common_clip_name = '_78000steps_softmax_3tokens_'

hparams = tacotron_params
MAX_WAV_VALUE = 32768.0

# load trained tacotron 2 model:
checkpoint_path = "outputs/checkpoint_78000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()

# load pre trained waveglow model for mel2audio:
waveglow_path = 'waveglow/waveglow_old.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda()

# short sentences:

test_text_short = ("That is not only my accusation.",
                   "provided by other agencies.",
                   "There were others less successful.")

# medium sentences:

test_text_medium = ("Yet even so early as the death of the first Sir John Paul.",
                    "The proceeds of the robbery were lodged in a Boston bank,",
                    "in the other was the sacred precinct of Jupiter Belus,")

# large sentences:

test_text_large = ("Two weeks pass, and at last you stand on the eastern edge of the plateau",
                   "These disciplinary improvements were, however, only slowly and gradually introduced.",
                   "A committee was appointed, under the presidency of the Duke of Richmond")


gst_head_scores = np.array([[0.25, 0.25, 0.5], [0.19, 0.71, 0.1], [0.58, 0.01, 0.41], [0.50, 0.01, 0.49],
                            [0.22, 0.3, 0.48], [0.18, 0.41, 0.41]])
gst_head_names = ("FirstSecondEqual", "Arrow", "SemiBlackHole", "BlackHole", "LinearAscendingOrder",
                  "SecondThirdEqual")
test_text_lengths = ("short", "medium", "large")

for j in range(6):

    gst_scores = torch.from_numpy(gst_head_scores[j])
    gst_scores = torch.autograd.Variable(gst_scores).cuda().float()
    gst_name = gst_head_names[j]  # is a string

    for i in range(3):

        test_short = test_text_short[i]
        test_medium = test_text_medium[i]
        test_large = test_text_large[i]
        tests_aux = (test_short, test_medium, test_large)
        
        for k in range(3):

            sequence = np.array(text_to_sequence(tests_aux[k], ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            # text2mel:
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, gst_scores)
        
            # save the predicted outputs from tacotron2:
            mel_outputs_path = predicted_melspec_folder + "output.pt"
            mel_outputs_postnet_path = predicted_melspec_folder + "output_postnet.pt"
            alignments_path = predicted_melspec_folder + "alignment.pt"
            torch.save(mel_outputs, mel_outputs_path)
            torch.save(mel_outputs_postnet, mel_outputs_postnet_path)
            torch.save(alignments, alignments_path)
        
            print("text2mel prediction successfully performed...")
        
            save_path = audio_path + "/" + gst_name + "/" + test_text_lengths[k] + "_" + str(i + 1) + \
                        common_clip_name + gst_name + ".wav"
        
            with torch.no_grad():
                audio = MAX_WAV_VALUE * waveglow.infer(mel_outputs_postnet, sigma=0.666)[0]
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
            write(save_path, 22050, audio)
        
            print("mel2audio synthesis successfully performed.")
