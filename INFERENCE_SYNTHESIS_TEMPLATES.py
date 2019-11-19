import torch
import numpy as np
from scipy.io.wavfile import write
from hyper_parameters import tacotron_params
import pandas as pd
from training import load_model
from text import text_to_sequence

import sys

sys.path.append('waveglow/')

predicted_melspec_folder = '/homedtic/apeiro/GST_Tacotron2_ORIGINAL/Predicted_melspec/'
audio_path = '/homedtic/apeiro/GST_Tacotron2_ORIGINAL/Synth_wavs/ATTENTION_WEIGHT_TEMPLATES/'
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
                   "There were others less successful.")

# medium sentences:

test_text_medium = ("Yet even so early as the death of the first Sir John Paul.",
                    "The proceeds of the robbery were lodged in a Boston bank,")

# large sentences:

test_text_large = ("Two weeks pass, and at last you stand on the eastern edge of the plateau",
                   "A committee was appointed, under the presidency of the Duke of Richmond")

'''
# short sentences:

test_text_short = ("three minutes after the shooting.",
                   "They agree that Hosty told Revill",)

# medium sentences:

test_text_medium = ("The latter too was to be laid before the House of Commons.",
                    "The assistant to the Director of the FBI testified that",)

# large sentences:

test_text_large = ("refuted by abundant evidence, and having no foundation whatever in truth.",
                   "Marina Oswald, however, recalled that her husband was upset by this interview.")
'''

csv_file = "/homedtic/apeiro/GST_Tacotron2_ORIGINAL/Attention_Weights_Template_Analysis.csv"
gst_head_scores = pd.read_csv(csv_file, index_col=False)
gst_head_scores = gst_head_scores.values
gst_head_scores = np.asarray(gst_head_scores)
NUM_TEMPLATES = len(gst_head_scores[:, 0])

test_text_lengths = ("short/", "medium/", "large/")

for j in range(NUM_TEMPLATES):

    gst_scores = gst_head_scores[j, :]
    gst_scores = torch.from_numpy(gst_scores)
    gst_scores = torch.autograd.Variable(gst_scores).cuda().float()

    for i in range(2):

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

            save_path = audio_path + test_text_lengths[k] + "sentence_" + str(i + 1) + common_clip_name \
                        + "attention_template_" + str(j + 1) + ".wav"

            with torch.no_grad():
                audio = MAX_WAV_VALUE * waveglow.infer(mel_outputs_postnet, sigma=0.666)[0]
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
            write(save_path, 22050, audio)

            print("mel2audio synthesis successfully performed.")
