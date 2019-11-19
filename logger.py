import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_gst_scores_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, gst_scores, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments, _ = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)

        align_idx = alignments[idx].data.cpu().numpy().T
        gst_scores = gst_scores.data.cpu().numpy().T
        # print("Validation GST scores before plotting to tensorboard: {}".format(gst_scores.shape))
        meltarg_idx = mel_targets[idx].data.cpu().numpy()
        melout_idx = mel_outputs[idx].data.cpu().numpy()

        self.add_image("alignment", plot_alignment_to_numpy(align_idx), iteration)
        self.add_image("gst_scores", plot_gst_scores_to_numpy(gst_scores), iteration)
        self.add_image("mel_target", plot_spectrogram_to_numpy(meltarg_idx), iteration)
        self.add_image("mel_predicted", plot_spectrogram_to_numpy(melout_idx), iteration)
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                F.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)
