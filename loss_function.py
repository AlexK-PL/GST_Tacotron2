from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        # Ensures dimension 1 will be size 1, the rest can be adapted. It is a column of length 189 with all zeroes
        # till the end of the current sequence, which is filled with 1's
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, _ = model_output
        gate_out = gate_out.view(-1, 1)
        # Mean Square Error (L2) loss function for decoder generation + post net generation
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        # Binary Cross Entropy with a Sigmoid layer combined. It is more efficient than using a plain Sigmoid
        # followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp
        # trick for numerical stability
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
