""" Loss functions. """

import torch
import torch.nn as nn

from utils.general import angular_error


class ComputeLoss:

    # Compute losses
    def __init__(self, model):
        self.device = next(model.parameters()).device

        # Define criteria
        self.MSELoss = nn.MSELoss().to(self.device)
        self.L1Loss = nn.L1Loss()
        self.BCEangular = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.91, device=self.device))  # 二元交叉熵损失

    def __call__(self, gaze_pred, labels):
        sight = 1.0
        angular = 0
        gloss, aloss = torch.zeros(1, device=self.device), \
                              torch.zeros(1, device=self.device)
        gloss += self.L1Loss(gaze_pred[..., 0:2], labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(gaze_pred[..., 0:2]))
        angular_error_matrix = angular_error(gaze_pred[..., 0:2], labels)
        aloss += angular_error_matrix.mean()
        gloss *= sight
        aloss *= angular
        return gloss + aloss, torch.cat((gloss, aloss)).detach()

