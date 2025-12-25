import os
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.general import (
    angular,
    gazeto3d,
    gazeto3d_matrix,
    angular_matrix,
    LOGGER,
    mean_gaze
)
from utils.loss import ComputeLoss


def run(
        weights=None,  # model weights path
        batch_size=32,  # batch_size
        device="",  # cuda device, i.e. 0
        worker=8,  # max dataloader workers (per RANK in DDP mode)
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        alpha=1
):
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device
        half &= device.type != 'cpu'
        model.half() if half else model.float()


    model.eval()
    computeLoss = ComputeLoss(model)
    pbar = enumerate(dataloader)
    total = 0
    error = 0
    avg_error = []
    sum_loss = 0
    reg_criterion = nn.MSELoss().to(device)
    nb = len(dataloader)
    pbar = tqdm(pbar, total=nb, file=sys.stdout, desc=('%20s' + '%10s' * 3) % ('Obj', 'gloss', 'aloss', 'avg_error'))
    mloss = torch.zeros(2, device=device)
    with torch.no_grad():
        for i, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            gaze_pred = model(images)

            loss,  loss_items = computeLoss(gaze_pred, labels)

            sum_loss += loss

            bs, _, ny, nx, _ = gaze_pred.shape

            gaze_pred = gaze_pred.view(bs, -1)

            pred_3d = gazeto3d_matrix(gaze_pred)
            target_3d = gazeto3d_matrix(labels)
            angular_error = angular_matrix(pred_3d, target_3d)
            error = angular_error.mean().cpu().item()
            avg_error.append(error)
            mloss = (mloss * i + loss_items) / (i + 1)
        avg_loss = sum_loss / nb
    pf = '%20s' + '%10.4g' * 3
    print(pf % ("all", *mloss, sum(avg_error) / len(avg_error)))
    return sum(avg_error) / len(avg_error), avg_loss.cpu().item()


def parse_opt(known=False):
    """ Parse command-line arguments for validate """
    parser = argparse.ArgumentParser(description="Gaze estimation using EmodelNet")
    parser.add_argument("--gazeImage_dir", dest="gazeImage_dir", help="Directory path for gaze images.",
                        default="datasets/MPIIFaceGaze/Image")
    parser.add_argument("--gazeLabel_dir", dest="gazeLabel_dir", help="Directory path for gaze labels.",
                        default="datasets/MPIIFaceGaze/Label")
    parser.add_argument("--dataset", dest="dataset",
                        default="MpiigazeKFold", type=str)
    parser.add_argument("--output", dest="output", help="Path for the output evaluating gaze test.",
                        default="output/test", type=str
                        )
    parser.add_argument("--num_bins", dest="num_bins", default=28, type=int)
    parser.add_argument("--angle", dest="angle", default=42)
    parser.add_argument("--device", dest="device", help="GPU device id to use [0] or multiple 0,1,2,3.",
                        default='0', type=str)
    parser.add_argument("--batch-size", dest="batch_size", help="Batch size.",
                        default=32, type=int)
    parser.add_argument("--fold", dest="fold", help="fold for Mpiigaze.", default=6, type=int)
    parser.add_argument("--workers", dest="workers", help="max dataloader worker (per RANK in DDP mode).", default=0,
                        type=int)
    return parser.parse_known_args()[0] if known else parser.parse_args()
