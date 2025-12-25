import os
import argparse
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
import utils.val as test

from torchvision import transforms
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory

from utils.torch_utils import (
    smart_DDP,
    smart_optimizer,
    torch_distributed_zero_first,
    select_device,
    reduce_value,
    cleanup,
    get_ignored_params,
    get_fc_params,
    get_non_ignored_params
)
from utils.dataloaders import create_dataloader
from Emodels.Emodel_AStarNet_WFD_SESA import Emodel_StarNet
from utils.general import init_seeds, LOGGER, increment_path, one_cycle, colorstr
from utils.logger.loggers import write_to_csv
from utils.plots import plot_results
from utils.loss import ComputeLoss

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", -1))
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"


def parse_opt(known=False):
    """ Parse command-line arguments for training """
    parser = argparse.ArgumentParser(description="Gaze estimation using EmodelNet")
    parser.add_argument("--gaze_root", dest="gaze_root", help="Directory path for gaze.",
                        default="datasets/FaceBased/MPIIFaceGaze")
    parser.add_argument("--dataset", dest="dataset",
                        default="MpiigazeKFold", type=str)
    parser.add_argument("--num_bins", dest="num_bins", default=28, type=int)
    parser.add_argument("--angle", dest="angle", default=42)
    parser.add_argument("--pretrained", dest="pretrained", help="Path of pretrained model.",
                        default=None, type=str)
    parser.add_argument("--device", dest="device", help="GPU device id to use [0] or multiple 0,1,2,3.",
                        default='0', type=str)
    parser.add_argument("--project", default=ROOT / "runs/train", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--epochs", dest="epochs", help="Maximum number of training epochs.",
                        default=60, type=int)
    parser.add_argument("--batch-size", dest="batch_size", help="Batch size.",
                        default=32, type=int)
                        #default=16, type = int)
    parser.add_argument("--freeze-layers", type=bool, default=False, help="Freeze layers.")
    parser.add_argument("--workers", dest="workers", help="max dataloader worker (per RANK in DDP mode).", default=8, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=3, help='number of warmup epochs')
    parser.add_argument('--adam', action='store_true', default=True, help='use torch.optim.Adam() optimizer')
    parser.add_argument("--lr", dest="lr", help="Base learning rate.",
                        default=0.0001, type=float)
    parser.add_argument("--weight_decay", dest="weight_decay", help="Weight decay",
                        default=0, type=float)
    parser.add_argument("--lrf", dest="lrf", help="Learning rate attenuation factor.",
                        default=0.01, type=float)
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--alpha", dest="alpha", help="Pitch loss scaler.",
                        default=1, type=float)
    parser.add_argument("--beta", dest="beta", help="Yaw loss scaler.",
                        default=0.01, type=float)
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--syncBN", action="store_true", help="Use SyncBatchNorm, only available in DDP mode.")
    parser.add_argument("--local-rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify.")
    parser.add_argument("--seed", type=int, default=114514, help="Global training seed.")
    return parser.parse_known_args()[0] if known else parser.parse_args()


def train(opt):
    gaze_root, dataset, num_bins, angle, pretrained, device, epochs, batch_size, workers, save_dir = (
        opt.gaze_root,
        opt.dataset,
        opt.num_bins,
        opt.angle,
        opt.pretrained,
        opt.device,
        opt.epochs,
        opt.batch_size,
        opt.workers,
        opt.save_dir,
    )

    data_transform = {
        "train": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        "val": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    }

    # Config
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK)

    if opt.cos_lr:
        lf = one_cycle(1, opt.lrf, epochs)
    else:
        def lf(x):
            return (1 - x / epochs) * (1.0 - opt.lrf) + opt.lrf

    if dataset == "MpiigazeKFold":
        # leave-one-subject-out cross-validation
        leave_one_avg_error = 0
        for fold in range(0, 15):
            LOGGER.info(f"Leave-one-subject-out cross-validation， fold {fold} for validation.")
            # Model
            model = Emodel_StarNet().to(device)
            computeLoss = ComputeLoss(model)

            if opt.freeze_layers:
                for name, para in model.named_parameters():
                    if "fc_yaw_gaze" or "fc_pitch_gaze" not in name:
                        para.requires_grad_(False)

            else:
                if opt.syncBN and cuda and RANK != -1:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

            nbs = 64
            weight_decay = opt.weight_decay
            accumulate = max(round(nbs / batch_size), 1)
            weight_decay *= batch_size * accumulate / nbs

            if opt.adam:
                optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=weight_decay,
                                             betas=(0.9, 0.999))
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=weight_decay)

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

            if cuda and RANK == -1 and torch.cuda.device_count() > 1:
                LOGGER.warning("WARNING DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.")
                model = torch.nn.DataParallel(model)

            if cuda and RANK != -1:
                model = smart_DDP(model)

            train_loader, train_data_set = create_dataloader(
                root=gaze_root + "/Image",
                path=gaze_root + "/Label",
                angle=angle,
                batch_size=batch_size,
                dataset_name=dataset,
                rank=LOCAL_RANK,
                workers=workers,
                transform=data_transform['train'],
                fold=fold,
                scaler=False,
            )
            val_loader, val_data_set = create_dataloader(
                root=gaze_root + "/Image",
                path=gaze_root + "/Label",
                angle=angle,
                batch_size=batch_size,
                dataset_name=dataset,
                rank=LOCAL_RANK,
                workers=workers,
                transform=data_transform['val'],
                fold=fold,
                scaler=False,
                train=False,
            )

            configuration = f"\ntrain configuration, device={device}, batch_size={batch_size}\n" \
                            f"Start training dataset={dataset}, fold={fold}\n"
            LOGGER.info(configuration)

            nb = len(train_loader)
            mloss = torch.zeros(2, device=device)
            nw = max(round(opt.warmup_epochs * nb), 1500)
            last_opt_step = -1
            min_error = 1e4
            LOGGER.info(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'gloss', 'aloss'))

            for epoch in range(epochs):
                sum_loss = iter_gaze = 0

                if RANK != -1:
                    train_loader.sampler.set_epoch(epoch)

                if RANK in {-1, 0}:
                    summary_name = "{}_{}".format("Emodel-" + dataset + '-fold' + str(fold),"summary")
                    output_weights = os.path.join(save_dir, summary_name)
                    if not os.path.exists(output_weights):
                        os.makedirs(output_weights)

                pbar = enumerate(train_loader)
                if RANK in {-1, 0}:
                    pbar = tqdm(pbar, total=nb, file=sys.stdout)

                for i, (image_gaze, labels_gaze) in pbar:
                    ni = i + nb * epoch
                    image_gaze = image_gaze.to(device)
                    labels_gaze = labels_gaze.to(device)
                    gaze_pred = model(image_gaze)
                    loss, loss_items = computeLoss(gaze_pred, labels_gaze)
                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    sum_loss += loss

                    iter_gaze += 1
                    if RANK in {-1, 0}:
                        mloss = (mloss * i + loss_items) / (i + 1)
                        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                        pbar.set_description(('%10s' * 2 + '%10.4g' * 2) % (
                            f'{epoch}/{epochs - 1}', mem, *mloss))

                lr = [x['lr'] for x in optimizer.param_groups]
                scheduler.step()

                if RANK in {-1, 0}:
                    avg_error, val_avg_loss = test.run(
                        device=device,
                        model=model,
                        dataloader=val_loader,
                        half=False,
                    )
                    if avg_error < min_error:
                        LOGGER.info("Saving best model...")
                        torch.save(model.state_dict(),
                                   output_weights + "/" + "best" + ".pkl")
                        min_error = avg_error

                    if epoch % 1 == 0 and epoch < epochs:
                        LOGGER.info("Saving last model...")
                        torch.save(model.state_dict(),
                                   output_weights + "/" + "last" + ".pkl")
                        save_loss = sum_loss / iter_gaze
                        lr_value = lr[0] if lr else 0
                        write_to_csv(output_weights, epoch, save_loss.cpu().item(), lr_value, val_avg_loss, avg_error)

            plot_results(file=output_weights + "/" + "results.csv")

            leave_one_avg_error += min_error
            LOGGER.info(f"leave-one-subject-out cross-validation mean avg error {leave_one_avg_error / (fold + 1):.3g}")
    elif dataset == "Gaze360":
        leave_one_avg_error = 0
        model = Emodel_StarNet().to(device)
        computeLoss = ComputeLoss(model)

        if opt.freeze_layers:
            for name, para in model.named_parameters():
                if "fc_yaw_gaze" or "fc_pitch_gaze" not in name:
                    para.requires_grad_(False)

        else:
            if opt.syncBN and cuda and RANK != -1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

        nbs = 64
        weight_decay = opt.weight_decay
        accumulate = max(round(nbs / batch_size), 1)
        weight_decay *= batch_size * accumulate / nbs
        if opt.adam:
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=weight_decay)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        if cuda and RANK == -1 and torch.cuda.device_count() > 1:
            LOGGER.warning("WARNING DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.")
            model = torch.nn.DataParallel(model)

        if cuda and RANK != -1:
            model = smart_DDP(model)

        train_loader, train_data_set = create_dataloader(
            root=gaze_root + "/Image",
            path=gaze_root + "/Label/train.label",
            angle=angle,
            batch_size=batch_size,
            dataset_name=dataset,
            rank=LOCAL_RANK,
            workers=workers,
            transform=data_transform['train'],
            binwidth=4,
            scaler=False,
            train=True,
        )
        val_loader, val_data_set = create_dataloader(
            root=gaze_root + "/Image",
            path=gaze_root + "/Label/val.label",
            angle=angle,
            batch_size=batch_size,
            dataset_name=dataset,
            rank=LOCAL_RANK,
            workers=workers,
            transform=data_transform['val'],
            binwidth=4,
            scaler=False,
            train=False,
        )

        configuration = f"\ntrain configuration, device={device}, batch_size={batch_size}\n" \
                        f"Start training dataset={dataset}\n"
        LOGGER.info(configuration)

        nb = len(train_loader)
        mloss = torch.zeros(2, device=device)
        min_error = 1e4
        LOGGER.info(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'gloss', 'aloss'))

        for epoch in range(epochs):
            sum_loss = iter_gaze = 0

            if RANK != -1:
                train_loader.sampler.set_epoch(epoch)

            if RANK in {-1, 0}:
                summary_name = "{}_{}".format("Emodel-" + dataset,"summary")
                output_weights = os.path.join(save_dir, summary_name)
                if not os.path.exists(output_weights):
                    os.makedirs(output_weights)

            pbar = enumerate(train_loader)
            if RANK in {-1, 0}:
                pbar = tqdm(pbar, total=nb, file=sys.stdout)

            for i, (image_gaze, labels_gaze) in pbar:
                image_gaze = image_gaze.to(device)
                labels_gaze = labels_gaze.to(device)

                gaze_pred = model(image_gaze)

                loss, loss_items = computeLoss(gaze_pred, labels_gaze)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                sum_loss += loss

                iter_gaze += 1
                if RANK in {-1, 0}:
                    mloss = (mloss * i + loss_items) / (i + 1)
                    mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                    pbar.set_description(('%10s' * 2 + '%10.4g' * 2) % (
                        f'{epoch}/{epochs - 1}', mem, *mloss))

            lr = [x['lr'] for x in optimizer.param_groups]
            scheduler.step()

            if RANK in {-1, 0}:
                avg_error, val_avg_loss = test.run(
                    device=device,
                    model=model,
                    dataloader=val_loader,
                    half=False,
                )
                if avg_error < min_error:
                    LOGGER.info("Saving best model...")
                    torch.save(model.state_dict(),
                               output_weights + "/" + "best" + ".pkl")
                    min_error = avg_error
                if epoch % 1 == 0 and epoch < epochs:
                    LOGGER.info("Saving last model...")
                    torch.save(model.state_dict(),
                               output_weights + "/" + "last" + ".pkl")
                    save_loss = sum_loss / iter_gaze
                    write_to_csv(output_weights, epoch, save_loss.cpu().item(), lr[1], val_avg_loss, avg_error)

        plot_results(file=output_weights + "/" + "results.csv")

        leave_one_avg_error += min_error
        LOGGER.info(f"leave-one-subject-out cross-validation mean avg error {leave_one_avg_error:.3g}")


def main(opt):
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                timeout=timedelta(seconds=10800))
    opt.device = device
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    train(opt)


def run(**kwargs):
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
