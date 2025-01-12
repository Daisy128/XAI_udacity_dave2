import pathlib
import lightning as pl
import numpy as np
import pandas as pd
import torch
import torchvision.transforms
import tqdm
from PIL import Image
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import itertools
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from model.segmentation.unet.training import SegmentationDataset
# from model.lane_keeping.dave_torch.dave_model import Dave2
from model.segmentation.unet.unet_model import SegmentationUnet
from utils.conf import ACCELERATOR, DEVICE, DEFAULT_DEVICE, CHECKPOINT_DIR


def test_set_evaluation(track, weather, model_name, csv_filename):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # Run parameters
    # input_shape = (3, 160, 320)
    # max_epochs = 2000
    # accelerator = ACCELERATOR
    # devices = [DEVICE]

    checkpoint_name = CHECKPOINT_DIR.joinpath("segmentation", f"unet_{track}_{weather}", model_name)

    miou_metric = torchmetrics.classification.BinaryJaccardIndex().to(DEFAULT_DEVICE)
    prc_metric = torchmetrics.classification.BinaryPrecision().to(DEFAULT_DEVICE)
    rec_metric = torchmetrics.classification.BinaryRecall().to(DEFAULT_DEVICE)
    acc_metric = torchmetrics.classification.BinaryAccuracy().to(DEFAULT_DEVICE)
    cm_metric = torchmetrics.classification.BinaryConfusionMatrix().to(DEFAULT_DEVICE)

    metric_values = {
        'miou': [],
        'acc': [],
        'rec': [],
        'prc': [],
        'TP': [],
        'TN': [],
        'FP': [],
        'FN': [],
    }

    driving_model = SegmentationUnet.load_from_checkpoint(checkpoint_name, map_location=DEFAULT_DEVICE)

    dataset = SegmentationDataset(dataset_dir="Data/collected_seg_data/", split="test", csv_file=csv_filename)

    loader = DataLoader(
        dataset,
        batch_size=4,
        prefetch_factor=2,
        num_workers=4,
    )
    i = 0
    # stored_segmentation_dir = dataset.dataset_dir.joinpath(f"computed_segmentation_{track}_{weather}")
    # stored_segmentation_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            img, true = batch
            true = true.to(DEFAULT_DEVICE)
            img = img.to(DEFAULT_DEVICE)
            pred = driving_model(img)
            # for img in pred:
            #     stored_segmentation_name = dataset.metadata['segmentation_filename'].values[i]
            #     stored_segmentation_name = "computed_" + stored_segmentation_name.split('/')[-1]
            #     torchvision.utils.save_image(img, stored_segmentation_dir.joinpath(stored_segmentation_name))
            #     i = i + 1
            miou = miou_metric(pred, true)
            acc = acc_metric(pred, true)
            prc = prc_metric(pred, true)
            rec = rec_metric(pred, true)
            cm = cm_metric(pred, true)
            metric_values['miou'].append(miou.detach().cpu())
            metric_values['acc'].append(acc.detach().cpu())
            metric_values['rec'].append(rec.detach().cpu())
            metric_values['prc'].append(prc.detach().cpu())
            metric_values['TP'].append(cm[1][1].detach().cpu())
            metric_values['TN'].append(cm[0][0].detach().cpu())
            metric_values['FN'].append(cm[1][0].detach().cpu())
            metric_values['FP'].append(cm[0][1].detach().cpu())

    print(f"mIoU computed on dataset {track}-{weather}: {np.array(metric_values['miou']).mean()}")
    print(f"acc computed on dataset {track}-{weather}: {np.array(metric_values['acc']).mean()}")
    print(f"rec computed on dataset {track}-{weather}: {np.array(metric_values['rec']).mean()}")
    print(f"prc computed on dataset {track}-{weather}: {np.array(metric_values['prc']).mean()}")
    print(f"TP computed on dataset {track}-{weather}: {np.array(metric_values['TP']).mean()}")
    print(f"TN computed on dataset {track}-{weather}: {np.array(metric_values['TN']).mean()}")
    print(f"FN computed on dataset {track}-{weather}: {np.array(metric_values['FN']).mean()}")
    print(f"FP computed on dataset {track}-{weather}: {np.array(metric_values['FP']).mean()}")



def all_set_evaluation_and_save(track, weather, model_name, csv_filename):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')

    # Run parameters
    # input_shape = (3, 160, 320)
    # max_epochs = 2000
    # accelerator = ACCELERATOR
    # devices = [DEVICE]

    checkpoint_name = CHECKPOINT_DIR.joinpath("segmentation", f"unet_{track}_{weather}", model_name)
    driving_model = SegmentationUnet.load_from_checkpoint(checkpoint_name, map_location=DEFAULT_DEVICE)

    dataset = SegmentationDataset(dataset_dir="Data/collected_seg_data/", split="all", csv_file=csv_filename)

    loader = DataLoader(
        dataset,
        batch_size=4,
        prefetch_factor=2,
        num_workers=4,
    )
    i = 0
    stored_segmentation_dir = dataset.dataset_dir.joinpath(f"computed_segmentation_{track}_{weather}")
    stored_segmentation_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            img, true = batch
            true = true.to(DEFAULT_DEVICE)
            img = img.to(DEFAULT_DEVICE)
            pred = driving_model(img)
            for img in pred:
                stored_segmentation_name = dataset.metadata['segmentation_filename'].values[i]
                stored_segmentation_name = "computed_" + stored_segmentation_name.split('/')[-1]
                torchvision.utils.save_image(img, stored_segmentation_dir.joinpath(stored_segmentation_name))
                i = i + 1


if __name__ == '__main__':
    track = "lake"
    weather = "sun"
    csv_filename = 'lake_sun_training.csv'
    model_name = "segmentation_unet_epoch=56_step=741_val_mIoU=0.9820912480354309_val_loss=0.02768610045313835.ckpt"
    #test_set_evaluation(track, weather, model_name, csv_filename)
    all_set_evaluation_and_save(track, weather, model_name, csv_filename)

    track = "mountain"
    weather = "sun"
    csv_filename = 'mountain_sun_training.csv'
    model_name = "segmentation_unet_epoch=89_step=1980_val_mIoU=0.9733365774154663_val_loss=0.042826734483242035.ckpt"
    #test_set_evaluation(track, weather, model_name, csv_filename)
    all_set_evaluation_and_save(track, weather, model_name, csv_filename)