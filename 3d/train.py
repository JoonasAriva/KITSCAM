import glob
import logging
import os
import sys
from pathlib import Path

import hydra
import monai
import numpy as np
import torch
import wandb
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
)
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchsummary import summary
dir_checkpoint = Path('./checkpoints/')
artifact_path = "/gpfs/space/home/joonas97/data/kits21/synthetic_arts/ARTIFACT_LIGHT/*.npy"
regular_path = "/gpfs/space/home/joonas97/data/kits21/synthetic_arts/NORMAL/*.npy"


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    print(f"Running {cfg.project}, Work in {os.getcwd()}")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')
    logging.info("Loading data")

    # get data paths
    positive = []
    for filename in glob.glob(artifact_path):
        positive.append(filename)

    negative = []
    for filename in glob.glob(regular_path):
        negative.append(filename)

    # add paths together, create labels
    all_cases = positive + negative
    labels = np.concatenate((np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)))
    labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()

    # split data to train/test
    X_train, X_test, y_train, y_test = train_test_split(all_cases, labels,
                                                        test_size=0.2, random_state=42, stratify=labels)

    # Define transforms
    train_transforms = Compose([ScaleIntensity(dtype=torch.float32), EnsureChannelFirst()])
    val_transforms = Compose([ScaleIntensity(dtype=torch.float32), EnsureChannelFirst()])

    # create a training data loader
    train_ds = ImageDataset(image_files=X_train, labels=y_train, transform=train_transforms, dtype=torch.float32)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=2,
                              pin_memory=True)

    # create a validation data loader
    val_ds = ImageDataset(image_files=X_test, labels=y_test, transform=val_transforms, dtype=torch.float32)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, num_workers=2,
                            pin_memory=True)

    logging.info("Building model")
    if cfg.model.name == "densenet":
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=cfg.model.classes).to(device)
    else:
        model = monai.networks.nets.EfficientNetBN(cfg.model.name, spatial_dims=3, in_channels=1,
                                               num_classes=cfg.model.classes).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg.training.learning_rate)

    #summary(model,(1,200,512,512))

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []

    # (Initialize wandb logging)
    if not cfg.check:
        experiment = wandb.init(project='artifacts_classification', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs, batch_size=cfg.training.batch_size,
                 learning_rate=cfg.training.learning_rate, artifact_data = artifact_path))

    for epoch in range(cfg.training.epochs):

        model.train()
        epoch_loss = 0
        step = 0

        tepoch = tqdm(train_loader, unit="batch", ascii=True)
        for batch_data in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            tepoch.set_postfix({'batch_loss': loss.item()})

            if cfg.check and step >= 6:
                print('dtype:', inputs.dtype)
                break

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        tepoch.set_postfix({'epoch_loss': np.round(epoch_loss, 4)})

        #model.eval()

        num_correct = 0.0
        metric_count = 0
        val_loss = 0
        val_step = 0
        for val_data in val_loader:
            val_step += 1
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                val_outputs = model(val_images)
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()
                value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))

                metric_count += len(value)
                num_correct += value.sum().item()
            if cfg.check and val_step >= 6:
                break

        metric = num_correct / metric_count

        if not cfg.check:
            experiment.log({
                'val loss': val_loss / val_step,
                'train loss': epoch_loss,
                'val accuracy': metric,
                'epoch': epoch
            })

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1

            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'best_model.pth'))
            logging.info(f'Best new model! Checkpoint {epoch} saved!')

        print(f"Current epoch: {epoch + 1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")

        if cfg.check:
            logging.info("Model check completed")
            return
    torch.save(model.state_dict(), str(dir_checkpoint / 'last_model.pth'))
    logging.info(f'Last checkpoint! Checkpoint {epoch} saved!')
    print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


if __name__ == '__main__':
    main()
