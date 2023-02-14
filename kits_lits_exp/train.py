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
from monai.data import DataLoader, ImageDataset, NumpyReader, pad_list_data_collate
from monai.transforms import *
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import WeightedRandomSampler, DataLoader
from tqdm import tqdm

dir_checkpoint = Path('./checkpoints/')

kits_path = '/gpfs/space/home/joonas97/nnUNet/nn_pipeline/nnUNet_preprocessed/Task500_KITS_and_LITS/nnUNetData_plans_v2.1_stage1/KITS*.npz'
lits_path = '/gpfs/space/home/joonas97/nnUNet/nn_pipeline/nnUNet_preprocessed/Task500_KITS_and_LITS/nnUNetData_plans_v2.1_stage1/LITS*.npz'


def collect_and_split_data_files(positive_class_path, negative_class_path):
    # get data paths
    positive = []
    for filename in glob.glob(positive_class_path):
        positive.append(filename)

    negative = []
    for filename in glob.glob(negative_class_path):
        negative.append(filename)
    positive = np.array(positive)
    negative = np.array(negative)
    # rebalance classes
    print("positive: ", len(positive), "negative: ", len(negative))
    smaller_class_size = min(len(positive), len(negative))
    how_much_for_testing = int(smaller_class_size * 0.2)

    indexes_pos = np.random.choice(len(positive), how_much_for_testing, replace=False)
    indexes_neg = np.random.choice(len(negative), how_much_for_testing, replace=False)
    print("test lenghts")
    print(len(indexes_neg), len(indexes_pos))
    X_test = np.concatenate((positive[indexes_pos], negative[indexes_neg]))
    y_test = np.concatenate(
        (np.ones(len(positive[indexes_pos]), dtype=int), np.zeros(len(negative[indexes_neg]), dtype=int)))
    # y_test = torch.nn.functional.one_hot(torch.as_tensor(y_test)).float()

    mask_pos = np.ones(len(positive), dtype=bool)
    mask_neg = np.ones(len(negative), dtype=bool)
    mask_pos[indexes_pos] = 0
    mask_neg[indexes_neg] = 0

    train_pos = positive[mask_pos]
    train_neg = negative[mask_neg]

    X_train = np.concatenate((train_pos, train_neg))
    y_train = np.concatenate(
        (np.ones(len(train_pos), dtype=int), np.zeros(len(train_neg), dtype=int)))
    # y_train = torch.nn.functional.one_hot(torch.as_tensor(y_train)).float()

    # equal_positive = positive[:smaller_class_size]
    # equal_negative = negative[:smaller_class_size]
    # print("after:")
    # print("positive: ", len(positive), "negative: ", len(negative))
    #
    # # add paths together, create labels
    # all_cases = positive + negative
    # labels = np.concatenate((np.ones(len(positive), dtype=int), np.zeros(len(negative), dtype=int)))
    # labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()
    #
    # # split data to train/test
    # X_train, X_test, y_train, y_test = train_test_split(all_cases, labels,
    #                                                     test_size=0.2, random_state=42, stratify=labels)
    print(
        f"Dataset lenghts: X_train: {len(X_train)} X_test:{len(X_test)} y_train: {len(y_train)} y_test: {len(y_test)}", )
    return X_train, X_test, y_train, y_test


def create_and_setup_dataloaders_old(X_train, X_test, y_train, y_test, batch_size):
    # Define transforms
    transforms = Compose(
        [Remove_Segmentation(), CastToType(dtype=np.float16), CenterSpatialCrop(roi_size=[300, 512, 512])])

    train_ds = ImageDataset(image_files=X_train, labels=y_train, transform=transforms, dtype=torch.float16,
                            reader=NumpyReader(npz_keys="data"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=True, collate_fn=pad_list_data_collate)

    # create a validation data loader
    val_ds = ImageDataset(image_files=X_test, labels=y_test, transform=transforms, dtype=torch.float16,
                          reader=NumpyReader(npz_keys="data"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2,
                            pin_memory=True, collate_fn=pad_list_data_collate)

    return train_loader, val_loader


def create_and_setup_dataloaders(X_train, X_test, y_train, y_test, batch_size):
    # Define transforms
    transforms = Compose(
        [Remove_Segmentation(), CastToType(dtype=np.float16), CenterSpatialCrop(roi_size=[512, 512, 512])])

    # Define the random sampler (for class imbalance)
    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    weight = 1. / class_sample_count

    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    y_test = torch.nn.functional.one_hot(torch.as_tensor(y_test)).float()
    y_train = torch.nn.functional.one_hot(torch.as_tensor(y_train)).float()

    train_ds = ImageDataset(image_files=X_train, labels=y_train, transform=transforms, dtype=torch.float16,
                            reader=NumpyReader(npz_keys="data"))
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=2,
                              pin_memory=True, collate_fn=pad_list_data_collate, sampler=sampler)

    # create a validation data loader
    val_ds = ImageDataset(image_files=X_test, labels=y_test, transform=transforms, dtype=torch.float16,
                          reader=NumpyReader(npz_keys="data"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2,
                            pin_memory=True, collate_fn=pad_list_data_collate)

    return train_loader, val_loader


class Remove_Segmentation(Transform):
    def __call__(self, inputs):
        inputs = inputs[0]
        inputs = torch.unsqueeze(inputs, dim=0)
        return inputs


@hydra.main(config_path="config", config_name="config", version_base='1.1')
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Running {cfg.project}, Work in {os.getcwd()}")
    np.random.seed(0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')
    logging.info("Loading data")

    # create a training data loader
    X_train, X_test, y_train, y_test = collect_and_split_data_files(kits_path, lits_path)

    train_loader, val_loader = create_and_setup_dataloaders(X_train, X_test, y_train, y_test,
                                                            batch_size=cfg.training.batch_size)

    logging.info("Building model")
    if cfg.model.name == "densenet":
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=cfg.model.classes).to(
            device)
    else:
        model = monai.networks.nets.EfficientNetBN(cfg.model.name, spatial_dims=3, in_channels=1,
                                                   num_classes=cfg.model.classes).to(device)
    # if you need to continue training
    if "checkpoint" in cfg.keys():
        print("Using checkpoint", cfg.checkpoint)
        model.load_state_dict(torch.load(os.path.join('/gpfs/space/home/joonas97', cfg.checkpoint)))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg.training.learning_rate)

    # summary(model,(1,200,512,512))

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []

    # (Initialize wandb logging)
    if not cfg.check:
        experiment = wandb.init(project='KITS+LITS_classification', resume='allow', anonymous='must')
        experiment.config.update(
            dict(epochs=cfg.training.epochs, batch_size=cfg.training.batch_size,
                 learning_rate=cfg.training.learning_rate, artifact_data=kits_path))

    for epoch in range(cfg.training.epochs):

        model.train()
        epoch_loss = 0
        step = 0

        tepoch = tqdm(train_loader, unit="batch", ascii=True)
        train_count = 0
        train_num_correct = 0
        for batch_data in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            step += 1
            # print(batch_data)
            inputs, labels = batch_data[0][:, 0, :, :, :].to(device), batch_data[1].to(device)
            inputs = torch.unsqueeze(inputs, 1)
            # tepoch.set_description(f"shape {inputs.shape}")
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            value = torch.eq(outputs.argmax(dim=1), labels.argmax(dim=1))
            train_count += len(value)
            train_num_correct += value.sum().item()

            tepoch.set_postfix({'running epoch acc': train_num_correct/train_count})

            if cfg.check and step >= 6:
                break

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        tepoch.set_postfix({'epoch_loss': np.round(epoch_loss, 4)})

        # model.eval()

        num_correct = 0.0
        metric_count = 0
        val_loss = 0
        val_step = 0
        for val_data in val_loader:
            val_step += 1
            val_images, val_labels = val_data[0][:, 0, :, :, :].to(device), val_data[1].to(device)
            val_images = torch.unsqueeze(val_images, 1)
            # tepoch.set_description(f"shape {val_images.shape}")
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
                'epoch': epoch,
                'train accuracy': train_num_correct / train_count
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
