import glob
import logging
import sys

import monai
import numpy as np
import torch
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
)
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)



kidney_chunks = []
for filename in glob.glob("/gpfs/space/home/joonas97/data/kits21/3d/*/KIDNEY/*.npy"):
    kidney_chunks.append(filename)

no_chunks = []
for filename in glob.glob("/gpfs/space/home/joonas97/data/kits21/3d/*/NO_KIDNEY/*.npy"):
    no_chunks.append(filename)

all_chunks = kidney_chunks + no_chunks
labels = np.concatenate((np.ones(len(kidney_chunks), dtype=int), np.zeros(len(no_chunks), dtype=int)))
labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()



X_train, X_test, y_train, y_test = train_test_split(all_chunks, labels,
                                                    test_size=0.2, random_state=42, stratify=labels)

# Define transforms
train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])

val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])

# create a training data loader
train_ds = ImageDataset(image_files=X_train, labels=y_train, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2,
                          pin_memory=pin_memory)

# create a validation data loader
val_ds = ImageDataset(image_files=X_test, labels=y_test, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=10, num_workers=2,
                        pin_memory=pin_memory)

model = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=2).to(device)
loss_function = torch.nn.CrossEntropyLoss()
# loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()
max_epochs = 25

for epoch in range(max_epochs):

    # print("-" * 10)
    # print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    tepoch = tqdm(train_loader, unit="batch", ascii=True)
    for batch_data in tepoch:
        tepoch.set_description(f"Epoch {epoch}")

        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        # print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        tepoch.set_postfix({'batch_loss': loss.item()})

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    # print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    tepoch.set_postfix({'epoch_loss': np.round(epoch_loss, 4)})
    if (epoch + 1) % val_interval == 0:
        model.eval()

        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                metric_count += len(value)
                num_correct += value.sum().item()

        metric = num_correct / metric_count
        metric_values.append(metric)

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model_classification3d_array.pth")
            print("saved new best metric model")

        print(f"Current epoch: {epoch + 1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_accuracy", metric, epoch + 1)

print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
