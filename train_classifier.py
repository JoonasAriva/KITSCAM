from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision.models import ResNet50_Weights
from torchvision import models

import os
import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)
os.chdir('/gpfs/space/home/joonas97/KITSCAM')
from utils import CustomDataset
from model_trainer import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = '/gpfs/space/home/joonas97/data/kits21/processed_2d_slices/'

logging.info("Loading data")

train_dataset = CustomDataset(os.path.join(data_dir, "train"))

val_dataset = CustomDataset(os.path.join(data_dir, "val"), mu=train_dataset.mu, std=train_dataset.std)

dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                                    shuffle=True, num_workers=2),
               'val': torch.utils.data.DataLoader(val_dataset, batch_size=4,
                                                  shuffle=False, num_workers=2)}

# dataset_sizes = {x: len(slices_datasets[x]) for x in ['train', 'val']}
class_names = train_dataset.classes

logging.info("Initialising model")
model_ft = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

logging.info("Starting training")
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, device, dataloaders,
                       num_epochs=15)

logging.info("Saving best model")
torch.save(model_ft.state_dict(), 'trained_resnet')
