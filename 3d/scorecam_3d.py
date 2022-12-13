import logging
import monai
import numpy as np
import os
import sys
import torch
from glob import glob
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/KITSCAM/')
sys.path.append('/gpfs/space/home/joonas97/KITSCAM/ScoreCAM')
os.chdir('/gpfs/space/home/joonas97/KITSCAM')

from ScoreCAM.cam.scorecam import *
from torchsummary import summary
from sklearn.model_selection import train_test_split
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity)

logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.info("Loading model")
model = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1, num_classes=2).to(device)
model.load_state_dict(torch.load('/gpfs/space/home/joonas97/KITSCAM/best_metric_model_classification3d_array.pth'))

summary(model, (1, 200, 512, 512))

model_dict = dict(type='efficientnet', arch=model, layer_name='_blocks', input_size=(512, 512))
efficientnet_scorecam = ScoreCAM(model_dict)

logging.info("Collecting data")
kidney_chunks = []
for filename in glob("/gpfs/space/home/joonas97/data/kits21/3d/*/KIDNEY/*.npy"):
    kidney_chunks.append(filename)

no_chunks = []
for filename in glob("/gpfs/space/home/joonas97/data/kits21/3d/*/NO_KIDNEY/*.npy"):
    no_chunks.append(filename)

all_chunks = kidney_chunks + no_chunks
labels = np.concatenate((np.ones(len(kidney_chunks), dtype=int), np.zeros(len(no_chunks), dtype=int)))
# labels = np.ones(len(kidney_chunks), dtype=int)
labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()

train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst()])

_, X_test, _, y_test = train_test_split(all_chunks, labels,
                                        test_size=0.2, random_state=42, stratify=labels)

# create a training data loader
train_ds = ImageDataset(image_files=X_test, labels=y_test, transform=train_transforms)
data_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2,
                         pin_memory=True)
cams = []
true_labels = []
imgs = []
os.chdir('/gpfs/space/home/joonas97/KITSCAM/3d')
logging.info("Extracting CAM maps")
for i, batch_data in tqdm(enumerate(data_loader), ascii=True, total=20):
    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
    cam = efficientnet_scorecam(inputs).cpu()

    cams.append(cam)

    labels = labels.cpu()
    true_labels.append(labels)

    inputs = inputs.cpu()
    imgs.append(inputs)

    if i >= 20:
        break
logging.info("Saving results")
np.save("first_cams_1x16x16", np.array(cams))
np.save("first_labels_1x16x16", np.array(true_labels))
np.save("first_inputs_1x16x16", np.array(imgs))
logging.info("All done!")
