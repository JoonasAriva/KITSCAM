import logging
import sys

import numpy as np
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/KITSCAM/ScoreCAM')
import os

os.chdir('/gpfs/space/home/joonas97/KITSCAM')

from ScoreCAM.cam.scorecam import *
from utils import CustomDataset

logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)
data_dir = '/gpfs/space/home/joonas97/data/kits21/processed_2d_slices/val/for_scorecam'

logging.info("Loading data from" + data_dir)
val_dataset = CustomDataset(data_dir)

logging.info("Initializing model")
resnet = models.resnet50()
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 2)
resnet.load_state_dict(torch.load('trained_resnet'))
resnet.eval()

resnet_model_dict = dict(type='resnet50', arch=resnet, layer_name='layer4', input_size=(512, 512))
resnet_scorecam = ScoreCAM(resnet_model_dict)

indexes = []
first_scorecam = True
logging.info("Starting to extract Score-CAM maps")
for i in tqdm(np.arange(3998)):
    input_image = val_dataset[i][0].float().unsqueeze(0)

    if torch.cuda.is_available():
        input_image = input_image.cuda()
    predicted_class = resnet(input_image).max(1)[-1]
    if predicted_class == 1:
        indexes.append(i)
        if first_scorecam:
            scorecams = resnet_scorecam(input_image)
            scorecams = scorecams[0].cpu().detach().numpy()
            first_scorecam = False
        else:
            scorecams = np.concatenate((scorecams, resnet_scorecam(input_image)[0].cpu().detach().numpy()), axis=0)

np.save("predicted_slices_with_kindey", np.array(indexes))
np.save("cam_maps", scorecams)

logging.info('All done!')
