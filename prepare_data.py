import os
import nibabel as nib
import numpy as np
import glob
from typing import Tuple, List
from tqdm import tqdm
import json


def classify_slices(segmentation: np.ndarray) -> Tuple[List, List]:
    # give me slice indexes where label 1 (kidney) is present and where the area of kidney is bigger than 1000 pixels
    # currently applying this 1000 pixel filter because im not sure if model can detect very small slices of kidneys
    kidney_slice_indexes = [slice_nr for slice_nr, i in enumerate(segmentation) if
                            1 in i and np.unique(i, return_counts=True)[1][1] > 1000]
    no_kidney_slice_indexes = [slice_nr for slice_nr, i in enumerate(segmentation) if 1 not in i]
    return kidney_slice_indexes, no_kidney_slice_indexes

def filter_slices(slice)

os.chdir('/gpfs/space/home/joonas97/data/kits21/kits21/data/')

# gather all KITS segmentation files
segmentations = []
for filename in glob.glob('*/aggregated_MAJ_seg.nii.gz'):
    segmentations.append(filename)

# load images and filter them to slices with and without kidney
organized_slices = {}
for file in tqdm(segmentations):
    case_id = file.split('/')[0]
    segmentation = nib.load(file).get_fdata()
    kidney_slices, no_kidney_slices = classify_slices(segmentation)

    # use only every fifth slice (neighbouring slices are very similar)
    organized_slices[case_id] = [kidney_slices[::5], no_kidney_slices[::5]]

np.random.seed(19)  # fix seed
patient_indexes = np.arange(300)
np.random.shuffle(patient_indexes)

train_patients = patient_indexes[:100]
train_slices = {key: organized_slices[key] for key in train_patients}
with open("train_slices.json", "w") as outfile:
    json.dump(train_slices, outfile)

scorecam_patients = patient_indexes[100:250]
scorecam_slices = {key: organized_slices[key] for key in scorecam_patients}
with open("scorecam_slices.json", "w") as outfile:
    json.dump(scorecam_slices, outfile)

test_patients = patient_indexes[250:]
test_slices = {key: organized_slices[key] for key in test_patients}
with open("test_slices.json", "w") as outfile:
    json.dump(test_slices, outfile)

firstiter = True
for (keys, values) in tqdm(organized_slices.items()):
    ct_scan = nib.load(keys + '/imaging.nii.gz').get_fdata()

    if firstiter:
        all_kidney_slices = ct_scan[values[0]]
        all_no_kidney_slices = ct_scan[values[1]]
        firstiter = False
    else:
        new_kidney_slices = ct_scan[values[0]]
        new_no_kidney_slices = ct_scan[values[1]]
        all_kidney_slices = np.concatenate((all_kidney_slices, new_kidney_slices), axis=0)
        all_no_kidney_slices = np.concatenate((all_no_kidney_slices, new_no_kidney_slices), axis=0)
