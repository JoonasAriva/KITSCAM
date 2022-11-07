import glob
import json
import logging
import os
from typing import Tuple, List, Dict

import nibabel as nib
import numpy as np
from tqdm import tqdm


def classify_slices(segmentation: np.ndarray) -> Tuple[List, List]:
    # give me slice indexes where label 1 (kidney) is present and where the area of kidney is bigger than 1000 pixels
    # currently applying this 1000 pixel filter because im not sure if model can detect very small slices of kidneys
    kidney_slice_indexes = [slice_nr for slice_nr, i in enumerate(segmentation) if
                            1 in i and np.unique(i, return_counts=True)[1][1] > 1000]
    no_kidney_slice_indexes = [slice_nr for slice_nr, i in enumerate(segmentation) if 1 not in i]
    return kidney_slice_indexes, no_kidney_slice_indexes


def get_slices(data: Dict, patient_indexes: np.ndarray, file_name: str = None) -> Dict:
    # filter ct scan dictionary based on patient ids and save resulting dict
    filtered_data = {key: data[key] for key in patient_indexes}
    if file_name:
        with open(file_name + ".json", "w") as outfile:
            json.dump(filtered_data, outfile)
    return filtered_data


logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)

os.chdir('/gpfs/space/home/joonas97/data/kits21/kits21/data/')

# gather all KITS segmentation files

segmentations = []
logging.info("collecting filenames")
for filename in glob.glob('*/aggregated_MAJ_seg.nii.gz'):
    segmentations.append(filename)

# load images and filter them to slices with and without kidney
organized_slices = {}
logging.info("organizing ct slice indexes to kidney/no kidney groups")
for file in tqdm(segmentations):
    case_id = file.split('/')[0]
    segmentation = nib.load(file).get_fdata()
    kidney_slices, no_kidney_slices = classify_slices(segmentation)

    # use only every fifth slice (neighbouring slices are very similar)
    organized_slices[case_id] = [kidney_slices[::5], no_kidney_slices[::5]]

np.random.seed(19)  # fix seed
patient_indexes = np.arange(300)
np.random.shuffle(patient_indexes)

logging.info("splitting slice indexes to train/val(scorecam)/test")
train_slices = get_slices(organized_slices, patient_indexes[:50], file_name='train_slices')
scorecam_slices = get_slices(organized_slices, patient_indexes[50:250], file_name='scorecam_slices')
test_slices = get_slices(organized_slices, patient_indexes[:50], file_name='test_slices')

logging.info("collecting ct images based on indexes")
for slice_type, slices in {'train': train_slices, 'scorecam': scorecam_slices, 'test': test_slices}:

    logging.info('starting with ' + slice_type)
    firstiter = True
    for (keys, values) in tqdm(slices.items()):
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

    logging.info('saving ' + slice_type + 'images')
    np.save(slice_type + 'kidney_slices', all_kidney_slices)
    np.save(slice_type + 'no_kidney_slices', all_no_kidney_slices)

logging.info("All done!")