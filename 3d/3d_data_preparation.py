import glob
import logging
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm


# nr_of_chunks = 5


def classify_chunk(chunk: np.ndarray) -> bool:
    values, counts = np.unique(chunk, return_counts=True)
    if 1 in values and counts[np.where(values == 1)] > 1000:
        return True
    else:
        return False


logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)

orig_data_dir = '/gpfs/space/home/joonas97/data/kits21/kits21/data/'
os.chdir(orig_data_dir)

files = []
logging.info("collecting filenames")
for filename in glob.glob('*/aggregated_MAJ_seg.nii.gz'):
    files.append(filename)

if not os.path.exists('/gpfs/space/home/joonas97/data/kits21/3d'):
    os.mkdir('/gpfs/space/home/joonas97/data/kits21/3d')

os.chdir('/gpfs/space/home/joonas97/data/kits21/3d')

kidney_slices = 0
no_kidney_slices = 0
logging.info("Starting to prepare the chunks")
pbar = tqdm(files, ascii=True)
for seg_file in pbar:
    case_id = seg_file.split('/')[0]
    if not os.path.exists(case_id):
        os.mkdir(case_id)
        os.mkdir(os.path.join(case_id, "KIDNEY"))
        os.mkdir(os.path.join(case_id, "NO_KIDNEY"))

    # segmentation file is for checking for categorizing kidney vs no kidney chunks
    segmentation = nib.load(os.path.join(orig_data_dir, seg_file)).get_fdata()
    image_file = seg_file.replace('aggregated_MAJ_seg', 'imaging')
    image = nib.load(os.path.join(orig_data_dir, image_file)).get_fdata()

    if segmentation.shape[1:3] != (512, 512):
        logging.info(f'shape {segmentation.shape} is not in shape of (n,512,512), skipping patient')
        continue

    scan_height = segmentation.shape[0]
    chunk_height = 45  # before: scan_height // nr_of_chunks
    nr_of_chunks = scan_height // chunk_height
    for i in range(nr_of_chunks):
        seg_chunk = segmentation[i * chunk_height:(i + 1) * chunk_height]
        scan_chunk = image[i * chunk_height:(i + 1) * chunk_height]
        if classify_chunk(seg_chunk):
            np.save(f"{case_id}/KIDNEY/{i * chunk_height}_{(i + 1) * chunk_height}", scan_chunk)
            kidney_slices += 1
        else:
            np.save(f"{case_id}/NO_KIDNEY/{i * chunk_height}_{(i + 1) * chunk_height}", scan_chunk)
            no_kidney_slices += 1
        pbar.set_postfix({'kidney_chunks': kidney_slices, 'free_chunks': no_kidney_slices})

logging.info(f'got {kidney_slices} kidney chucks and {no_kidney_slices} no kidney chunks')
