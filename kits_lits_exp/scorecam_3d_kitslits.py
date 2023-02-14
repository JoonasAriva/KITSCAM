import logging
import os
import sys

import hydra
import monai
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append('/gpfs/space/home/joonas97/KITSCAM/')
sys.path.append('/gpfs/space/home/joonas97/KITSCAM/ScoreCAM')
# os.chdir('/gpfs/space/home/joonas97/KITSCAM')
import glob

print(os.getcwd())
from ScoreCAM.cam.scorecam import *
from train import create_and_setup_dataloaders_old
from torchsummary import summary


def collect_and_split_data_files_old(positive_class_path, negative_class_path):
    # get data paths
    positive = []
    for filename in glob.glob(positive_class_path):
        positive.append(filename)

    negative = []
    for filename in glob.glob(negative_class_path):
        negative.append(filename)

    # rebalance classes
    print("positive: ", len(positive), "negative: ", len(negative))
    smaller_class_size = min(len(positive), len(negative))

    equal_positive = positive[:smaller_class_size]
    equal_negative = negative[:smaller_class_size]
    print("after:")
    print("positive: ", len(positive), "negative: ", len(negative))

    # add paths together, create labels
    all_cases = equal_positive + equal_negative
    labels = np.concatenate((np.ones(len(equal_positive), dtype=int), np.zeros(len(equal_negative), dtype=int)))
    labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float()

    # split data to train/test
    X_train, X_test, y_train, y_test = train_test_split(all_cases, labels,
                                                        test_size=0.2, random_state=42, stratify=labels)
    print(
        f"Dataset lenghts: X_train: {len(X_train)} X_test:{len(X_test)} y_train: {len(y_train)} y_test: {len(y_test)}", )
    return X_train, X_test, y_train, y_test


@hydra.main(config_path="config", config_name="config_cam", version_base='1.1')
def main(cfg: DictConfig) -> None:
    if "checkpoint" not in cfg.keys():
        print("Please specify checkpoint with `+checkpoint=path`")
        return

    print(OmegaConf.to_yaml(cfg))
    print(f"Running {cfg.project}, Work in {os.getcwd()}")

    kits_path = '/gpfs/space/home/joonas97/nnUNet/nn_pipeline/nnUNet_preprocessed/Task500_KITS_and_LITS/nnUNetData_plans_v2.1_stage1/KITS*.npz'
    lits_path = '/gpfs/space/home/joonas97/nnUNet/nn_pipeline/nnUNet_preprocessed/Task500_KITS_and_LITS/nnUNetData_plans_v2.1_stage1/LITS*.npz'

    logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading model")
    model = monai.networks.nets.EfficientNetBN(cfg.model.name, spatial_dims=3, in_channels=1, num_classes=2).to(
        device)
    model.load_state_dict(torch.load(os.path.join('/gpfs/space/home/joonas97', cfg.checkpoint)))

    summary(model, (1, 200, 512, 512))
    cam_layers = ["_blocks.0", "_blocks.3", "_blocks.6"]
    scorecam_objects = {}
    for layer in cam_layers:
        scorecam_objects[layer] = ScoreCAM(dict(type='efficientnet', arch=model, layer_name=layer))

    logging.info("Collecting data")
    # create a training data loader
    X_train, X_test, y_train, y_test = collect_and_split_data_files_old(kits_path, lits_path)

    _, val_loader = create_and_setup_dataloaders_old(X_train, X_test, y_train, y_test,
                                                     batch_size=1)

    logging.info("Extracting CAM maps")
    for i, batch_data in tqdm(enumerate(val_loader), ascii=True, total=50):

        filename = batch_data[0].meta["filename_or_obj"][0].split("/")[-1].split(".")[0]
        inputs, labels = batch_data[0][:, 0, :, :, :].to(device), batch_data[1].to(device)
        inputs = torch.unsqueeze(inputs, 1)

        cams = []
        with torch.cuda.amp.autocast():
            for layer, cam_object in scorecam_objects.items():
                cam, predicted_class = cam_object(inputs)
                cams.append([cam.cpu(), layer])

        labels = labels.cpu()
        inputs = inputs.cpu()

        everything = np.array([cams, inputs, labels, predicted_class.cpu()])
        np.save(f"{filename}", everything)

    logging.info("All done!")


if __name__ == "__main__":
    main()
