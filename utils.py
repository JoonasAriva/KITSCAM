import torch
import numpy as np
from torchvision import transforms


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mu=None, std=None):
        super(CustomDataset, self).__init__()
        # store the raw tensors

        self._x = np.load(data_dir + '/data.npy')

        self._y = np.load(data_dir + '/labels.npy')

        print("data loaded")
        print("Shape: ", self._x.shape)
        if mu is None or std is None:
            print("calculating mean and std")
            self.mu = np.mean(self._x)
            self.std = np.std(self._x)
            print("std and mu calculated")
        else:
            self.mu = mu
            self.std = std

        self.compose = transforms.Compose([
            transforms.Normalize((self.mu), (self.std))])
        self.classes = ["no_kidney", "kidney"]

    def __len__(self):
        # a DataSet must know its size
        return self._x.shape[0]

    def __getitem__(self, index):

        x = self._x[index, :]
        x = torch.from_numpy(x)
        y = self._y[index]

        if len(x.shape) == 2:
            x = torch.stack([x, x, x], 0)

        x = self.compose(x)
        return x, y
