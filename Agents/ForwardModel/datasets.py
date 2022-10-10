import pandas as pd
import torch.optim
from torch.utils.data import Dataset


def normalize(x):
    processed = x - torch.mean(x, axis=0)
    processed /= torch.std(processed, axis=0)
    return processed


class MinigridDataset(Dataset):
    def __init__(self, val=False, sight=1):

        if val is False:
            data = pd.read_csv("../../Data/Datasets/dataset_16x16_Full_Transition.txt", delimiter=",")
        else:
            data = pd.read_csv("../../Data/Datasets/dataset_16x16_Full_Transition_val.txt", delimiter=",")

        num_features = 6
        size_of_surroundings = (sight * 2 + 1) ** 2
        size_of_world = 256

        symbolic = torch.tensor(data.iloc[:, 0:6].values).type(torch.FloatTensor)
        next_symbolic = torch.tensor(data.iloc[:, 6:12].values).type(torch.FloatTensor)
        action = torch.tensor(data.iloc[:, 12:13].values).type(torch.FloatTensor)

        local_start = num_features * 2 + 1
        local = torch.tensor(data.iloc[:, local_start:local_start + size_of_surroundings].values).type(torch.FloatTensor)
        local = local.reshape(local.shape[0], 7, 7)
        local = torch.flatten(local[:][:, 2:5][:, :, 2:5], start_dim=1, end_dim=2)

        next_local = torch.tensor(data.iloc[:, local_start + size_of_surroundings:local_start + 2 * size_of_surroundings].values).type(torch.FloatTensor)
        next_local = next_local.reshape(next_local.shape[0], 7, 7)
        next_local = torch.flatten(next_local[:][:, 2:5][:, :, 2:5], start_dim=1, end_dim=2)

        world_start = local_start + 2 * size_of_surroundings
        world = torch.tensor(data.iloc[:, world_start:world_start + size_of_world].values).type(torch.FloatTensor)
        next_world = torch.tensor(data.iloc[:, world_start + size_of_world:world_start + 2 * size_of_world].values).type(torch.FloatTensor)

        self.x_symbolic = symbolic
        self.x_action = action
        self.x_local = local
        self.x_world = world

        self.y_symbolic = next_symbolic
        self.y_local = next_local
        self.y_world = next_world

    def __len__(self):
        return len(self.x_symbolic)

    def __getitem__(self, index):
        x = (self.x_symbolic[index], self.x_action[index], self.x_local[index], self.x_world[index])
        y = (self.y_symbolic[index], self.y_local[index], self.y_world[index])
        return x, y


class MinigridDatasetProfiler(Dataset):
    def __init__(self, val=False, sight=1):

        if val is False:
            data = pd.read_csv("../../Data/Datasets/dataset_16x16_Full_Transition.txt", delimiter=",")
        else:
            data = pd.read_csv("../../Data/Datasets/dataset_16x16_Full_Transition_val.txt", delimiter=",")

        num_features = 6
        size_of_surroundings = (sight * 2 + 1) ** 2
        size_of_world = 256

        symbolic = torch.tensor(data.iloc[:, 0:6].values).type(torch.FloatTensor)
        next_symbolic = torch.tensor(data.iloc[:, 6:12].values).type(torch.FloatTensor)
        action = torch.tensor(data.iloc[:, 12:13].values).type(torch.FloatTensor)

        local_start = num_features * 2 + 1
        local = torch.tensor(data.iloc[:, local_start:local_start + size_of_surroundings].values).type(torch.FloatTensor)
        local = local.reshape(local.shape[0], 7, 7)
        local = torch.flatten(local[:][:, 2:5][:, :, 2:5], start_dim=1, end_dim=2)

        next_local = torch.tensor(data.iloc[:, local_start + size_of_surroundings:local_start + 2 * size_of_surroundings].values).type(torch.FloatTensor)
        next_local = next_local.reshape(next_local.shape[0], 7, 7)
        next_local = torch.flatten(next_local[:][:, 2:5][:, :, 2:5], start_dim=1, end_dim=2)

        world_start = local_start + 2 * size_of_surroundings
        world = torch.tensor(data.iloc[:, world_start:world_start + size_of_world].values).type(torch.FloatTensor)
        next_world = torch.tensor(data.iloc[:, world_start + size_of_world:world_start + 2 * size_of_world].values).type(torch.FloatTensor)

        self.x = torch.cat((symbolic, action, local), -1)
        self.y = next_symbolic
        # self.x_symbolic = symbolic
        # self.x_action = action
        # self.x_local = local
        # self.x_world = world

        self.y_symbolic = next_symbolic
        self.y_local = next_local
        self.y_world = next_world

    def __len__(self):
        return len(self.x_symbolic)

    def __getitem__(self, index):
        # x = (self.x_symbolic[index], self.x_action[index], self.x_local[index], self.x_world[index])
        # y = (self.y_symbolic[index], self.y_local[index], self.y_world[index])
        return self.x, self.y
