import torch.optim
from torch.utils.data import Dataset
import pandas as pd


class Gridworld_SSA_Dataset(Dataset):
    def __init__(self):
        data = pd.read_csv("../../Data/Datasets/dataset_16x16_SSA.txt", delimiter=",")
        data = data.drop_duplicates()
        self.x = torch.tensor(data.iloc[:, 0:12].values).type(torch.FloatTensor)
        self.y = torch.tensor(data.iloc[:, 12:13].values).type(torch.FloatTensor)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class Gridworld_SAS_Dataset(Dataset):
    def __init__(self):
        data = pd.read_csv("../../Data/Datasets/dataset_16x16_SSA.txt", delimiter=",")
        data = data.drop_duplicates()

        state = torch.tensor(data.iloc[:, 0:6].values).type(torch.FloatTensor)
        next_state = torch.tensor(data.iloc[:, 6:12].values).type(torch.FloatTensor)
        action = torch.tensor(data.iloc[:, 12:13].values).type(torch.FloatTensor)

        self.x = torch.cat((state, action), -1)
        self.y = next_state

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]



class Gridworld_SSA_One_Hot_Dataset(Dataset):
    def __init__(self):
        data = pd.read_csv("../../Data/Datasets/dataset_one_hot_16x16_SSA.txt", delimiter=",")
        data = data.drop_duplicates()
        self.x = torch.tensor(data.iloc[:, 0:78].values).type(torch.FloatTensor)
        self.y = torch.tensor(data.iloc[:, 78:85].values).type(torch.FloatTensor)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class Gridworld_SAS_One_Hot_Dataset(Dataset):
    def __init__(self):
        data = pd.read_csv("../../Data/Datasets/dataset_one_hot_16x16_SSA.txt", delimiter=",")
        data = data.drop_duplicates()

        state = torch.tensor(data.iloc[:, 0:39].values).type(torch.FloatTensor)
        next_state = torch.tensor(data.iloc[:, 39:78].values).type(torch.FloatTensor)
        action = torch.tensor(data.iloc[:, 78:85].values).type(torch.FloatTensor)

        self.x = torch.cat((state, action), -1)
        self.y = next_state

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index, 0:16], self.y[index, 16:32], self.y[index, 32:36],\
               self.y[index, 36:37], self.y[index, 37:38], self.y[index, 38:39]
