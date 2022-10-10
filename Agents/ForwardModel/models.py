import torch
import torch.nn as nn


class NN_Forward_Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(NN_Forward_Model, self).__init__()

        self.input = nn.Linear(input_size, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_3 = nn.Linear(hidden_size, hidden_size)
        self.hidden_4 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.input(x)
        out = self.relu(out)
        out = self.hidden_1(out)
        out = self.relu(out)
        out = self.hidden_2(out)
        out = self.relu(out)
        out = self.hidden_3(out)
        out = self.relu(out)
        out = self.hidden_4(out)
        out = self.relu(out)
        out = self.output(out)
        return out

class NN_Separated_Forward_Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, device):
        super(NN_Separated_Forward_Model, self).__init__()

        self.device = device

        self.input = nn.Linear(input_size, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)
        # self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        # self.hidden_3 = nn.Linear(hidden_size, hidden_size)
        # self.hidden_4 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):

        # symbolic = x[0]
        # action = x[1]
        # local = x[2]
        # world = x[3]

        # x = torch.cat((x[0], x[1], x[2]), -1).to(self.device, non_blocking=True)

        out = self.input(x)
        out = self.relu(out)
        out = self.hidden_1(out)
        out = self.relu(out)
        # out = self.hidden_2(out)
        # out = self.relu(out)
        # out = self.hidden_3(out)
        # out = self.relu(out)
        # out = self.hidden_4(out)
        # out = self.relu(out)
        out = self.output(out)
        return out