import torch.nn as nn


class NN_Forward_Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):

        super(NN_Forward_Model, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)


    def forward (self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out


class NN_Multihead_Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):

        super(NN_Multihead_Model, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)

        # Multihead output
        self.x_output = nn.Linear(hidden_size, output_size['x_out'])
        self.y_output = nn.Linear(hidden_size, output_size['y_out'])
        self.rot_output = nn.Linear(hidden_size, output_size['rot_out'])
        self.key_output = nn.Linear(hidden_size, output_size['key_out'])
        self.door_opened_output = nn.Linear(hidden_size, output_size['door_opened_out'])
        self.door_unlocked_output = nn.Linear(hidden_size, output_size['door_unlocked_out'])

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):

        hidden = self.l1(x)
        hidden = self.relu(hidden)
        hidden = self.l2(hidden)

        x_out = self.x_output(hidden)
        # x_out = self.softmax(x_out)

        y_out = self.y_output(hidden)
        # y_out = self.softmax(y_out)

        rot_out = self.rot_output(hidden)
        # rot_out = self.softmax(rot_out)

        key_out = self.key_output(hidden)
        door_opened_output = self.door_opened_output(hidden)
        door_unlocked_output = self.door_unlocked_output(hidden)

        return x_out, y_out, rot_out, key_out, door_opened_output, door_unlocked_output