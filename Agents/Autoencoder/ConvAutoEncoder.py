import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvAutoEncoder(nn.Module):

    def __init__(self, lr=0.001):
        super(ConvAutoEncoder, self).__init__()

        #Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.enc_linear1 = nn.Linear(4 * 32 * 32, 512)
        self.enc_linear2 = nn.Linear(512, 256)
        self.enc_linear3 = nn.Linear(256, 128)
        #Decoder
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, [4, 32, 32])
        self.dec_linear3 = nn.Linear(128, 256)
        self.dec_linear2 = nn.Linear(256, 512)
        self.dec_linear1 = nn.Linear(512, 4 * 32 * 32)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2, stride=2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
      #  x = F.relu(self.enc_linear1(self.flatten(x)))
      #  x = F.relu(self.enc_linear2(x))
      #  x = F.relu(self.enc_linear3(x))

      #  x = F.relu(self.dec_linear3(x))
      #  x = F.relu(self.dec_linear2(x))
      #  x = F.relu(self.dec_linear1(x))
      #  x = F.relu(self.t_conv2(self.unflatten(x)))
        x = F.relu(self.t_conv2(x))
        x = T.sigmoid(self.t_conv1(x))

        return x
