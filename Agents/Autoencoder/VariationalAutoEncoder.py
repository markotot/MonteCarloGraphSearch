import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
import matplotlib.pyplot as plt
from time import sleep
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.enc_linear1 = nn.Linear(4 * 32 * 32, 512)
        self.enc_linear2 = nn.Linear(512, 256)
        self.enc_linear3 = nn.Linear(256, 128)
        self.flatten = nn.Flatten()

        self.var_linear2 = nn.Linear(128, 128)
        self.var_linear3 = nn.Linear(128, 128)

        self.N = T.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.enc_linear1(self.flatten(x)))
        x = F.relu(self.enc_linear2(x))
        x = F.relu(self.enc_linear3(x))

        mu = self.var_linear2(x)
        sigma = T.exp(self.var_linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - T.log(sigma) - 1/2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.unflatten = nn.Unflatten(1, [4, 32, 32])
        self.dec_linear3 = nn.Linear(128, 256)
        self.dec_linear2 = nn.Linear(256, 512)
        self.dec_linear1 = nn.Linear(512, 4 * 32 * 32)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.dec_linear3(x))
        x = F.relu(self.dec_linear2(x))
        x = F.relu(self.dec_linear1(x))
        x = F.relu(self.t_conv2(self.unflatten(x)))
        x = T.sigmoid(self.t_conv1(x))
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder()
        self.decoder = Decoder()

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)



#%% Load data
#dataset = np.load("../../minigrid_dataset.npy")
dataset = np.load("minigrid_dataset.npy")
dataset = dataset
dataset = T.from_numpy(dataset).float()
my_dataset = TensorDataset(dataset) # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=128) # create your dataloader

#%%
def train(autoencoder, data, epochs=200):
    opt = T.optim.Adam(autoencoder.parameters())
    losses = []
    loop = trange(epochs)

    for epoch in loop:
        epoch_losses = []
        for x in data:
            x = x[0].to(autoencoder.device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            loop.set_description(f"Loss:{loss.item()}")
            epoch_losses.append(loss.item())
            opt.step()
        losses.append(np.mean(epoch_losses))
        if epoch % 100 == 0 and epoch != 0:
            plt.plot(losses[-100:])
            plt.show()

    return autoencoder, losses

vae = VariationalAutoencoder()# GPU
vae = vae.to(vae.device)
vae, losses = train(vae, my_dataloader)
plt.plot(losses)
plt.show()

#%%
x = dataset[np.random.choice(range(len(dataset)))]
with T.no_grad():
    result = vae(x[None, ].cuda())
x = np.transpose(x.cpu().numpy().squeeze(), axes=[1, 2, 0])
result = np.transpose(result.cpu().numpy().squeeze(), axes=[1, 2, 0])
plt.subplot(1, 2, 1)
plt.imshow(x)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(result)

plt.axis('off')
plt.show()
print(result.sum())
