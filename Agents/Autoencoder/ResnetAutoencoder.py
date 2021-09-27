import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from torchvision.models.resnet import resnet18, resnet152
from tqdm import trange


class ResNet_VAE(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=256, drop_p=0.3, CNN_embed_dim=128):
        super(ResNet_VAE, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        x = self.relu(self.fc_bn4(self.fc4(z)))
        x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = torch.nn.functional.interpolate(x, size=(128, 128), mode='bilinear')
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)

        return x_reconst, z, mu, logvar

#%% Load data
#dataset = np.load("../../minigrid_dataset.npy")
dataset = np.load("minigrid_dataset.npy")
dataset = dataset
dataset = torch.from_numpy(dataset).float()
my_dataset = TensorDataset(dataset) # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=128) # create your dataloader

#%%

device = 'cuda'
loss = nn.BCELoss(reduction="sum")

def loss_criterion(inputs, targets, mu, logvar):
    # Reconstruction loss

    bce_loss = loss(inputs, targets)
    # Regularization term
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce_loss + kl_loss

def loss_function(recon_x, x, mu, logvar):
    BCE = loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(autoencoder, optimizer, data, epochs=200):

    losses = []
    loop = trange(epochs)
    for epoch in loop:
        epoch_losses = []
        autoencoder.train(True)  # For training
        for i, x in enumerate(data):
            x = x[0].to(device)
            optimizer.zero_grad()

            X_reconst, z, mu, logvar = autoencoder(x)  # VAE
            loss = loss_criterion(X_reconst, x, mu, logvar)
            loss.backward()
            loop.set_description(f"Loss:{loss.item()}")
            epoch_losses.append(loss.item())
            optimizer.step()
        losses.append(np.mean(epoch_losses))
        if epoch % 100 == 0 and epoch != 0:
            plt.plot(losses[-100:])
            plt.show()
    return losses

model = ResNet_VAE()

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
losses = train(model, optimizer, my_dataloader)
plt.plot(losses)
plt.show()

#%%
x = dataset[np.random.choice(range(len(dataset)), size=2)]
with torch.no_grad():
    result = model(x.to(device))
x = np.transpose(x[0].cpu().numpy().squeeze(), axes=[1, 2, 0])
result = np.transpose(result[0][0].cpu().numpy().squeeze(), axes=[1, 2, 0])
plt.subplot(1, 2, 1)
plt.imshow(x)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(result)

plt.axis('off')
plt.show()
print(result.sum())






