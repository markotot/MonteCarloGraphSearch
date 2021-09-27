from Agents.Autoencoder.ConvAutoEncoder import ConvAutoEncoder
import matplotlib.pyplot as plt
import Agents.Autoencoder.DataCreator as DataCreator
import numpy as np
import torch as T

n_epochs = 10
num_samples = 1e4
batch_size = 50
model = ConvAutoEncoder()
images = np.array(DataCreator.create_data(num_samples))
images = images / 255
images = np.moveaxis(images, 3, 1) # move channels from 3 -> 1 axis

np.save("minigrid_dataset", images)

#%%
for epoch in range(1, n_epochs + 1):

    train_loss = 0.0

    for i in range(int(num_samples / batch_size)):
        batch = images[i * batch_size: (i + 1) * batch_size]
        batch = T.from_numpy(batch).to(model.device).float()
        model.optimizer.zero_grad()
        outputs = model(batch)
        loss = model.loss(outputs, batch)
        loss.backward()
        model.optimizer.step()
        train_loss = loss.item() * batch.size(0)

    train_loss = train_loss / len(batch)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

#%%
images = T.from_numpy(images[0:5]).to(model.device).float()
outs = model(images[0:5])
out = outs.cpu().detach().numpy()[0]
out = np.transpose(out, axes=[1, 2, 0])

input = images.cpu().detach().numpy()[0]
input = np.transpose(input, axes=[1, 2, 0])


plt.imshow(input)
plt.show()
plt.imshow(out)
plt.show()
plt.close()
