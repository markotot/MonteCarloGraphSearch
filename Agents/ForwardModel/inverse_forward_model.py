from torch.utils.data import DataLoader, random_split

from models import *
from datasets import *

if __name__ == "__main__":

    torch.manual_seed(0)

    data_set = Gridworld_Local_SAS_Dataset(sight=3, val=False)
    data_set_size = 50000

    train_set, test_set = random_split(data_set, [data_set_size, len(data_set) - data_set_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = data_set.x.shape[1]
    output_size = data_set.y.shape[1]

    hidden_size = 64
    learning_rate = 0.0001
    num_epochs = 4000

    model = NN_Forward_Model(input_size, output_size, hidden_size).to(device)
    #critetion = nn.CrossEntropyLoss()
    critetion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)

    i = 0

    for epoch in range(num_epochs):

        epoch_train_loss = 0
        epoch_test_loss = 0
        for input, labels in train_loader:

            input = input.to(device)
            labels = labels.to(device)

            #forward
            output = model(input)

            loss = critetion(output, labels)
            epoch_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:

            for input, labels in test_loader:
                input = input.to(device)
                labels = labels.to(device)

                # forward
                output = model(input)

                loss = critetion(output, labels)
                epoch_test_loss += loss.item()

            print(f"Epoch: {epoch}\t Train loss: {epoch_train_loss / len(train_loader)}\t Test loss: {epoch_test_loss / len(test_loader)}")

    torch.save(model.state_dict(), "model.ckpt")