from torch.utils.data import DataLoader, random_split

from models import *
from datasets import *


torch.manual_seed(0)

data_set = Gridworld_SAS_Dataset()
train_set, test_set = random_split(data_set, [2000, len(data_set) - 2000])
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = data_set.x.shape[1]
output_size = data_set.y.shape[1]
hidden_size = 64
learning_rate = 0.0001
num_epochs = 1000

model = NN_Forward_Model(input_size, output_size, hidden_size).to(device)
model.load_state_dict(torch.load("model.ckpt"), strict=True)

correct = 0
incorrect = 0

#actions_number = [0] * (output_size + 6)
for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)

    output = model(x)
    #selected_action = output.round().item()
    #actions_number[int(selected_action)] += 1
    if torch.equal(output.round(), y.round()):
        correct += 1
    else:
        incorrect += 1

print(f"Correct: {correct}\t Incorrect: {incorrect}\t Percentage: {correct / (correct + incorrect)}")
#print(f"Output numbers: {actions_number}")