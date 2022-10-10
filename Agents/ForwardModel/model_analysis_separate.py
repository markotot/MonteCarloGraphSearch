from torch.utils.data import DataLoader, random_split

from Agents.ForwardModel.train_forward_model import get_output_head
from analysis_utils import *
from datasets import *
from models import *

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "fm_None_aNone_train"
file_name = f"trained_models/{model_name}_analysis.txt"
file = open(file_name, mode="w", buffering=1)

sight = 3
hidden_size = 512
data_set_size = 1100

data_set = MinigridDataset(sight=sight, val=True, normalize=False)
validation_set, test_set = random_split(data_set, [data_set_size, len(data_set) - data_set_size])

validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)
input_size = data_set.x.shape[1]

output_size_symbolic = get_output_head(data_set[0][1], output_head="symbolic").shape[0]
model_symbolic = NN_Forward_Model(input_size, output_size_symbolic, hidden_size).to(device)
model_symbolic.load_state_dict(torch.load(f"trained_models/{model_name}.ckpt"), strict=True)

output_size_local = get_output_head(data_set[0][1], output_head="local").shape[0]
model_local = NN_Forward_Model(input_size, output_size_local, hidden_size).to(device)
model_local.load_state_dict(torch.load(f"trained_models/{model_name}.ckpt"), strict=True)

output_size_world = get_output_head(data_set[0][1], output_head="world").shape[0]
model_world = NN_Forward_Model(input_size, output_size_world, hidden_size).to(device)
model_world.load_state_dict(torch.load(f"trained_models/{model_name}.ckpt"), strict=True)

i = 0
for x, y in validation_loader:
    x = x.to(device)
    y = y.to(device)

    output = model(x)

    input_state = x.cpu().detach().numpy()[0]
    label_state = y.cpu().detach().numpy()[0]
    output_state = output.round().cpu().detach().numpy()[0]
    action = int(input_state[6])

    calculate_obsolete_actions(input_state, label_state, action, output_head=head)
    if action == confusion_matrix_action:
        calculate_confusion_matrix(output_state, label_state, feature_idx)
    calculate_correlated_features_prediction(output_state, label_state, action)
    calculate_head_predictions(input_state, output_state, label_state, output_head=head, input_head=input_head)

    if i % 100 == 0:
        print(f"Data samples processed: {i}")
    i += 1

print_analysis(file)
if head == "symbolic":
    print_symbolic(file)
elif head == "local":
    print_local(file)
elif head == "world":
    print_world(file)
else:
    print_symbolic(file)
    print_local(file)
    print_world(file)

plot_prediction_images(10)
