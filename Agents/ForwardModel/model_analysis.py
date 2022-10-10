from torch.utils.data import DataLoader, random_split

from train_forward_model import get_output_head
from analysis_utils import *
from datasets import *
from models import *

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hidden_size = 128
output_head = "symbolic"
separate_action = None
# output_head = None

model_name = f"fm_{output_head}_a{separate_action}_train"
file_name = f"trained_models/{model_name}_analysis.txt"
file = open(file_name, mode="w", buffering=1)

# input_head = "without_local"
input_head = None

sight = 3
data_set_size = 1100

data_set = MinigridDataset(sight=sight, val=True)
validation_set, test_set = random_split(data_set, [data_set_size, len(data_set) - data_set_size])

validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)
input_size = len(data_set[0][0][0]) + len(data_set[0][0][1]) + len(data_set[0][0][2])
output_size = get_output_head(data_set[0][1], output_head=output_head).shape[0]

model = NN_Separated_Forward_Model(input_size, output_size, hidden_size, device).to(device)
model.load_state_dict(torch.load(f"trained_models/{model_name}.ckpt"), strict=True)

i = 0
for x, y in validation_loader:

    y = get_output_head(y, output_head)
    y = y.to(device, non_blocking=True)

    output = model(x)

    input_state = x.cpu().detach().numpy()[0]
    label_state = y.cpu().detach().numpy()[0]
    output_state = output.round().cpu().detach().numpy()[0]
    action = int(input_state[6])

    calculate_obsolete_actions(input_state, label_state, action, output_head=output_head)
    if action == confusion_matrix_action:
        calculate_confusion_matrix(output_state, label_state, feature_idx)
    calculate_correlated_features_prediction(output_state, label_state, action)
    calculate_head_predictions(input_state, output_state, label_state, output_head=output_head, input_head=input_head)

    if i % 100 == 0:
        print(f"Data samples processed: {i}")
    i += 1

print_analysis(file)
if output_head == "symbolic":
    print_symbolic(file)
elif output_head == "local":
    print_local(file)
elif output_head == "world":
    print_world(file)
else:
    print_symbolic(file)
    print_local(file)
    print_world(file)

plot_prediction_images(10)
