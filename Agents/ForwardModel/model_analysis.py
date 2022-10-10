from torch.utils.data import DataLoader, random_split

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

# head = "world"
head = None
# input_head = "without_local"
input_head = None

data_set_size = 1100

data_set = Gridworld_Full_Transition_Dataset(sight=sight, preprocess=False, val=True, output_head=head)
validation_set, test_set = random_split(data_set, [data_set_size, len(data_set) - data_set_size])

validation_loader = DataLoader(validation_set, batch_size=1, shuffle=True)
input_size = data_set.x.shape[1]
output_size = data_set.y.shape[1]

model = NN_Forward_Model(input_size, output_size, hidden_size).to(device)
model.load_state_dict(torch.load(f"trained_models/{model_name}.ckpt"), strict=True)

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
