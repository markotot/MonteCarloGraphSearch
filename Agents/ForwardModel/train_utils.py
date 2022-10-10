import numpy as np

from datasets import *
from models import *


def log_loss_tensorboard(writer, train_loss, test_loss, test_accuracy, epoch):
    writer.add_scalars('Loss',
                       {'train': train_loss,
                        'test': test_loss,
                        },
                       epoch)
    writer.add_scalars('Accuracy',
                       {'test': test_accuracy
                        },
                       epoch
                       )
    writer.flush()


def log_weights_tensorboard(writer, model, epoch):
    writer.add_histogram('0_input_layer', model.input.weight, global_step=epoch)
    writer.add_histogram('1_hidden_layer', model.hidden_1.weight, global_step=epoch)
    writer.add_histogram('2_hidden_layer', model.hidden_2.weight, global_step=epoch)
    writer.add_histogram('3_hidden_layer', model.hidden_3.weight, global_step=epoch)
    writer.add_histogram('4_output_layer', model.output.weight, global_step=epoch)

    writer.flush()


def count_correct_prediction(output, label):
    output_states = output.round().cpu().detach().numpy()
    label_states = label.cpu().detach().numpy()

    correct_predictions = 0
    incorrect_predictions = 0

    for i in range(len(output_states)):
        if np.equal(output_states[i], label_states[i]).all():
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    return correct_predictions, incorrect_predictions


def augment_dataset(train_set):
    additional_indices = []
    for index in train_set.indices:
        input_data, label_data = train_set.dataset[index]
        if torch.equal(input_data[7 + 49:7 + 49 + 256], label_data[2]):
            pass
        else:
            additional_indices.insert(0, index)

    additional_indices = np.array(additional_indices).repeat(20)
    for index in additional_indices:
        train_set.indices.append(index)


def separate_dataset(data_set, action):
    separated_dataset = []
    action = torch.tensor(float(action))
    for index in data_set.indices:
        input_data, label_data = data_set.dataset[index]
        if torch.isclose(input_data[1], action):
            separated_dataset.insert(0, index)
    data_set.indices = separated_dataset


def log_hyperparams(file, model_name, model_params, dataset_params):
    file.write(f"Model: {model_name}\n")
    file.write(f"Model params: {model_params}\n")
    file.write(f"Model params: {dataset_params}\n\n")


def print_info(model_name, dataset_params, model_params, validation_set, train_set):
    print(f" Normalize: {dataset_params['normalize']}\t"
          f"lr: {model_params['lr']}\t"
          f"w_decay: {model_params['weight_decay']}")

    print(f"Model name: {model_name}")
    print(f"Validation:{len(validation_set)}")
    print(f"Train:{len(train_set)}\n")


def get_output_head(label, output_head):
    if output_head is None:
        head_label = torch.cat(label, -1)
    elif output_head == "symbolic":
        head_label = label[0]
    elif output_head == "local":
        head_label = label[1]
    elif output_head == "world":
        head_label = label[2]
    return head_label