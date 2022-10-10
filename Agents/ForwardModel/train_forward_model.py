import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from Utils.Logger import Logger
from models import *
from datasets import *


# tensorboard --logdir='./Agents/ForwardModel/runs/'

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
        if torch.equal(input_data[7 + 49:7 + 49 + 256], label_data):
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
        if torch.isclose(input_data[6], action):
            separated_dataset.insert(0, index)
    data_set.indices = separated_dataset


def log_hyperparams(file, model_name, model_params, dataset_params):
    file.write(f"Model: {model_name}\n")
    file.write(f"Model params: {model_params}\n")
    file.write(f"Model params: {dataset_params}\n\n")


if __name__ == "__main__":
    torch.manual_seed(0)

    model_params = {
        'lr': 0.0005,
        'weight_decay': 0.0000,
        'hidden_size': 256,
        'number_of_epochs': 5000,
        'load_model': False,
        'load_model_name': "fm_local_aNone_train_2"
    }

    dataset_params = {
        'input_head': None,
        # 'input_head': 'without_local',
        'output_head': "world",
        # 'output_head': None,
        'normalize': False,
        'train_size': 100000,
        'batch_size': 4096,
        'augment_dataset': True,
        'separate_action': None
    }

    model_name = f"fm_{dataset_params['output_head']}_a{dataset_params['separate_action']}"
    file_name = f"trained_models/{model_name}.txt"
    file = open(file_name, mode='w', buffering=1)

    train_data_set = Gridworld_Full_Transition_Dataset(sight=3,
                                                       preprocess=dataset_params['normalize'],
                                                       val=False,
                                                       normalize=False,
                                                       input_head=dataset_params['input_head'],
                                                       output_head=dataset_params['output_head'])

    validation_data_set = Gridworld_Full_Transition_Dataset(sight=3,
                                                            preprocess=dataset_params['normalize'],
                                                            val=True,
                                                            normalize=False,
                                                            input_head=dataset_params['input_head'],
                                                            output_head=dataset_params['output_head'])

    validation_set, drop_val_set = random_split(validation_data_set, [dataset_params['train_size'],
                                                                      len(validation_data_set) - dataset_params[
                                                                          'train_size']])
    train_set, drop_set = random_split(train_data_set, [len(train_data_set), 0])

    if dataset_params['augment_dataset']:
        augment_dataset(train_set)

    # enable for smaller training set
    # train_set, drop_set = random_split(train_data_set, [dataset_params['train_size'], len(train_data_set) - dataset_params['train_size']])

    if dataset_params['separate_action'] is not None:
        separate_dataset(train_set, dataset_params['separate_action'])
        separate_dataset(validation_set, dataset_params['separate_action'])

    train_loader = DataLoader(train_set, batch_size=dataset_params['batch_size'], shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=dataset_params['batch_size'], shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = train_data_set.x.shape[1]
    output_size = train_data_set.y.shape[1]
    model = NN_Forward_Model(input_size, output_size, model_params['hidden_size']).to(device)

    if model_params['load_model']:
        model.load_state_dict(torch.load(f"trained_models/{model_params['load_model_name']}.ckpt"), strict=True)

    critetion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'], weight_decay=model_params['weight_decay'])

    i = 0
    print(f"Train data size: {len(train_data_set)}\t"
          f" Normalize: {dataset_params['normalize']}\t"
          f"lr: {model_params['lr']}\t"
          f"w_decay: {model_params['weight_decay']}")

    print(f"Model name: {model_name}")
    print(f"Validation:{len(validation_set)}")
    print(f"Train:{len(train_set)}")
    writer = SummaryWriter(
        f"runs/Minigrid/{model_name}/lr_{model_params['lr']}, w_decay_{model_params['weight_decay']}")
    # writer.add_graph(model, data_set.x.to(device))
    writer.flush()

    log_hyperparams(file, model_name, model_params, dataset_params)

    for epoch in range(model_params['number_of_epochs']):

        epoch_train_loss = 0
        epoch_test_loss = 0
        for input, labels in train_loader:
            input = input.to(device)
            labels = labels.to(device)

            # forward
            output = model(input)

            loss = critetion(output, labels)
            epoch_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            correct_predictions = 0
            incorrect_predictions = 0
            for input, labels in validation_loader:
                input = input.to(device)
                labels = labels.to(device)

                # forward
                output = model(input)
                loss = critetion(output, labels)
                epoch_test_loss += loss.item()
                cor, inc = count_correct_prediction(output, labels)
                correct_predictions += cor
                incorrect_predictions += inc

            average_train_loss = epoch_train_loss / len(train_loader)
            average_test_loss = epoch_test_loss / len(validation_loader)
            accuracy = correct_predictions / (correct_predictions + incorrect_predictions)

            file.write(f"{Logger.time_now()} - Epoch: {epoch}\t Train loss: {average_train_loss}\t"
                       f"Test loss: {average_test_loss}\t Accuracy: {accuracy}\n")

            print(f"Epoch: {epoch}\t Train loss: {average_train_loss}\t Test loss: {average_test_loss}")
            print(f"Accuracy: {accuracy}")
            log_loss_tensorboard(writer=writer, train_loss=average_train_loss, test_loss=average_test_loss,
                                 test_accuracy=accuracy, epoch=epoch)
            log_weights_tensorboard(writer=writer, model=model, epoch=epoch)

        if epoch % 200 == 0:
            torch.save(model.state_dict(), f"trained_models/{model_name}_train.ckpt")
    torch.save(model.state_dict(), f"trained_models/{model_name}.ckpt")
    file.close()
    writer.close()
