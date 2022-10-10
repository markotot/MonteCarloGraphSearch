from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import timeit

from Utils.Logger import Logger
from train_utils import *

# tensorboard --logdir='./Agents/ForwardModel/runs/'




if __name__ == "__main__":
    torch.manual_seed(0)

    model_params = {
        'lr': 0.0005,
        'weight_decay': 0.0000,
        'hidden_size': 128,
        'number_of_epochs': 5000,
        'load_model': False,
        'load_model_name': "fm_local_a2_train"
    }

    dataset_params = {
        'input_head': None,
        # 'input_head': 'without_local',
        'output_head': "symbolic",
        # 'output_head': None,
        'normalize': False,
        'train_size': 100000,
        'batch_size': 4096,
        'augment_dataset': False,
        'separate_action': None
    }

    model_name = f"fm_{dataset_params['output_head']}_a{dataset_params['separate_action']}"
    file_name = f"trained_models/{model_name}.txt"
    file = open(file_name, mode='w', buffering=1)

    train_data_set = MinigridDatasetProfiler(sight=3, val=False)
    validation_data_set = MinigridDatasetProfiler(sight=3, val=True)

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

    train_loader = DataLoader(train_set, batch_size=dataset_params['batch_size'], shuffle=True, num_workers=0,
                              pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=dataset_params['batch_size'], shuffle=True, num_workers=0,
                                   pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = len(train_data_set[0][0][0]) + len(train_data_set[0][0][1]) + len(train_data_set[0][0][2])
    output_size = get_output_head(train_data_set[0][1], dataset_params['output_head']).shape[0]

    model = NN_Separated_Forward_Model(input_size, output_size, model_params['hidden_size'], device).to(device)

    total_params = sum(param.numel() for param in model.parameters())
    print(f"Total params: {total_params}")

    if model_params['load_model']:
        model.load_state_dict(torch.load(f"trained_models/{model_params['load_model_name']}.ckpt"), strict=True)

    critetion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=model_params['lr'], weight_decay=model_params['weight_decay'])

    i = 0
    writer = SummaryWriter(
        f"runs/Minigrid/{model_name}/lr_{model_params['lr']}, w_decay_{model_params['weight_decay']}")
    writer.flush()
    # print_info(model_name, dataset_params, model_params, validation_set, train_set)
    log_hyperparams(file, model_name, model_params, dataset_params)

    for epoch in range(model_params['number_of_epochs']):
        epoch_train_loss = 0
        epoch_test_loss = 0
        start = timeit.default_timer()
        for input, label in train_loader:
            # input = torch.cat((input[0], input[1], input[2]), -1).to(device, non_blocking=True)
            # label = get_output_head(label=label, output_head=dataset_params['output_head'])
            input = input.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # with torch.autograd.profiler.profile() as prof:

            output = model(input)
            loss = critetion(output, label)
            epoch_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 50 == 1:
            correct_predictions = 0
            incorrect_predictions = 0
            for input, label in validation_loader:

                input = input.to(device, non_blocking=True)
                # label = get_output_head(label=label, output_head=dataset_params['output_head'])
                label = label.to(device, non_blocking=True)

                # forward

                output = model(input)

                loss = critetion(output, label)
                epoch_test_loss += loss.item()

                cor, inc = count_correct_prediction(output, label)
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
            # log_weights_tensorboard(writer=writer, model=model, epoch=epoch)
        if epoch % 100 == 1:
            torch.save(model.state_dict(), f"trained_models/{model_name}_train.ckpt")

        stop = timeit.default_timer()
        print(f"One epoch: {stop - start}")
    torch.save(model.state_dict(), f"trained_models/{model_name}.ckpt")
    file.close()
    writer.close()
