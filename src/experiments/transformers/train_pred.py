import os
import sys
import time

import mlflow
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.transformers.data_loader import Dataset_Test, Dataset_Train_Val, LiveDataLoader
from data.transformers.utils import save_args_to_text

from models.transformers.models import Informer, Transformer ,Pyraformer

from .train_pred_utils import EarlyStopping, adjust_learning_rate, eval_metrics


def select_optimizer(model, args):
    optimiser = optim.Adam(model.parameters(), lr=args.learning_rate)
    return optimiser


def select_criterion():
    criterion = nn.MSELoss()
    return criterion


def build_model(args):
    model_dict = {
        'informer': Informer,
        'transformer': Transformer,
        'pyraformer' : Pyraformer
    }

    if args.model == 'informer' or args.model == 'transformer':
        model = model_dict[args.model](
            args.enc_in,
            args.dec_in,
            args.c_out,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.factor,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.dropout,
            args.attn,
            args.embed,
            args.freq,
            args.activation,
            args.output_attention,
            args.distil,
            args.mix,
            args.device
        ).float()
    elif args.model == 'pyraformer':
        model = model_dict[args.model](
            args.enc_in,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.dropout,
            args.window_size,
            args.inner_size
        ).float()

    return model


def get_data(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        timestamp = args.timestamp
        root_path = args.root_path_test
        data_path = args.data_path_test
        Data = Dataset_Test
    else:
        shuffle_flag = True  # used in papers but not sure if shuffling the data is good for now keep it
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        timestamp = args.timestamp
        root_path = args.root_path_train
        data_path = args.data_path_train
        Data = Dataset_Train_Val

    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        normalise=args.normalise,
        freq=freq,
        timestamp=args.timestamp
    )

    print(flag, len(data_set))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader


def validation(args, model, val_loader, loss_criterion):
    # Initialize an empty list to store the loss for each batch
    total_loss = []

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Loop through each batch of validation data
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
            # Move batch_x and batch_x_mark to the device
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            # Create decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)

            # Encode and decode the input sequence
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # Pick last dimension if we are using multi-variate prediction
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)

            # Detach the predictions and true values from the computation graph and move them to the CPU
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            # Calculate the loss for the batch and append it to the list of losses
            loss = loss_criterion(pred, true)
            total_loss.append(loss)

    # Calculate the average loss for all batches and return it
    total_loss = np.average(total_loss)
    model.train()

    return total_loss


def train(model, args, setting):
    train_data, train_loader = get_data(args, flag='train')
    vali_data, vali_loader = get_data(args, flag='val')

    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(patience=args.patience, print_info=True)

    #
    model_optim = select_optimizer(model, args)
    criterion = select_criterion()

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # record starting time
    start_time = time.time()

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), total=train_steps):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)
            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            # With TDQM we no longer need to print the progress keep it for now doe

            # if (i + 1) % 100 == 0:
            #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            #     speed = (time.time() - time_now) / iter_count
            #     left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
            #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            #     iter_count = 0
            #     time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = validation(args=args, model=model, val_loader=vali_loader, loss_criterion=criterion)

        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        early_stopping(vali_loss, model, path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(model_optim, epoch + 1, args)

    # record ending time
    end_time = time.time()

    # calculate total training time in seconds and average training time
    total_training_time = end_time - start_time
    average_training_time_per_epoch = total_training_time / args.train_epochs

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    # save in MLFlow

    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    experiment_name = args.experiment_name

    # Check if experiment exists, otherwise create it
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:

        # Track parameters
        filtered_args = {k: v for k, v in args.__dict__.items() if not k.startswith('__') and not k.endswith('__')}
        for arg_name, arg_value in filtered_args.items():
            mlflow.log_param(arg_name, arg_value)

        # Get the run ID
        run_id = mlflow.active_run().info.run_id
        args.run_id = run_id

        # Track metrics
        mlflow.log_metric("Num_Params", num_params)
        mlflow.log_metric("Total_Traning_Time", total_training_time)
        mlflow.log_metric("Average_traning_Time", average_training_time_per_epoch)

    # Save args after training to txt file
    arguments_path = path + '/' + 'args.txt'
    save_args_to_text(args, arguments_path)

    return model, args


def test(model, args, setting):
    test_data, test_loader = get_data(args, flag='test')

    if test:
        start_time = time.time()  # Record the start time
        print('loading model')
        model.load_state_dict(torch.load(os.path.join('../experiments/checkpoints/' + setting, 'checkpoint.pth')))
        end_time = time.time()  # Record the end time
        load_time = end_time - start_time  # Calculate the time it took to load the model

    preds = []
    trues = []

    start_time = time.time()  # Record the start time for predictions
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            batch_x = batch_x.float().to(args.device)
            batch_y = batch_y.float().to(args.device)

            batch_x_mark = batch_x_mark.float().to(args.device)
            batch_y_mark = batch_y_mark.float().to(args.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)

    end_time = time.time()  # Record the end time for predictions
    prediction_time = end_time - start_time  # Calculate the time it took to execute the predictions
    avg_prediction_time = prediction_time / len(test_loader)

    preds = np.array(preds)
    trues = np.array(trues)
    print('test shape:', preds.shape, trues.shape)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    # result save
    folder_path = '../experiments/test_results/' + setting + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    mae, mse, rmse, mape, mspe = eval_metrics(preds, trues)
    # print('mse:{}, mae:{}'.format(mse, mae))
    print('MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%, MSPE: {:.4f}%'.format(mae, mse, rmse, mape,
                                                                                        mspe))

    # for now keep the saved np arrays but there might be a better way to store this
    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    np.save(folder_path + 'pred.npy', preds)
    np.save(folder_path + 'true.npy', trues)

    # save in MLFlow
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_id=args.run_id) as run:
        # Track metrics
        mlflow.log_metric("Model_Load_Time", load_time)
        mlflow.log_metric("Prediction_Time", prediction_time)
        mlflow.log_metric("Avg_Prediction_Time", avg_prediction_time)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAPE", mape)
        mlflow.log_metric("MSPE", mspe)

    return None


def load_model_pred(args, path):
    # Need to pass in the args to build the model will need to look into doing this from a json file (after a
    # checkpoint)
    model = build_model(args)

    print('loading model')
    model.load_state_dict(torch.load(path, map_location=args.device))

    return model


def predict_price(model, args, data_df, pred_len):
    data_set = LiveDataLoader(
        data_df=data_df,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        normalise=args.normalise,
        freq=args.freq,
        timestamp=args.timestamp
    )

    batch_x, batch_y, batch_x_mark, batch_y_mark = data_set[0]

    batch_x = torch.from_numpy(batch_x).float().to(args.device)
    batch_y = torch.from_numpy(batch_y).float().to(args.device)
    batch_x_mark = torch.from_numpy(batch_x_mark).float().to(args.device)
    batch_y_mark = torch.from_numpy(batch_y_mark).float().to(args.device)

    preds = []
    trues = []

    batch_size = 1  # since you are passing the whole data as input

    model.eval()
    with torch.no_grad():
        # decoder input
        dec_inp = torch.zeros(batch_size, args.label_len + args.pred_len, 1).float()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(args.device)

        # encoder - decoder
        if args.use_amp:
            with torch.cuda.amp.autocast():
                if args.output_attention:
                    outputs = \
                        model(batch_x[:batch_size], batch_x_mark[:batch_size], dec_inp, batch_y_mark[:batch_size])[0]
                else:
                    outputs = model(batch_x[:batch_size], batch_x_mark[:batch_size], dec_inp, batch_y_mark[:batch_size])
        else:
            if args.output_attention:
                outputs = model(batch_x[:batch_size], batch_x_mark[:batch_size], dec_inp, batch_y_mark[:batch_size])[0]
            else:
                outputs = model(batch_x[:batch_size], batch_x_mark[:batch_size], dec_inp, batch_y_mark[:batch_size])

        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(args.device)
        outputs = outputs.detach().cpu().numpy()
        batch_y = batch_y.detach().cpu().numpy()

        pred = outputs
        true = batch_y

        preds.append(pred)
        trues.append(true)

    return preds
