import numpy as np
import torch
from torch import optim, nn


def eval_metrics(pred, true):
    """
    Calculate several metrics for evaluating model performance.

    Args:
        pred (np.array): Predicted values.
        true (np.array): Ground truth values.

    Returns:
        tuple: Tuple of floats containing the mean absolute error (MAE),
            mean squared error (MSE), root mean squared error (RMSE),
            mean absolute percentage error (MAPE), and mean squared percentage
            error (MSPE).
    """

    # Calculate each metric using their respective functions
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred - true) / true)) * 100
    mspe = np.mean(np.square((pred - true) / true)) * 100

    # Return a tuple of the metrics
    return mae, mse, rmse, mape, mspe


def adjust_learning_rate(optimizer, epoch, args):
    if args.lr_adj == 'type1':
        # Type 1 learning rate adjustment
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))

    elif args.lr_adj == 'type2':
        # Type 2 learning rate adjustment
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
    else:
        # Default learning rate
        lr = args.learning_rate

    # Update optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Print learning rate update message
    print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, print_info=False, tolerance=0):
        """
        Args:
        - patience: (int) number of epochs to wait before stopping the training process
        - print_info: (bool) whether to print information about the stopping process
        - tolerance: (float) minimum change in the monitored metric to qualify as an improvement
        """
        self.patience = patience
        self.print_info = print_info
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.tolerance = tolerance

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.tolerance:
            self.counter += 1
            if self.print_info:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.print_info:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
