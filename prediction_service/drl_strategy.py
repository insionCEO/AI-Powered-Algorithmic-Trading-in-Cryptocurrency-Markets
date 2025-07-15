"""
Created on Thu Apr 27 22:36:18 2023

@author: diogo

Hi Tomás and rest of the team! I apologise in advance for the length of the file, but hopefully this initial comment will give you
all the info you might need to better understand how to run this code! If any issue arises, don't hesitate to
send me a message. Let's get started!

There are only 2 functions you'll need to use, and one class that you'll be directly interacting with. Before you
do anything, you'll need create an object like this:
    
    model = DRLModel(training)
    
Where 'training' is a flag value. In your case, simply send the value 0 as we won't be training the model!

Now, using that object everything else should be straight forward. Two functions where created:
    
    model.load_model(portfolio, model_weights_path)
    
This loads the model into memory. The parameter 'model_weights_path' should be the directory where the weights
for the networks of the model are stored in. The weights for all of the networks should be stored together in the
same folder. The Portfolio object will need to contain a few parameters:
    
    - initial_investment: contains the value in cash of investment at the beginning of the execution
    - max_stock: the maximum stock of a currency that the algorithm can own at a time
    - max_percentile_investment: a value between 0 and 1 that represents how much (at most) the model should
    invest at a time based in the value of initial_investment
    - cash_balance: the amount of cash that the portfolio currently contains

You should pass the portfolio itself by value to be safe. Just pass a copy of the current portfolio object. This
function returns nothing.

The second function that you'll be using is the following;

    model.get_prediction(past_data, pred_data, portfolio)

This function returns an action [which can be either 0 (buy), 1 (hold) or 2 (sell)] and a volume (which
represents the amount of bitcoin the algorithm wants you to purchase. Here are some details about the parameters
you should pass to it:

    - past_data: an array with the midpoint and spread values for the past 96 data_points (including the current one)
    - pred_data: an array containing the predictions obtained by Andrzej's model
    - portfolio: a copy of the current state of the portfolio, which the function uses to update the values
    within the environment already,
    
Hopefully this is descriptive enough and explains everything that you can do and how you can obtain the actions!
Again, don't hesitate to contact me if you got any questions or run into any errors :)

"""

import gymnasium as gym
import random
import numpy as np
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# check if gpu/cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# definitions of model parameters

GAMMA = 0.99  # Discount factor
EPSILON = 0.2  # Epsilon Greedy approach to increase exploration

# Create a custom environment using the 1 second order book data of Cryptocurrency with the Trade-Completion reward function
# Maximo Transaction fee: max(1.3$, 0.013§ x volume) - Interactive Brokers


class CustomEnv(gym.Env):
    def __init__(self, data=[], initial_investment=10000, max_stock=1, look_back=144, max_percentile_investment=0.2, training_data=[], portfolio=[]):

        # check if a dataloader exists. If not, the environment was loaded solely for retrieving predictions
        if (data != []):
            self.data = data  # Load_data is a function that loads the csv data
            self.batch_size = data.batch_size
            self.num_features = len(data.dataset[0])
            self.training = 1  # indicate the model is training
        else:
            self.data = data  # Load_data is a function that loads the csv data
            self.batch_size = 1  # use the value 1 for a prediction at a time
            self.num_features = 2  # static current number of features extracted from dvc data
            self.training = 0  # indicate the model is not training

        self.action_space = gym.spaces.Discrete(3)  # buy, hold, sell
        # we add 3 features for the past RSI technical indicators
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.batch_size, self.num_features + 3))

        # checks wether the information about the portfolio is given directly or if a portfolio object is passed to the function instead
        if (portfolio == []):
            self.initial_investment = initial_investment
            self.max_stock = max_stock
            self.max_percentile_investment = max_percentile_investment

        else:
            self.initial_investment = portfolio.initial_investment
            self.max_stock = portfolio.max_stock
            self.max_percentile_investment = portfolio.max_percentile_investment
            self.portfolio = portfolio

        self.look_back = look_back
        self.training_data = training_data
        self.training_counter = 0
        self.future_pred = 0
        self.reset()

    def reset(self):
        self.current_step = 0
        self.previous_reward = 0
        self.last_trade = None
        self.current_stock = 0
        self.cash_balance = self.initial_investment
        self.total_profit = 0.0
        self.past_bids = []
        self.past_asks = []

        # check if the model is training or not, so that an observation is not sought after immediately if there is no training data
        if (self.training == 1):
            self.current_step = self._next_observation()

    def partial_reset(self):
        self.cash_balance = self.initial_investment
        self.total_profit = 0

    def initial_state(self):
        return self.current_step

    def get_current_funds(self):
        return self.cash_balance

    def get_current_profit(self):
        return self.total_profit

    def get_prediction(self):
        current_pred_tensor = []
        if (self.training == 0):
            return 0
        else:
            current_pred_list = self.training_data[self.training_counter]
            for i in current_pred_list:
                current_pred_tensor.append(torch.tensor(i).to(device))
            self.training_counter += 1
            return self.calculate_rsi(current_pred_tensor, look_back=len(current_pred_tensor))

    # function responsible for updating portfolio data for integration purposes
    def update_portfolio(self, portfolio_dict):

        self.cash_balance = portfolio_dict["portfolio_cash_balance"]
        self.current_stock = portfolio_dict["portfolio_current_stock"]
        self.max_stock = portfolio_dict["portfolio_max_stock"]
        self.max_percentile_investment = portfolio_dict[
            "portfolio_max_percentile_investment"]
        self.initial_investment = portfolio_dict["portfolio_initial_investment"]
        # self.portfolio = portfolio

    def step(self, actions):
        rewards = []

        for action, state in zip(actions, self.current_step):
            # Calculate the current ask price and bid price
            ask_price = state[1]  # position of asks
            bid_price = state[0]  # position of bids
            profit = 0
            volume = 0
            transaction_fee = 0

            if action == 0:  # Buy
                available_cash = self.cash_balance
                max_buy_volume = max(0, min(
                    float(available_cash / ask_price), (self.max_stock - self.current_stock)))
                if max_buy_volume <= 0 or self.current_stock > 0:
                    reward = -0.1
                else:
                    volume = max_buy_volume * self.max_percentile_investment
                    self.current_stock += volume
                    cost = volume * ask_price
                    self.last_trade = (ask_price, volume)
                    self.transaction_fee_percent = max(1.3, 0.013 * cost)
                    transaction_fee = self.transaction_fee_percent
                    self.cash_balance -= (cost + transaction_fee)
                    reward = 0.1

            elif action == 1:  # Hold
                reward = -0.05

            elif action == 2 and self.last_trade:  # Sell
                if self.current_stock == 0:
                    reward = -0.1
                else:
                    volume = self.current_stock
                    self.current_stock -= volume
                    sale_price = bid_price * volume
                    self.transaction_fee_percent = max(1.3, 0.013 * sale_price)
                    transaction_fee = self.transaction_fee_percent
                    self.cash_balance += (sale_price - transaction_fee)
                    profit = (
                        (sale_price - self.last_trade[0] * volume) - self.transaction_fee_percent)
                    self.total_profit += profit
                    reward = profit / \
                        (self.initial_investment * self.max_percentile_investment)
                    self.last_trade = None

            else:
                reward = -0.1

            # update the value of the last reward for when batch_size is 1
            self.previous_reward = reward

            # append the rewards regardless of if batch_size is larger than 1
            rewards.append(reward)

        trewards = torch.FloatTensor(rewards)
        trewards = trewards.to(device)

        # obtain the next observation and store it in memory in case training is being performed
        if (self.training == 1):
            self.current_step = self._next_observation()

        return self.current_step, trewards, profit, volume

    def calculate_rsi(self, prices, look_back=-1):

        if look_back == -1:
            look_back = self.look_back

        if len(prices) < 2:
            rsi = 50.0
            return torch.tensor(rsi).to(device)

        prices = torch.tensor(prices)
        deltas = torch.diff(prices, dim=0)
        gains = torch.zeros_like(deltas)
        losses = torch.zeros_like(deltas)
        mask = (deltas > 0)
        gains[mask] = deltas[mask]
        losses[~mask] = -deltas[~mask]
        avg_gain = torch.mean(gains[-look_back:], dim=0)
        avg_loss = torch.mean(losses[-look_back:], dim=0)

        if avg_loss == 0:
            return torch.tensor(0.0).to(device)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _next_observation(self):

        # run the function that calculates the rsi of the predictions
        pred_rsi = self.get_prediction()

        # stack pred RSI tensor
        pred_rsi_tensor = torch.tensor(pred_rsi).to(device)

        # reshape the tensors
        pred_rsi_tensor = pred_rsi_tensor.unsqueeze(dim=0)
        pred_rsi_tensor = pred_rsi_tensor.unsqueeze(dim=1)

        data_batch = next(self.iter_data)
        data_batch_tensor = data_batch[0].to(device)
        midpoint = torch.index_select(
            data_batch_tensor, 1, torch.tensor([0]).to(device))
        spread = torch.index_select(
            data_batch_tensor, 1, torch.tensor([1]).to(device))
        current_bid_prices = midpoint - 0.5 * spread
        current_ask_prices = midpoint + 0.5 * spread

        bid_rsis = []
        ask_rsis = []
        profit = []
        cash = []
        stock = []
        reward = []
        bid_p = []
        ask_p = []

        for data_point in data_batch_tensor:
            bid_prices = data_point[0].cpu()
            ask_prices = data_point[1].cpu()
            bid_prices = midpoint - 0.5 * spread
            ask_prices = midpoint + 0.5 * spread

            # append bid and ask prices to corresponding lists
            self.past_bids.append(bid_prices)
            self.past_asks.append(ask_prices)

            if len(self.past_bids) == self.look_back:
                self.past_bids.pop(0)

            if len(self.past_asks) == self.look_back:
                self.past_asks.pop(0)

            bid_rsi = self.calculate_rsi(self.past_bids)
            ask_rsi = self.calculate_rsi(self.past_asks)

            # append RSI values to corresponding lists
            bid_rsis.append(bid_rsi)
            ask_rsis.append(ask_rsi)

            # append other parameter values to corresponding lists
            profit.append(self.total_profit)
            cash.append(self.cash_balance)
            stock.append(self.current_stock)
            reward.append(self.previous_reward)

        # stack bid and ask RSI tensors
        bid_rsis_tensor = torch.stack(bid_rsis).to(device)
        ask_rsis_tensor = torch.stack(ask_rsis).to(device)

        # reshape the tensors
        bid_rsis_tensor = bid_rsis_tensor.unsqueeze(dim=1)
        ask_rsis_tensor = ask_rsis_tensor.unsqueeze(dim=1)

        # create the state for batch sizes above 1
        # state = torch.cat((data_batch_tensor, bid_rsis_tensor, ask_rsis_tensor), dim=1)

        # creating tensors that include information about current stock, funds available, profit of the episode and reward
        profit_tensor = torch.tensor(profit).to(device)
        funds_tensor = torch.tensor(cash).to(device)
        stock_tensor = torch.tensor(stock).to(device)
        reward_tensor = torch.tensor(reward).to(device)

        profit_tensor = profit_tensor.unsqueeze(dim=1)
        funds_tensor = funds_tensor.unsqueeze(dim=1)
        stock_tensor = stock_tensor.unsqueeze(dim=1)
        reward_tensor = reward_tensor.unsqueeze(dim=1)

        # create the state utilising information about current stock, funds available and profit of the episode
        state = torch.cat((current_bid_prices, current_ask_prices, bid_rsis_tensor, ask_rsis_tensor,
                          profit_tensor, funds_tensor, stock_tensor, reward_tensor, pred_rsi_tensor), dim=1)

        return state.float().to(device)

    # this function has been created solely to return an observation based on any data fed into it, without a dataloader. It was created for integration purposes
    def observe_without_training(self, past_data, pred_data):

        # update the new past data
        self.past_data = past_data

        # update the new predictions
        self.pred_data = pred_data

        # run the function that calculates the rsi of the predictions
        pred_rsi = self.calculate_rsi(
            self.pred_data, look_back=len(self.pred_data))

        # stack pred RSI tensor
        pred_rsi_tensor = torch.tensor(pred_rsi).to(device)

        # reshape the tensors
        pred_rsi_tensor = pred_rsi_tensor.unsqueeze(dim=0)
        pred_rsi_tensor = pred_rsi_tensor.unsqueeze(dim=1)

        # data_batch = torch.tensor(self.past_data[-1])
        data_batch = torch.tensor(self.past_data[-1]).unsqueeze(dim=0)

        data_batch_tensor = data_batch.to(device)
        bid_prices = torch.index_select(
            data_batch_tensor, 1, torch.tensor([0]).to(device))
        ask_prices = torch.index_select(
            data_batch_tensor, 1, torch.tensor([1]).to(device))

        bid_rsis = []
        ask_rsis = []
        profit = []
        cash = []
        stock = []
        reward = []

        for data_point in data_batch_tensor:
            bid_prices = data_point[0].cpu()
            ask_prices = data_point[1].cpu()

            # append bid and ask prices to corresponding lists
            self.past_bids.append(bid_prices)
            self.past_asks.append(ask_prices)

            if len(self.past_bids) == self.look_back:
                self.past_bids.pop(0)

            if len(self.past_asks) == self.look_back:
                self.past_asks.pop(0)

            bid_rsi = self.calculate_rsi(self.past_bids)
            ask_rsi = self.calculate_rsi(self.past_asks)

            # append RSI values to corresponding lists
            bid_rsis.append(bid_rsi)
            ask_rsis.append(ask_rsi)

            # append other parameter values to corresponding lists
            profit.append(self.total_profit)
            cash.append(self.cash_balance)
            stock.append(self.current_stock)
            reward.append(self.previous_reward)

        # stack bid and ask RSI tensors
        bid_rsis_tensor = torch.stack(bid_rsis).to(device)
        ask_rsis_tensor = torch.stack(ask_rsis).to(device)

        # reshape the tensors
        bid_rsis_tensor = bid_rsis_tensor.unsqueeze(dim=1)
        ask_rsis_tensor = ask_rsis_tensor.unsqueeze(dim=1)

        # create the state for batch sizes above 1
        # state = torch.cat((data_batch_tensor, bid_rsis_tensor, ask_rsis_tensor), dim=1)

        # creating tensors that include information about current stock, funds available, profit of the episode and reward
        profit_tensor = torch.tensor(profit).to(device)
        funds_tensor = torch.tensor(cash).to(device)
        stock_tensor = torch.tensor(stock).to(device)
        reward_tensor = torch.tensor(reward).to(device)

        profit_tensor = profit_tensor.unsqueeze(dim=1)
        funds_tensor = funds_tensor.unsqueeze(dim=1)
        stock_tensor = stock_tensor.unsqueeze(dim=1)
        reward_tensor = reward_tensor.unsqueeze(dim=1)

        # create the state utilising information about current stock, funds available and profit of the episode
        state = torch.cat((data_batch_tensor, bid_rsis_tensor, ask_rsis_tensor, profit_tensor,
                          funds_tensor, stock_tensor, reward_tensor, pred_rsi_tensor), dim=1)

        self.current_step = state.float().to(device)
        return self.current_step

    # Define the actor network


class ActorNet(nn.Module):
    def __init__(self, state):
        super(ActorNet, self).__init__()

        self.hidden_size = 256
        self.input_size = 8

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        output, (h_n, c_n) = self.lstm(x)
        x = F.relu(self.fc1(output[:, -1, :]))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class ValueCriticNet(nn.Module):
    def __init__(self, state):
        super(ValueCriticNet, self).__init__()

        self.num_layers = 2
        self.hidden_size = 256
        self.input_size = 8
        self.batch_size = state[0]

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        output, (h_n, c_n) = self.lstm(x)
        x = F.relu(self.fc1(output[:, -1, :]))
        x = self.fc2(x)
        return x.squeeze()


class ActionValueNet(nn.Module):
    def __init__(self, state):
        super(ActionValueNet, self).__init__()

        self.num_layers = 2
        self.hidden_size = 256
        self.input_size = 8
        self.batch_size = state[0]

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=32, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        output, (h_n, c_n) = self.lstm(x)
        x = F.relu(self.fc1(output[:, -1, :]))
        x = self.fc2(x)
        return x

# Modify the A2CAgent class to include the action-value critic network


class A2CAgent:
    def __init__(self, state, action_size, training):
        self.actor = ActorNet(state).to(device)
        self.value_critic = ValueCriticNet(state).to(device)
        self.action_critic = ActionValueNet(state).to(device)
        self.actor_optimizer = optim.RMSprop(
            self.actor.parameters(), lr=0.00001)
        self.value_critic_optimizer = optim.SGD(
            self.value_critic.parameters(), lr=0.00001)
        self.action_critic_optimizer = optim.SGD(
            self.action_critic.parameters(), lr=0.00001)
        self.set_training_status(training)

    # function responsible for changing if a model is still within the process of training or not
    def set_training_status(self, training):
        self.training = training

        # if the model is not training, deactivate dropout layers
        if (self.training == 0):
            self.actor.eval()
            self.value_critic.eval()
            self.action_critic.eval()
        else:
            self.actor.train()
            self.value_critic.train()
            self.action_critic.train()

    def get_action(self, state):
        if self.training == 1 and random.random() < EPSILON:
            # Select a random action
            action = torch.randint(0, 3, (state.size(0),), dtype=torch.long)
        else:
            action_probs = self.actor(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
        return action.to(device)

    def train(self, states, actions, rewards, next_states):
        # Calculate the predicted values for value critic, action critic, and actor

        value_values = self.value_critic(states).squeeze()
        next_value_values = self.value_critic(next_states).squeeze()

        action_values = self.action_critic(states)
        actor_values = self.actor(states).squeeze()

        # Calculate TD targets for value critic and action critic
        td_targets_value = rewards + GAMMA * next_value_values
        rewards = rewards.unsqueeze(1)
        td_targets_action = rewards + GAMMA * actor_values
        rewards = rewards.squeeze(1)

        # Calculate advantages for updating the actor
        advantages = td_targets_action - value_values.detach().unsqueeze(1)

        # Calculate value critic loss
        value_critic_loss = F.mse_loss(value_values, td_targets_value.detach())

        # Calculate action critic loss
        action_critic_loss = F.mse_loss(
            action_values, td_targets_action.detach())

        # Calculate actor loss
        old_probs = self.actor(states).detach()
        log_probs = torch.log(self.actor(states).gather(
            1, actions.unsqueeze(1).long()))
        ratio = torch.exp(log_probs - torch.log(old_probs))
        clipped_ratio = torch.clamp(
            ratio, min=1.0 - EPSILON, max=1.0 + EPSILON)
        surrogate_loss = - \
            torch.min(ratio * advantages.detach(),
                      clipped_ratio * advantages.detach()).mean()

        # Update value critic, action critic, and actor
        self.value_critic_optimizer.zero_grad()
        value_critic_loss.backward()
        self.value_critic_optimizer.step()

        self.action_critic_optimizer.zero_grad()
        action_critic_loss.backward()
        self.action_critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        surrogate_loss.backward()
        self.actor_optimizer.step()

        return surrogate_loss.item(), action_critic_loss.item(), value_critic_loss.item()

    # function that stores the weights of the agent after training
    def store_weights(self, results_path):

        # stores the actor network's weights
        torch.save({"state_dict": self.actor.state_dict()},
                   os.path.join(results_path, 'actor.pth'))

        # stores the Value Critic's network weights
        torch.save({"state_dict": self.value_critic.state_dict()},
                   os.path.join(results_path, 'value_critic.pth'))

        # stores the Actor Critic's Network weights
        torch.save({"state_dict": self.action_critic.state_dict()},
                   os.path.join(results_path, 'action_critic.pth'))

    def load_weights(self, results_path):

        # Load the saved model for the actor
        checkpoint = torch.load(os.path.join(
            results_path, 'actor.pth'), map_location=torch.device(device))

        # Extract the state dictionary from the checkpoint
        model_state_dict = checkpoint["state_dict"]

        # Load the state dictionary into the model
        self.actor.load_state_dict(model_state_dict)

        # Load the saved model for the actor
        checkpoint = torch.load(os.path.join(
            results_path, 'action_critic.pth'), map_location=torch.device(device))

        # Extract the state dictionary from the checkpoint
        model_state_dict = checkpoint["state_dict"]

        # Load the state dictionary into the model
        self.action_critic.load_state_dict(model_state_dict)

        # Load the saved model for the actor
        checkpoint = torch.load(os.path.join(
            results_path, 'value_critic.pth'), map_location=torch.device(device))

        # Extract the state dictionary from the checkpoint
        model_state_dict = checkpoint["state_dict"]

        # Load the state dictionary into the model
        self.value_critic.load_state_dict(model_state_dict)

# this class implements all the methods used for training and storing of the model


class DRLModel:
    # initialises the software
    def __init__(self, training=0):
        self.training = training

    # It's a function that loads the data in the format used by the Environment
    def load_data(self, filename, batch_size):
        data = []
        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Append values from row to data list
                data.append([float(row['midpoint']), float(row['spread'])])

        # Convert data list to PyTorch tensor
        X = torch.tensor(data, dtype=torch.float32)

        # Create a PyTorch dataset
        dataset = TensorDataset(X)

        # Create a PyTorch data loader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return data_loader

    # loads recorded predictions in a npy file
    def load_predictions(self, predictions_path):
        # load transformer predictions
        data_pred = np.load(predictions_path)
        return data_pred

    # Creates a custom environment for training
    def create_environment_training(self, data_path, predictions_path):
        # Create a custom environment with your own data
        datapath = data_path
        batch_size = 1
        data_loader = self.load_data(datapath, batch_size)
        env = CustomEnv(
            data_loader, training_data=self.load_predictions(predictions_path))

        # Create the A2C agent
        agent = A2CAgent(state=env.observation_space.shape,
                         action_size=env.action_space.n, training=self.training)

        return env, agent

    # Creates a custom environment for the purposes of integration
    def create_environment_inference(self):
        # Create a custom environment with no data
        env = CustomEnv()

        # Create the A2C agent
        agent = A2CAgent(state=env.observation_space.shape,
                         action_size=env.action_space.n, training=self.training)

        return env, agent

    # Performs training of the instantiated agent
    def train_model(self, env, agent):
        # Train the agent for 1000 episodes
        num_episodes = 340
        num_steps_per_episode = 200

        state = env.initial_state()

        for i in range(num_episodes):

            # Perform a partial reset of the environment
            env.partial_reset()

            # Collect experience
            states = torch.tensor([])
            actions = torch.tensor([])
            rewards = torch.tensor([])
            next_states = torch.tensor([])

            # send the tensors to the device
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)

            for t in range(num_steps_per_episode):
                # Get an action from the agent
                action = agent.get_action(state)

                # Take a step in the environment
                next_state, reward, profit, volume = env.step(action)

                # Record the experience
                states = torch.cat((states, state), dim=0)
                actions = torch.cat((actions, action), dim=0)
                rewards = torch.cat((rewards, reward), dim=0)
                next_states = torch.cat((next_states, next_state), dim=0)

                # Update the current state
                state = next_state

            # Train the agent on the collected experience
            actor_loss, action_critic_loss, value_critic_loss = agent.train(
                states, actions, rewards, next_states)

            # Print the episode results
            print(f"Episode {i+1}/{num_episodes} - Actor loss: {actor_loss:.4f}, Action Critic loss: {action_critic_loss:.4f}, Value Critic loss: {value_critic_loss:.4f}")
            print(
                f"Current Funds: {env.get_current_funds()}, Episode Profit: {env.get_current_profit()}")

    def initialize_training(self, data_path, predictions_path, results_path):

        # create the environment
        self.env, self.agent = self.create_environment_training(
            data_path, predictions_path)

        # train the model
        self.train_model(self.env, self.agent)

        # save results
        self.agent.store_weights(results_path)

    def initialize_testing(self, data_path, predictions_path, results_path):

        # create the environment
        self.env, self.agent = self.create_environment_training(
            data_path, predictions_path)

        # load the weights
        self.agent.load_weights(results_path)

        # disable training
        self.training = 0

        # update the training value within the agent
        self.agent.set_training_status(self.training)

        # test the model
        self.test_model(self.env, self.agent)

    def test_model(self, env, agent):
        # Train the agent for 1000 episodes
        num_episodes = 80
        num_steps_per_episode = 200

        state = env.initial_state()

        for i in range(num_episodes):

            # Collect experience
            states = torch.tensor([])
            actions = torch.tensor([])
            rewards = torch.tensor([])
            next_states = torch.tensor([])

            # send the tensors to the device
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)

            for t in range(num_steps_per_episode):
                # Get an action from the agent
                action = agent.get_action(state)

                # Take a step in the environment
                next_state, reward, profit, volume = env.step(action)

                # Record the experience
                states = torch.cat((states, state), dim=0)
                actions = torch.cat((actions, action), dim=0)
                rewards = torch.cat((rewards, reward), dim=0)
                next_states = torch.cat((next_states, next_state), dim=0)

                # Update the current state
                state = next_state

            # Print the episode results
            print(
                f"Episode: {i}, Current Funds After Testing: {env.get_current_funds()}, Episode Profit: {env.get_current_profit()}")

    # this method has been created for integration purposes, to create and load the model into memory
    def load_model(self, results_path):

        # create the environment
        self.env, self.agent = self.create_environment_inference()

        # load the weights
        self.agent.load_weights(results_path)

    # this method has been created for integration so that predictions can be retrieved with no training involved
    def get_prediction(self, past_data, pred_data, portfolio_dict):

        # update the portfolio data
        self.env.update_portfolio(portfolio_dict)

        # get the state based on the data
        state = self.env.observe_without_training(past_data, pred_data)

        # Get an action from the agent
        action = self.agent.get_action(state)

        # Take a step in the environment
        next_state, reward, profit, volume = self.env.step(action)

        return action, volume
