import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from sklearn.model_selection import KFold

def prepare_observations(stock_prices):
    price_direction = []
    for i in range(0, len(stock_prices) - 1):
        diff = stock_prices[i + 1] - stock_prices[i]
        if diff > 0:
            price_direction.append(1)
        else:
            price_direction.append(0)
    return price_direction

def train_hmm_model(train_data, n_hidden_states):
    model = hmm.GaussianHMM(n_components=n_hidden_states)
    model.fit(train_data)
    return model

def predict_direction(model, test_data):
    next_direction = model.predict(test_data)
    return next_direction

def calculate_accuracy(predicted_direction, test_data):
    actual_direction = test_data
    count = sum(pred == actual for pred, actual in zip(predicted_direction, actual_direction))
    accuracy = (count / len(actual_direction)) * 100.0
    return accuracy, count

def calculate_mse_mae_rmse(true_data, predicted_data):
    mse = mean_squared_error(true_data, predicted_data)
    mae = mean_absolute_error(true_data, predicted_data)
    rmse = math.sqrt(mse)
    return mse, mae, rmse

# Read in the CSV dataset
data = pd.read_csv("INFY_NS2.csv")
stock_prices = data['Adj Close'].values
price_direction = prepare_observations(stock_prices)

# Define the number of hidden states and the number of folds
n_hidden_states = 2
n_splits = 5  # Number of folds for k-fold cross-validation

kf = KFold(n_splits=n_splits)

accuracy_list = []
mse_list = []
mae_list = []
rmse_list = []

for train_index, test_index in kf.split(price_direction):
    train_data = [price_direction[i] for i in train_index]
    test_data = [price_direction[i] for i in test_index]

    # Train the HMM model
    train_data = np.array(train_data).reshape(-1, 1)
    model = train_hmm_model(train_data, n_hidden_states)

    # Make predictions on the test set
    test_data = np.array(test_data).reshape(-1, 1)
    predicted_direction = predict_direction(model, test_data)
    
    accuracy, count = calculate_accuracy(predicted_direction, test_data)
    accuracy_list.append(accuracy)

    mse, mae, rmse = calculate_mse_mae_rmse(test_data, predicted_direction)
    mse_list.append(mse)
    mae_list.append(mae)
    rmse_list.append(rmse)

print(f'Average Accuracy: {np.mean(accuracy_list):.2f}%')
print(f'Average MSE: {np.mean(mse_list):.2f}')
print(f'Average MAE: {np.mean(mae_list):.2f}')
print(f'Average RMSE: {np.mean(rmse_list):.2f}')
