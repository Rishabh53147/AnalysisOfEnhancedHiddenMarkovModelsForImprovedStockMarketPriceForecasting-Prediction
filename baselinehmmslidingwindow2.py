import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, mean_squared_error

def prepare_observations(stock_prices):
    price_direction = []
    for i in range(len(stock_prices) - 1):
        diff = stock_prices[i + 1] - stock_prices[i]
        if diff > 0:
            price_direction.append(1)
        else:
            price_direction.append(0)
    return price_direction

def train_hmm_model(train_data, n_hidden_states):
    model = hmm.CategoricalHMM(n_components=n_hidden_states)
    model.fit(train_data)
    return model

def predict_direction(model, test_data):
    next_direction = model.predict(test_data)
    return next_direction

def calculate_accuracy(predicted_direction, test_data):
    actual_direction = test_data
    count = 0
    for i in range(len(actual_direction)):
        if actual_direction[i] == predicted_direction[i]:
            count = count + 1
    accuracy = (count / len(actual_direction)) * 100.0
    return accuracy, count

# Read in the CSV dataset
data = pd.read_csv('INFY_NS2.csv')
stock_prices = data['Adj Close'].values
price_direction = prepare_observations(stock_prices)

# Define the number of hidden states and the length of the sliding window
n_hidden_states = 2
window_length = 10

# Initialize lists to store the predictions and actual values
prediction_list = []
actual_list = []

# Initialize lists to store prediction errors
mae_list = []
mse_list = []
rmse_list = []

# Fit the model using a sliding window approach on the training data
for i in range(window_length, len(price_direction)):
    # Split the data into training and test sets
    train_data = price_direction[i-window_length:i]
    test_data = [price_direction[i]]  # Convert test_data to a list of one element

    # Train the HMM model
    train_data = np.array(train_data).reshape(-1, 1)
    model = train_hmm_model(train_data, n_hidden_states)

    # Make predictions on the test set
    predicted_direction = predict_direction(model, np.array(test_data).reshape(-1, 1))
    prediction_list.append(predicted_direction[0])
    actual_list.append(test_data[0])

# Calculate accuracy
accuracy, count = calculate_accuracy(prediction_list, actual_list)

# Calculate MAE, MSE, and RMSE
mae = mean_absolute_error(actual_list, prediction_list)
mse = mean_squared_error(actual_list, prediction_list)
rmse = np.sqrt(mse)

print(f'Accuracy: {accuracy:.2f}%')
print(f'Days accurate: {count}')
print(f'Testing data: {len(actual_list)}')
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
