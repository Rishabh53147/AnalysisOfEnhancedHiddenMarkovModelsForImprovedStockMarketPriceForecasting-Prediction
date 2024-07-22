import numpy as np
import sys
import pandas as pd
from hmmlearn import hmm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def prepare_observations(stock_prices):
    price_diff = np.diff(stock_prices)
    price_direction = np.where(price_diff > 0, 1, 0).reshape(-1, 1)
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
    count = 0
    for i in range(0, len(actual_direction)):
        try:
            if actual_direction[i] == predicted_direction[i]:
                count = count + 1
        except:
            break
    accuracy = (count / (len(actual_direction) - 1)) * 100.0
    return accuracy, count

def calculate_errors(predicted_direction, actual_direction):
    mae = mean_absolute_error(actual_direction[1:], predicted_direction)
    mse = mean_squared_error(actual_direction[1:], predicted_direction)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

data = pd.read_csv("RELIANCE_NS.csv")
stock_prices = data["Adj Close"].values

price_direction = prepare_observations(stock_prices)

n_hidden_states = 2
# Perform cross-validation
n_splits = 5
fold_size = len(price_direction) // n_splits
accuracies = []
count_list = []

for i in range(n_splits):
    test_start = i * fold_size
    test_end = (i + 1) * fold_size

    # Split the data into training and test sets
    train_data = np.concatenate((price_direction[:test_start], price_direction[test_end:]))
    test_data = price_direction[test_start:test_end]

    # Train the HMM model
    model = train_hmm_model(train_data, n_hidden_states)

    # Make predictions on the test set
    predicted_direction = predict_direction(model, test_data)

    # Calculate accuracy
    predicted_direction = predicted_direction[1:]
    accuracy, count = calculate_accuracy(predicted_direction, test_data)
    accuracies.append(accuracy)
    count_list.append(count)

# Calculate errors
mae, mse, rmse = calculate_errors(predicted_direction, test_data)

# Display actual and predicted values
print(f"Mean Absolute Error: {mae:.19f}")
print(f"Mean Squared Error: {mse:.19f}")
print(f"Root Mean Squared Error: {rmse:.19f}")
print()

# Compute the average accuracy across all folds
mean_accuracy = np.round(np.mean(accuracies), 2)
mean_count = np.round(np.mean(count_list))
print("Mean Accuracy:", mean_accuracy)
print("Mean Days accurate:", mean_count)

# Plot the actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test_data[1:], label='Actual', marker='o')
plt.plot(predicted_direction, label='Predicted', marker='x')
plt.xlabel('Time Steps')
plt.ylabel('Direction (0/1)')
plt.legend()
plt.title(f"Actual vs Predicted")
plt.grid(True)
plt.show()


