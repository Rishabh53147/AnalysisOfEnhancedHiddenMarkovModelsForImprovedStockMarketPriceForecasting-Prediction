import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

def prepare_observations(stock_prices):
    price_direction = []
    for i in range(len(stock_prices) - 1):
        diff = stock_prices[i + 1] - stock_prices[i]
        if diff > 0:
            price_direction.append(1)
        else:
            price_direction.append(0)
    return price_direction

# Read in the CSV dataset
data = pd.read_csv('HDB.csv')
stock_prices = data['Adj Close'].values
price_direction = prepare_observations(stock_prices)


# Define the number of hidden states and other hyperparameters to tune
n_hidden_states = [2, 3, 4]
n_splits = 5  # Number of splits for time series cross-validation

# Initialize lists to store results
best_accuracy = 0
best_params = {'n_components': None}

# Perform time series cross-validation
tscv = TimeSeriesSplit(n_splits=n_splits)

for n_states in n_hidden_states:
    avg_accuracy = 0

    for train_index, test_index in tscv.split(price_direction):
        train_data = np.array(price_direction)[train_index].reshape(-1, 1)
        test_data = np.array(price_direction)[test_index].reshape(-1, 1)

        # Create and fit the HMM model
        model = hmm.CategoricalHMM(n_components=n_states)
        model.fit(train_data)

        # Predict using the model
        predicted_direction = model.predict(test_data)

        # Calculate accuracy for this fold
        accuracy = accuracy_score(predicted_direction, test_data)
        avg_accuracy += accuracy

    # Calculate the average accuracy across folds
    avg_accuracy /= n_splits

    # If this set of hyperparameters resulted in a better accuracy, update the best parameters
    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_params['n_components'] = n_states

# Train the best model with the selected hyperparameters
best_model = hmm.CategoricalHMM(n_components=best_params['n_components'])
best_model.fit(np.array(price_direction).reshape(-1, 1))

# Calculate MAE, MSE, and RMSE for the best model
predicted_direction = best_model.predict(np.array(price_direction).reshape(-1, 1))
mae = mean_absolute_error(price_direction, predicted_direction)
mse = mean_squared_error(price_direction, predicted_direction)
rmse = np.sqrt(mse)

print(f'Best Hidden States: {best_params["n_components"]}')
print(f'Best Cross-Validated Accuracy: {best_accuracy:.4f}')
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
