import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the stock market dataset
data = pd.read_csv('INFY_NS.csv')  # Replace 'stock_data.csv' with your dataset file

# Extract the 'Close' prices
close_prices = data['Close'].values

# Step 2: Fit ARIMA model
order = (7, 0, 0)  # ARIMA order (p, d, q)
arima_model = ARIMA(close_prices, order=order)
arima_result = arima_model.fit()
residuals = arima_result.resid

# Step 3: Fit Hidden Markov Model
num_states = 2  # Number of hidden states in the HMM
hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=10)
hmm_model.fit(residuals.reshape(-1, 1))

# Make predictions using the ARIMA-HMM model
predicted_residuals = hmm_model.sample(len(close_prices))[0]
predicted_prices = close_prices + predicted_residuals.flatten()

# Determine the direction of actual and predicted price changes
actual_changes = np.sign(np.diff(close_prices))
predicted_changes = np.sign(np.diff(predicted_prices))  # Exclude the last prediction

# Calculate accuracy percentage
accuracy = np.mean(actual_changes == predicted_changes) * 100

# Calculate evaluation metrics
mae = mean_absolute_error(close_prices, predicted_prices)
mse = mean_squared_error(close_prices, predicted_prices)
rmse = np.sqrt(mse)

# Plot the original and predicted stock prices
plt.figure(figsize=(12, 3))
plt.plot(predicted_prices, label='Predicted Close Prices', color='red')
plt.plot(close_prices, label='Actual Close Prices', color='blue')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.title('INFOSYS NS')
plt.show()

# Print the evaluation metrics and accuracy percentage
print("Mean Absolute Error (MAE): {:.2f}".format(mae))
print("Mean Squared Error (MSE): {:.2f}".format(mse))
print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
print("Accuracy: {:.4f}%".format(accuracy))
