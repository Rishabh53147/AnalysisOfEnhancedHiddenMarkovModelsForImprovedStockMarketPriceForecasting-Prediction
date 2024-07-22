    import numpy as np
    import pandas as pd
    from hmmlearn import hmm
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error

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

    # Read in the CSV dataset
    data = pd.read_csv("INFY_NS2.csv")
    stock_prices = data['Adj Close'].values
    price_direction = prepare_observations(stock_prices)

    # Define the number of hidden states
    n_hidden_states = 2

    # Split the data into training and test sets
    train_data = np.array(price_direction[:-1]).reshape(-1, 1)
    test_data = np.array(price_direction[1:]).reshape(-1, 1)

    # Train the HMM model
    model = train_hmm_model(train_data, n_hidden_states)

    # Make predictions on the test set
    predicted_direction = predict_direction(model, test_data)

    # Calculate accuracy
    accuracy, count = calculate_accuracy(predicted_direction, test_data)
    accuracy_value = accuracy[0]  # Access the accuracy value from the array

    # Calculate MSE, MAE, and RMSE
    mse = mean_squared_error(test_data, predicted_direction)
    mae = mean_absolute_error(test_data, predicted_direction)
    rmse = np.sqrt(mse)

    print(f'Accuracy: {accuracy_value:.2f}%')
    print(f'Days accurate: {count}')
    print(f'Testing data: {len(test_data)}')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
