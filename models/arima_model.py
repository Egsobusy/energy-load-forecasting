import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os

def train_and_evaluate_arima(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Drop all unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Apply one-hot encoding to non-numeric columns before splitting the data
    data = pd.get_dummies(data, drop_first=True)
    
    # Prepare time-series data
    target = 'PJME_MW'
    time_series = data[target]

    # Split data into train and test
    train_size = int(len(time_series) * 0.8)
    train, test = time_series[:train_size], time_series[train_size:]

    retrain = os.getenv('INTROBA_RETRAIN', 'false').lower() == 'true'

    if retrain:
        # Train the model
        arima_model = ARIMA(train, order=(5, 1, 0))  # Example order, adjust as needed
        arima_model_fit = arima_model.fit()

        # Save the model
        with open('models/saved_models/arima_model.pkl', 'wb') as f:
            pickle.dump(arima_model_fit, f)
    else:
        # Load the model
        with open('models/saved_models/arima_model.pkl', 'rb') as f:
            arima_model_fit = pickle.load(f)

    # Predict and evaluate ARIMA model
    arima_predictions = arima_model_fit.forecast(steps=len(test))
    arima_mse = mean_squared_error(test, arima_predictions)

    print(f"ARIMA - Mean Squared Error: {arima_mse}")
    print("ARIMA Model Summary:")
    print(arima_model_fit.summary())

    show_eval = os.getenv('INTROBA_SHOWEVAL', 'false').lower() == 'true'

    if show_eval:
        print("ARIMA - Random Evaluation Results:")
        for i in range(5):
            print(f"Expected: {test.iloc[i]}, Predicted: {arima_predictions.iloc[i]}")

# Example usage
if __name__ == "__main__":
    train_and_evaluate_arima('pjm_processed.csv')