import pandas as pd
from models.random_forest_model import train_and_evaluate_random_forest
from models.arima_model import train_and_evaluate_arima
from models.mlp_model import train_and_evaluate_mlp

def simulate_future_energy_usage(data_path, model_type, future_steps):
    """
    Simulate future energy usage based on the selected model.

    Parameters:
        data_path (str): Path to the dataset.
        model_type (str): The type of model to use ('random_forest', 'arima', 'mlp').
        future_steps (int): Number of future steps to predict (only applicable for ARIMA).
    """
    # Load and preprocess data
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    # Generate features for future data
    target = 'PJME_MW'
    features = [col for col in data.columns if col not in ['Date', target]]

    # Example: Generate future feature data (this is a placeholder, adjust as needed)
    future_data = pd.DataFrame({
        'Feature1': [data[features[0]].mean()] * future_steps,
        'Feature2': [data[features[1]].mean()] * future_steps
    })

    if model_type == 'random_forest':
        print("Random Forest does not support future simulation. Training and evaluation only.")
        train_and_evaluate_random_forest(data_path)

    elif model_type == 'arima':
        time_series = data[target]

        train_size = int(len(time_series) * 0.8)
        train = time_series[:train_size]

        from statsmodels.tsa.arima.model import ARIMA
        arima_model = ARIMA(train, order=(5, 1, 0))  # Example order, adjust as needed
        arima_model_fit = arima_model.fit()

        future_predictions = arima_model_fit.forecast(steps=future_steps)
        print(f"Future Predictions ({future_steps} steps): {future_predictions}")

    elif model_type == 'mlp':
        print("MLP does not support future simulation. Training and evaluation only.")
        train_and_evaluate_mlp(data_path)

    else:
        print("Invalid model type. Choose from 'random_forest', 'arima', or 'mlp'.")

if __name__ == "__main__":
    simulate_future_energy_usage('pjm_processed.csv', model_type='arima', future_steps=10)