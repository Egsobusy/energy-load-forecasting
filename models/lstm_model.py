import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import os

def train_and_evaluate_lstm(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Drop all unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Apply one-hot encoding to non-numeric columns before splitting the data
    data = pd.get_dummies(data, drop_first=True)

    # Prepare features and target
    target = 'PJME_MW'
    features = [col for col in data.columns if col not in [target]]
    X = data[features].values
    y = data[target].values

    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape data for LSTM (samples, timesteps, features)
    timesteps = 10
    X_lstm, y_lstm = [], []
    for i in range(len(X) - timesteps):
        X_lstm.append(X[i:i+timesteps])
        y_lstm.append(y[i+timesteps])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    # Build LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(timesteps, X.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())

    retrain = os.getenv('INTROBA_RETRAIN', 'false').lower() == 'true'

    if retrain:
        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
        model.save('models/saved_models/lstm_model.h5')
    else:
        # Load the model
        model = load_model('models/saved_models/lstm_model.h5')

    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test)

    mse = np.mean((y_test - y_pred)**2)
    print(f"LSTM - Mean Squared Error: {mse}")

    show_eval = os.getenv('INTROBA_SHOWEVAL', 'false').lower() == 'true'

    if show_eval:
        print("LSTM - Random Evaluation Results:")
        for i in range(5):
            print(f"Expected: {y_test[i][0]}, Predicted: {y_pred[i][0]}")

# Example usage
if __name__ == "__main__":
    train_and_evaluate_lstm('pjm_processed.csv')