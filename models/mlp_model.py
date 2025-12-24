import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_and_evaluate_mlp(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Drop all unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Apply one-hot encoding to non-numeric columns before splitting the data
    data = pd.get_dummies(data, drop_first=True)

    # Prepare features and target
    target = 'PJME_MW'
    features = [col for col in data.columns if col != target]
    X = data[features]
    y = data[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    retrain = os.getenv('INTROBA_RETRAIN', 'false').lower() == 'true'

    if retrain:
        # Train MLP model
        mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp_model.fit(X_train, y_train)

        # Predict and evaluate MLP model
        y_pred = mlp_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MLP - Mean Squared Error: {mse}")
        print(f"MLP - R^2 Score: {r2}")

        # Save the model
        joblib.dump(mlp_model, 'models/saved_models/mlp_model.pkl')
    else:
        # Load the model
        mlp_model = joblib.load('models/saved_models/mlp_model.pkl')

    show_eval = os.getenv('INTROBA_SHOWEVAL', 'false').lower() == 'true'

    if show_eval:
        # Predict using the loaded model
        y_pred = mlp_model.predict(X_test)

        print("MLP - Random Evaluation Results:")
        for i in range(5):
            print(f"Expected: {y_test.iloc[i]}, Predicted: {y_pred[i]}")

# Example usage
if __name__ == "__main__":
    train_and_evaluate_mlp('pjm_processed.csv')