import pandas as pd
from sklearn.model_selection import train_test_split
from models.random_forest_model import train_and_evaluate_random_forest
from models.arima_model import train_and_evaluate_arima
from models.mlp_model import train_and_evaluate_mlp
from models.lstm_model import train_and_evaluate_lstm
from models.transformer_model import train_and_evaluate_transformer
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Load data
data = pd.read_csv('pjm_processed.csv')

# Prepare features and target
target = 'PJME_MW'
features = [col for col in data.columns if col != target]
X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate Random Forest model
train_and_evaluate_random_forest('pjm_processed.csv')

# Train and evaluate ARIMA model
train_and_evaluate_arima('pjm_processed.csv')

# Run MLP model
train_and_evaluate_mlp('pjm_processed.csv')

# Run LSTM model
train_and_evaluate_lstm('pjm_processed.csv')

# Run Transformer model
train_and_evaluate_transformer('pjm_processed.csv')
