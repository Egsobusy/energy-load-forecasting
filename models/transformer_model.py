import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.input_layer(src)
        transformer_output = self.transformer(src, src)
        # Use the last time step for each sequence in the batch
        output = self.output_layer(transformer_output[:, -1, :])
        return output

def train_and_evaluate_transformer(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Drop all unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Apply one-hot encoding to non-numeric columns before splitting the data
    data = pd.get_dummies(data, drop_first=True)

    # Prepare features and target
    target = 'PJME_MW'
    features = [col for col in data.columns if col not in ['Date', target]]
    X = data[features].values
    y = data[target].values

    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape data for Transformer (samples, timesteps, features)
    timesteps = 10
    X_transformer, y_transformer = [], []
    for i in range(len(X) - timesteps):
        X_transformer.append(X[i:i+timesteps])
        y_transformer.append(y[i+timesteps])
    X_transformer, y_transformer = np.array(X_transformer), np.array(y_transformer)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_transformer, dtype=torch.float32)
    y_tensor = torch.tensor(y_transformer, dtype=torch.float32)

    # Split data
    train_size = int(0.8 * len(X_tensor))
    train_dataset = TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_dataset = TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    input_dim = X_tensor.shape[2]
    model = TransformerModel(input_dim=input_dim, d_model=64, nhead=4, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    retrain = os.getenv('INTROBA_RETRAIN', 'false').lower() == 'true'

    if retrain:
        # Train model
        for epoch in range(20):
            model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                # Reshape the target tensor to match the model's output
                y_batch = y_batch.view(-1, 1)
                # Adjust the loss function to use the entire output directly
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}")
        torch.save(model.state_dict(), 'models/saved_models/transformer_model.pth')
    else:
        # Load the model
        model.load_state_dict(torch.load('models/saved_models/transformer_model.pth'))

    # Evaluate model
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions.append(output.numpy())
            actuals.append(y_batch.numpy())

    predictions = scaler_y.inverse_transform(np.vstack(predictions))
    actuals = scaler_y.inverse_transform(np.vstack(actuals))
    mse = np.mean((actuals - predictions)**2)
    print(f"Transformer - Mean Squared Error: {mse}")

    show_eval = os.getenv('INTROBA_SHOWEVAL', 'false').lower() == 'true'

    if show_eval:
        print("Transformer - Random Evaluation Results:")
        for i in range(5):
            print(f"Expected: {actuals[i].item()}, Predicted: {predictions[i].item()}")

# Example usage
if __name__ == "__main__":
    train_and_evaluate_transformer('pjm_processed.csv')