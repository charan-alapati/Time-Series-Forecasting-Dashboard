import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm_model(df, seq_length=10, epochs=20, lr=0.001):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(df.values.reshape(-1, 1))

    X, y = create_sequences(data, seq_length)
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    model = LSTMForecaster()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

    return model, scaler

def forecast_future(model, scaler, data, seq_length=10, steps=10):
    model.eval()
    predictions = []
    input_seq = data[-seq_length:].reshape(-1, 1)
    for _ in range(steps):
        seq = torch.tensor(scaler.transform(input_seq).reshape(1, seq_length, 1), dtype=torch.float32)
        with torch.no_grad():
            pred = model(seq).item()
        pred_unscaled = scaler.inverse_transform([[pred]])[0][0]
        predictions.append(pred_unscaled)
        input_seq = np.append(input_seq[1:], pred_unscaled)
    return predictions
