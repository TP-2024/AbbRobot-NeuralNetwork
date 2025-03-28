import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from dqn.controllers.utils.nn_model import IKNet


def train_model(data_path, model_path):
    data = np.load(data_path)
    inputs = data['inputs']
    outputs = data['outputs']

    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    model = IKNet(input_dim=3, output_dim=y_train.shape[1])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model training complete. Model saved to {model_path}.")
