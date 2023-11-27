```python
import torch

# Define input data
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# Define the model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 10)
        self.fc2 = torch.nn.Linear(10, 5)
        self.fc3 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the loss function
loss_fn = torch.nn.MSELoss()

# Initialize the model and optimizer
model = LinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    # Forward pass
    y_pred = model(x)

    # Compute loss
    loss = loss_fn(y_pred, y)

    # Zero gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()

# Test the model
x_test = torch.tensor([[5]], dtype=torch.float32)
y_pred = model(x_test)
print(y_pred.item())  # Expected output: 10.0
```

### Notes:
- You don't apply a [[Cross Entropy Loss|softmax]] to the outputs because this will restrict the outputs to be between `[0, 1]`.