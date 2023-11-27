```python
import torch
import torch.nn.functional as F

# Define input data
x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[0], [0], [1], [1]], dtype=torch.float32)

# Define the logistic regression model
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)

# Define the loss function
loss_fn = torch.nn.BCELoss()

# Initialize the model and optimizer
model = LogisticRegressionModel()
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
print(y_pred.item())  # Expected output: 0.948

```