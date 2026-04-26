import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Load Iris
iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target

# Create one-hot labels
def to_one_hot(y, num_classes=3):
    y_one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    y_one_hot[np.arange(y.shape[0]), y] = 1.0
    return y_one_hot

y_oh = to_one_hot(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_oh, test_size=0.2, random_state=42)

# Convert to torch tensors
x_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
x_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test)

# Create dataloader
dataset = torch.utils.data.TensorDataset(x_train_t, y_train_t)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# 2. Define Network matching Agent Structure (4 -> 8 -> 3)
class AgentNet(nn.Module):
    def __init__(self):
        super(AgentNet, self).__init__()
        # Hidden layer: 8 neurons
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        # Output layer: 3 neurons
        self.fc2 = nn.Linear(8, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = AgentNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Train
epochs = 200
print("Starting training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluate
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            out_test = model(x_test_t)
            preds = torch.argmax(out_test, dim=1)
            labels = torch.argmax(y_test_t, dim=1)
            acc = (preds == labels).float().mean().item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f} - Test Acc: {acc*100:.2f}%")

# 4. Export to weights.json
print("Exporting weights to weights.json...")
weights_dict = {}

# Hidden layer (H0-H7)
fc1_w = model.fc1.weight.detach().numpy()
fc1_b = model.fc1.bias.detach().numpy()
for i in range(8):
    weights_dict[f"H{i}"] = {
        "weights": fc1_w[i].tolist(),
        "bias": float(fc1_b[i])
    }

# Output layer (O0-O2)
fc2_w = model.fc2.weight.detach().numpy()
fc2_b = model.fc2.bias.detach().numpy()
for i in range(3):
    weights_dict[f"O{i}"] = {
        "weights": fc2_w[i].tolist(),
        "bias": float(fc2_b[i])
    }

with open("weights.json", "w", encoding="utf-8") as f:
    json.dump(weights_dict, f, ensure_ascii=False, indent=2)
print("Done! Exported 11 agents' weights.")
