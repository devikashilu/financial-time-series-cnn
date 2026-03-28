import torch
import torch.nn as nn
import torch.optim as optim

class SpectrogramCNN(nn.Module):
    def __init__(self, in_channels: int = 5):
        super(SpectrogramCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # We will calculate the flattened dimension dynamically in the forward pass 
        # or just use AdaptiveAvgPool2d to enforce a fixed output size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        out = self.fc2(x)
        return out

def train_model(X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 15, lr: float = 0.001):
    """
    Trains the CNN using MSE loss.
    Returns: Trained model, list of epoch losses
    """
    model = SpectrogramCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    # We train in a single batch or mini-batches for simplicity
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        losses.append(epoch_loss / len(loader))
        
    return model, losses

def predict_model(model: nn.Module, X_test: torch.Tensor):
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
    return preds.squeeze().numpy()
