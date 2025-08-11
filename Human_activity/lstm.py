from data_cleaning import setup_data
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

train, test = setup_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_to_loader(train, test):
    # Separate features and labels
    X_train = train.drop('Activity', axis=1).values
    y_train = train['Activity'].values
    X_test = test.drop('Activity', axis=1).values
    y_test = test['Activity'].values

    # Convert DataFrame to PyTorch tensors
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train-1, dtype=torch.long)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test-1, dtype=torch.long)

    # Create TensorDataset
    train_dataset = TensorDataset(train_tensor, train_labels)
    test_dataset = TensorDataset(test_tensor, test_labels)
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def init_model(train, test):
    """
    Initializes the LSTM model with the given parameters.
    
    Args:
        input_size (int): Number of features in the input data.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        output_size (int): Number of output classes.
        
    Returns:
        LSTMModel: An instance of the LSTM model.
    """

    input_size = test.shape[1] - 1  # Exclude the activity
    num_layers = 2
    hidden_size = 64
    output_size = len(train['Activity'].unique())  # Number of unique activities

    # Create LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

            out, _ = self.lstm(x, (h0, c0))
            out = (out[:, -1, :])  # Get the last time step
            out = self.linear(out)
            return out
    return LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)