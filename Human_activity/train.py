import torch
from utils import save_to_file
from tqdm import tqdm
from torcheval.metrics import MulticlassAccuracy
from test import evaluation
import os
# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = 'Human_activity/results/'

def training(model, train_loader, test_loader, EPOCHS, name:str):
    """
    Trains the LSTM model on the training data.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        model (nn.Module): The LSTM model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.

    Returns:
        tuple: A tuple containing the trained model, average losses, and average accuracy per epoch.
    """
    if name.endswith('.pth'):
        name = name[:-4]
    # Get the labels from the training loader
    labels = train_loader.dataset.tensors[1]
    classes = len(torch.unique(labels))
    # Initialize the accuracy metric
    metric = MulticlassAccuracy(num_classes=classes).to(device)
    avg_losses_per_epoch = []
    avg_train_accuracy_per_epoch = []
    avg_test_accuracy_per_epoch = []

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train loop
    model.to(device)
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        model.train()
        cumulative_loss = 0.0
        metric.reset()
        for i, (train_tensor, train_labels) in enumerate(train_loader):
            train_tensor = train_tensor.to(device)
            train_labels = train_labels.to(device)
            # Forward pass
            optimizer.zero_grad()
            outputs = model(train_tensor.unsqueeze(1))  # Add sequence dimension
            loss = criterion(outputs, train_labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.item()
            # Calculate accuracy
            metric.update(outputs, train_labels)
        avg_loss = cumulative_loss / len(train_loader)
        avg_acc = metric.compute().item()
        avg_losses_per_epoch.append(avg_loss)
        avg_train_accuracy_per_epoch.append(avg_acc)
        
        # Evaluate on test set at the end of each epoch
        test_accuracy = evaluation(test_loader, model)
        avg_test_accuracy_per_epoch.append(test_accuracy)
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f'Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}, Test Accuracy: {test_accuracy:.4f}')


    print("\n-------Training complete-------")
    print(f'\n\nEpoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
    print(f'\nFinal Loss: {avg_losses_per_epoch[-1]:.4f}, Final Accuracy: {avg_train_accuracy_per_epoch[-1]:.4f}\n')
    # Create directory
    os.makedirs(os.path.join(RESULTS_DIR, name), exist_ok=True)
    # Save loss and accuracy values for plotting to text file
    save_to_file(os.path.join(RESULTS_DIR,f'{name}/avg_losses_{name}.txt'), avg_losses_per_epoch)
    save_to_file(os.path.join(RESULTS_DIR,f'{name}/avg_accuracy_{name}.txt'), avg_train_accuracy_per_epoch)
    save_to_file(os.path.join(RESULTS_DIR,f'{name}/avg_test_accuracy_{name}.txt'), avg_test_accuracy_per_epoch)
    return model