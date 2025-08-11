import torch
from utils import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluation(test_loader, model, old_model=False):
    """
    Evaluates the model on the test data.

    Args:
        test_loader (DataLoader): DataLoader for the test data.
        model (nn.Module): The LSTM model to be evaluated.
        new_model (bool): If True, loads the model from disk before evaluation.

    Returns:
        float: The accuracy of the model on the test set.
    """
    if old_model:
        load_model(model)
    else:
        correct = 0
        total = 0

        with torch.no_grad():
            model.eval()
            for test_tensor, test_labels in (test_loader):
                test_tensor = test_tensor.to(device)
                test_labels = test_labels.to(device)
                outputs = model(test_tensor.unsqueeze(1))  # Add sequence dimension
                _, predicted = torch.max(outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy