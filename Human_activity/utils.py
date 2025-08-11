import torch
import os
def save_to_file(filename, data):
    """
    Saves the given data to a text file.
    
    Args:
        filename (str): The name of the file to save the data.
        data (list): The data to be saved.
    """
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

def read_file(filename):
    """
    Reads model values from a text file and returns them as a list.
    
    Args:
        filename (str): The path to the text file containing model values.
        
    Returns:
        list: A list of model values read from the file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")
    
    with open(filename, 'r') as f:
        values = [float(line.strip()) for line in f.readlines()]
    
    return values


def save_model(model, path='Human_activity/models/', name='base_model.pth'):
    """
    Saves the trained model to the specified path.
    """
    # Ensure path ends with a separator
    if not path.endswith(os.sep):
        path += os.sep
    full_path = os.path.join(path, name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    torch.save(model.state_dict(), full_path)

def load_model(model, path='Human_activity/models/', name='base_model.pth'):
    """
    Loads the model state from the specified path.
    
    Args:
        model (torch.nn.Module): The model to load the state into.
        path (str): The directory path from which to load the model state.
        name (str): The model filename.
        
    Returns:
        torch.nn.Module: The model with loaded state.
    """
    # Ensure path ends with a separator
    if not path.endswith(os.sep):
        path += os.sep
    full_path = os.path.join(path, name)
    if os.path.exists(full_path):
        model.load_state_dict(torch.load(full_path, map_location=torch.device('cpu')))
        return model
    else:
        raise FileNotFoundError(f"Model file {full_path} not found.")