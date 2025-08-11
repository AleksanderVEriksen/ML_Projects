import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import read_file

RESULTS_DIR = 'Human_activity/results'
# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def loss_and_accuracy(name:str):
    if name.endswith('.pth'):
        name = name[:-4]
    # Load the average losses and accuracy from the text files
    losses_path = os.path.join(RESULTS_DIR, f'{name}/avg_losses_{name}.txt')
    accuracy_path = os.path.join(RESULTS_DIR, f'{name}/avg_accuracy_{name}.txt')
    if not os.path.exists(losses_path) or not os.path.exists(accuracy_path):
        print("Loss and accuracy files not found. Please ensure they are generated during training.")
        return
    avg_losses = read_file(losses_path)
    avg_accuracy = read_file(accuracy_path)
    
    # Plot the loss curve and accuracy curve
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(avg_losses) + 1), avg_losses, color='blue', label='Loss')
    plt.plot(range(1, len(avg_accuracy) + 1), avg_accuracy, color='orange', label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training Loss & Accuracy Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'{name}/loss_accuracy_curve_{name}.png'))
    plt.show()

def plot_feature_importance(model, name:str):
    """
    Plots the feature importance based on the learned weights of the model.
    
    Args:
        model (nn.Module): The trained LSTM model.
    """

    if name.endswith('.pth'):
        name = name[:-4]

    # Read feature names
    features_path = os.path.join(os.path.dirname(__file__), "UCI HAR Dataset", "features.txt")
    features = pd.read_csv(features_path, sep='\s+', header=None)
    feature_names = features[1].tolist()  # 561 features

    # Get input-to-hidden weights from the first LSTM layer
    # PyTorch LSTM: weight_ih_l0 shape is (4*hidden_size, input_size)
    weight_ih_l0 = model.lstm.weight_ih_l0.detach().cpu().numpy()  # shape: (4*hidden_size, input_size)
    # Aggregate across gates and hidden units (mean of absolute values for each input feature)
    FI = abs(weight_ih_l0).mean(axis=0)  # shape: (input_size,)

    if len(feature_names) != len(FI):
        raise ValueError(f"Feature names ({len(feature_names)}) and weights ({len(FI)}) length mismatch.")

    df = pd.DataFrame({'Feature': feature_names, 'learned_weights': FI})
    df = df.reindex(df.learned_weights.abs().sort_values(ascending=False).index)
    top5 = df.head(5)
    plt.figure(figsize=(8, 4))
    plt.bar(top5.Feature, top5.learned_weights, color='skyblue')
    plt.xlabel('Top 5 Features')
    plt.ylabel('Importance')
    plt.title('Top 5 Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{name}/top5feature_importance_{name}.png"))
    plt.show()
