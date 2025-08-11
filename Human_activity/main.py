from data_cleaning import setup_data
from lstm import init_model, data_to_loader
from train import training
from utils import save_model, load_model
from results import loss_and_accuracy, plot_feature_importance
import sys
from test import evaluation
import os

def print_usage():
    print("Usage: python main.py [EPOCHS] [Mode] [Name]")
    print("  EPOCHS (optional): Number of training epochs (default: 1000)")
    print("  Mode (optional): If set, loads the model and performs the specified operation:")
    print("    - 'train': Train the model")
    print("    - 'test': Evaluate the model on the test set")
    print("    - 'plot': Plot the training results")
    print("  Name (optional): Name of the model file to use (default: 'base_model.pth')")

def print_models():
    print("Available models:")
    models_dir = 'models'
    if os.path.exists(models_dir):
        models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if models:
            print("\n".join(models))
        else:
            print("No models found.")
    else:
        print("Models directory does not exist.")

def main(EPOCHS: int = 1000, Mode: str = None, Name:str = 'base_model.pth'):
    Name = Name.strip()  # Add this line at the start of your main function
    if not Name.endswith('.pth'):
        Name += '.pth'
    # Step 1: Data Preparation
    print("Setting up and cleaning data...")
    train_df, test_df = setup_data()
    print("Data ready.")

    # Step 2: Model Initialization
    train_loader, test_loader = data_to_loader(train_df, test_df)
    model = init_model(train_df, test_df)

    # Checks if the userdefined model exist
    if Name != 'base_model.pth':
        if os.path.exists('models/' + Name):
            model = load_model(model, name=Name)
    
    # If Mode is not specified, do all steps
    if Mode is None and Name == 'base_model.pth' or Mode is None and Name != 'base_model.pth':
        # Step 3: Training & Evaluation
        print("Starting training...")
        model = training(model, train_loader, test_loader, EPOCHS)
        # Step 4: Save Model
        print("Saving model...")
        save_model(model, name=Name)
        print("Model saved.")

        # Step 5: Plotting
        print("Plotting results...")
        loss_and_accuracy()
        print("Plotting feature importance...")
        plot_feature_importance(model) 
        print("All steps complete.")
    # If Mode is specified, check if the model exists
    if Mode is not None:
        if not os.path.exists('models/' + Name):
            print(f"Model {Name} does not exist. Please train the model first before adjusting the mode.")
            return
        # If Mode is train, only train the model regardless of the model name
        # Give user option to overwrite model name
        if Mode == 'train':
            print("Training the model...")
            model = training(model, train_loader, test_loader, EPOCHS)
            print("Model trained.")
            name = input("Overwrite model name : y/n ")
            if name.lower() == 'y':
                name = input("Enter new model name (default: base_model.pth): ")
                if not name.endswith('.pth'):
                    name += '.pth'
                save_model(model, name=name)
            elif name.lower() == 'n':
                print("Model saved as current name.")
                name = Name
            else:
                print("Invalid input. Model not saved.")
            save_model(model, name=Name)

        # If Mode is test, only evaluate the model
        elif Mode == 'test':
            print("Evaluating the model...")
            accuracy = evaluation(test_loader, model)
            print(f"Test Accuracy: {accuracy:.2f}%")

        # If Mode is plot, only plot the results
        elif Mode == 'plot':
            print("Plotting results...")
            loss_and_accuracy()
            print("Plotting feature importance...")
            plot_feature_importance(model)
if __name__ == "__main__":
    print("Welcome to the Human Activity Recognition Project!")
    print("This project uses LSTM to classify human activities based on sensor data.")
    print("Press -h or --help for usage instructions.")
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help", "help"):
        print_usage()
        print_models()
        sys.exit(0)
    mode_list = ['train', 'test', 'plot']
    # Default values
    epochs = 1000
    Mode = None
    model = 'base_model'
    # Parse arguments if provided
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] in mode_list:
        Mode = sys.argv[2]
    if len(sys.argv) > 3:
        if not sys.argv[3].endswith('.pth'):
            model = sys.argv[3] + '.pth'
        else:
            model = sys.argv[3]
    main(epochs, Mode, model)