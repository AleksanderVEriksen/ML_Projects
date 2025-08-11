from data_cleaning import setup_data
from lstm import init_model, data_to_loader
from train import training
from utils import save_model, load_model
from results import loss_and_accuracy, plot_feature_importance
from simulate_data import run_simulation
import sys
from test import evaluation
import os

def print_usage():
    print("Usage: python main.py [EPOCHS] [Mode] [Model_name]")
    print("  EPOCHS (optional): Number of training epochs (default: 1000)")
    print("  Mode (optional): If set, loads the model and performs the specified operation:")
    print("    - 'simulate': simulate activity data")
    print("    - 'train': Train the model")
    print("    - 'test': Evaluate the model on the test set")
    print("    - 'plot': Plot the training results")
    print("  Model_name (optional): Model_name of the model file to use (default: 'base_model.pth')")

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

def main(EPOCHS: int = 10, Mode: str = None, Model_name:str = 'base_model.pth'):
    Model_name = Model_name.strip()  # Add this line at the start of your main function
    if not Model_name.endswith('.pth'):
        Model_name += '.pth'
    # Step 1: Data Preparation
    train_df, test_df = setup_data()

    # Step 2: Model Initialization
    train_loader, test_loader = data_to_loader(train_df, test_df)
    model = init_model(train_df, test_df)

    # Checks if the user-defined model exist
    if Model_name != 'base_model.pth':
        if os.path.exists('models/' + Model_name):
            model = load_model(model, name=Model_name)
    print(EPOCHS, Mode, Model_name)
    # If Mode is not specified, do all steps
    if Mode is None and Model_name == 'base_model.pth' or Mode is None and Model_name != 'base_model.pth':
        
        # Step 3: Simulation
        run_simulation()
        # Step 4: Training & Evaluation
        print("\n---Starting training---\n")
        model = training(model, train_loader, test_loader, EPOCHS, name=Model_name)
        # Step 5: Save Model
        print("Saving model...")
        save_model(model, name=Model_name)
        print("Model saved.")
        # Step 6: Plotting
        print("Plotting results...")
        loss_and_accuracy(Model_name)
        print("Plotting feature importance...")
        plot_feature_importance(model, Model_name) 
        print("All steps complete.")
    # If Mode is specified, check if the model exists
    if Mode is not None:
        if not os.path.exists('Human_activity/models/' + Model_name):
            print(f"Model {Model_name} does not exist. Please check the name for spelling errors, or start the program without changing the model_name.")
            print("Usage: python main.py [EPOCHS] [Mode] [Model_name]")
            return
        # If Mode is train, only train the model regardless of the model Model_name
        # Give user option to overwrite model Model_name
        if Mode == 'train':
            answer = input(f"Overwrite model model_name: {Model_name}? (y/n) - ")
            if answer.lower() == 'y':
                Model_name = input("Enter new model Model_name (default: base_model.pth): ")
                if not Model_name.endswith('.pth'):
                    Model_name += '.pth'
                print(f"Training model and saving as {Model_name}")
                model = training(model, train_loader, test_loader, EPOCHS, name=Model_name)
            elif answer.lower() == 'n':
                print("Model saved as current Model_name.")
                Model_name = Model_name
                model = training(model, train_loader, test_loader, EPOCHS, name=Model_name)
            else:
                print("Invalid input. (y/n)")
                sys.exit(1)
            save_model(model, name=Model_name)

        # If Mode is test, only evaluate the model
        elif Mode == 'test':
            print("Evaluating the model...")
            accuracy = evaluation(test_loader, model)
            print(f"Test Accuracy: {accuracy:.2f}%")

        # If Mode is plot, only plot the results
        elif Mode == 'plot':
            print("Plotting results...")
            loss_and_accuracy(Model_name)
            print("Plotting feature importance...")
            plot_feature_importance(model, Model_name)
        elif Mode == 'simulate':
            print("Simulating activity data...")
            run_simulation()

if __name__ == "__main__":
    print("Welcome to the Human Activity Recognition Project!")
    print("This project uses LSTM to classify human activities based on sensor data.")
    print("Press -h or --help for usage instructions.")
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help", "help"):
        print_usage()
        print_models()
        sys.exit(0)
    mode_list = ['train', 'test', 'plot', 'simulate']
    # Default values
    epochs = 10
    Mode = None
    model_name = 'base_model'

    # Parse arguments if provided
    # ---- One argument provided ----
    # If epochs is provided, convert it to int
    if 3 > len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            epochs = int(sys.argv[1])
        elif sys.argv[1] in mode_list:
            Mode = sys.argv[1]
        else:
            model_name = sys.argv[1]
            if not model_name.endswith('.pth'):
                model_name += '.pth'

    # ---- Two arguments provided ----
    # If epochs and mode is provided
    if 4 > len(sys.argv) > 2:
        if sys.argv[1].isdigit():
            epochs = int(sys.argv[1])
            if sys.argv[2] in mode_list:
                Mode = sys.argv[2]
            else:
                print(f"Invalid mode: {sys.argv[2]}. Available modes: {', '.join(mode_list)}")
        elif sys.argv[1] in mode_list:
            Mode = sys.argv[1]
            model_name = sys.argv[2]
            if not model_name.endswith('.pth'):
                model_name += '.pth'
        elif sys.argv[1] not in mode_list:
            print(f"Invalid mode: {sys.argv[1]}. Available modes: {', '.join(mode_list)}")
            sys.exit(1)

    # -- Three arguments provided --
    if len(sys.argv) > 3:
        if sys.argv[1].isdigit():
            epochs = int(sys.argv[1])
        else:
            print(f"Invalid epochs: {sys.argv[1]}. It should be a number.")
            sys.exit(1)
        Mode = sys.argv[2] if sys.argv[2] in mode_list else None
        if not sys.argv[3].endswith('.pth'):
            model_name = sys.argv[3] + '.pth'
        else:
            model_name = sys.argv[3]
    main(epochs, Mode, model_name)