import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_path = 'Human_activity/UCI HAR Dataset/'
activity_labels = pd.read_csv(dataset_path+'activity_labels.txt', header=None, sep='\s+', names=['activity_id', 'activity_name'])
activity_labels.set_index('activity_id', inplace=True)

# Get Body X, Y and Z accelerometer data
body_acc_x = pd.read_csv(dataset_path+'train/Inertial Signals/body_acc_x_train.txt', header=None, sep='\s+')
body_acc_y = pd.read_csv(dataset_path+'train/Inertial Signals/body_acc_y_train.txt', header=None, sep='\s+')
body_acc_z = pd.read_csv(dataset_path+'train/Inertial Signals/body_acc_z_train.txt', header=None, sep='\s+')
y_train = pd.read_csv(dataset_path+'train/y_train.txt', header=None, names=['activity_id'])

def show_activity_labels():
    """
    Displays the activity labels from the dataset.
    """
    print("Activity Labels:")
    for index, row in activity_labels.iterrows():
        print(f"{index}: {row['activity_name']}")


def simulate_activity_data(activity_id:int, num_windows: int = 1):
    """
    Simulates human activity data for testing purposes.
    
    Returns:
        
    """
    if activity_id not in activity_labels.index:
        raise ValueError(f"Invalid activity_id: {activity_id}. Must be one of {activity_labels.index.tolist()}.")
    label_indice = y_train[y_train['activity_id'] == activity_id].index
    
    
    if not label_indice.empty:
        # Aquire the data for the first occurrence of the activity_id
        first_index = label_indice[0]
        acc_x_label = body_acc_x.iloc[first_index]
        acc_y_label = body_acc_y.iloc[first_index]
        acc_z_label = body_acc_z.iloc[first_index]
        # Create a time vector for the data
        # Assuming the data is sampled at 50Hz, we can create a time vector of 128 samples
        # which corresponds to 2.56 seconds of data
        seconds = num_windows * 2.56
        samples = num_windows * 128
        time = np.linspace(0, seconds, samples)
        activity_name = activity_labels.loc[activity_id, 'activity_name']

        # Plot the data and simulate the activity
        plt.figure(figsize=(12, 6))
        plt.plot(time, acc_x_label, label='Body Acc X', color='blue')
        plt.plot(time, acc_y_label, label='Body Acc Y', color='green')
        plt.plot(time, acc_z_label, label='Body Acc Z', color='red')
        plt.title(f'Simulated Activity Data for {activity_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Acceleration (g)')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print(f"No data found for activity_id: {activity_id}. Please check the dataset.")


def run_simulation():
    """
    Runs the simulation for all activities in the dataset.
    """
    print("Running simulation for a specific activity...")
    show_activity_labels()
    activity_id = int(input("Enter the activity ID (1-6): "))
    num_windows = int(input("Enter the number of windows to simulate (default is 1): ") or 1)
    simulate_activity_data(activity_id, num_windows)