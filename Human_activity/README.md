# Human Activity Recognition

This project utilizes an LSTM approach to classify human activities based on sensor data from a waist-mounted smartphone. The data can be acquired from [Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
The activities are WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, STANDING, SITTING, and LAYING.

## How to run in cmd

### Install dependencies

```pip install -r requirements```

### Run main

- run within the ML_Project folder:

```python Human_activities/main.py```

- run within the Human_activities folder:

```python main.py```

- args:

```text
"""
python main.py [EPOCHS] [MODE] [Model_name]

EPOCHS: how many epochs you want to use to train the model (Default: 1000)

MODE: [train, test, plot, simulate] If you want to run a specific task (Default: None)

Model_name: the model name you want to use (Default: base_model.pth)

- h: Overview of args and available models (if they exist)

"""
```

## The project

The project is divided into 8 python files which contains different functionalities. The files are as follows:

- data_cleaning.py
  - preprocess the data by removing nan, converting it to a dataframe and then scales it.

- lstm.py
  - Converts the cleaned data into a Tensor-Dataloader, which is then assigned to a Dataloader. The LSTM model is also defined
- simulate_data.py
  - Simulates the data based on a user-defined choice. The simulation uses the body X,y and Z values inside the train Inertial Signals folder, and then plots them to a graph with a span of 2.56 seconds. The graphs are saved in the folder ```/results/```. The user can either simulate one of the activities, or chose to simulate everything.
- results.py
  - Plots the accuracy and loss value of the model based on the epochs. It also plots the top 5 most important features used to determine the classes. The graphs are saved in a the folder ```/results/```.
- train.py
  - Trains the initialized model with user-defined epoch and model value if preferred.Default is 1000 epochs and base_model.pth(if it exists, else train a new one). Also saves the train/test accuracy and loss values each epoch to file.
- test.py
  - Tests the model on the train data
- utils.py
  - Different helper functions to save code space. Functions such as save_model, load_model, read_file and save file.
- main.py
  - The main program. Uses all the files above to create a user-friendly approach for the user to use the program.

## Results

This is the result of the model. The results contain the graphs for the simulation of all activities, loss and accuracy of the model, and the top 5 features used to classify.

### Activity simulation

<img width="1200" height="600" alt="simulated_activity_WALKING" src="https://github.com/user-attachments/assets/3d55bb1e-9cb8-433c-9bef-a6caf15171f5" />
<img width="1200" height="600" alt="Simulated_Activity_WALKING_UPSTAIRS" src="https://github.com/user-attachments/assets/72257b61-67fc-4936-b8a7-bcd5717cf609" />
<img width="1200" height="600" alt="Simulated_Activity_WALKING_DOWNSTAIRS" src="https://github.com/user-attachments/assets/971c2285-e826-4e53-8648-b80a25b02f6e" />
<img width="1200" height="600" alt="Simulated_Activity_LAYING" src="https://github.com/user-attachments/assets/04a43b89-32d1-4b4b-aaed-9de5cf502c67" />
<img width="1200" height="600" alt="Simulated_Activity_SITTING" src="https://github.com/user-attachments/assets/ac80271d-3526-450e-8677-ba5a5b8e4f6f" />
<img width="1200" height="600" alt="simulated_activity_STANDING" src="https://github.com/user-attachments/assets/9886a0f8-e2a7-47fe-91fb-482180a73506" />

### Accuracy and Loss

<img width="1200" height="600" alt="loss_accuracy_curve_base_model" src="https://github.com/user-attachments/assets/ed903bfd-9260-4883-8557-aa605ad15b00" />

### Top 5 features

<img width="800" height="400" alt="top5feature_importance_base_model pth" src="https://github.com/user-attachments/assets/767ed94d-3374-4bae-962e-db4eabfaaaf2" />
