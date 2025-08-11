# Human Activity Recognition

This project utilizes an LSTM approach to classify human activities based on sensor data from a waist-mounted smartphone. The data can be acquired from ![Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).
The activities are WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, STANDING, SITTING, and LAYING.

## The project

The project is divided into 8 python files which contains different functionalities. The files are as follows:

* data_cleaning.py
  * preprocess the data by removing nan, converting it to a dataframe and then scales it.

* lstm.py
  * Converts the cleaned data into a Tensor-Dataloader, which is then assigned to a Dataloader. The LSTM model is also defined
* simulate_data.py
  * Simulates the data based on a user-defined choice. The simulation uses the body X,y and Z values inside the train Inertial Signals folder, and then plots them to a graph with a span of 2.56 seconds. The graphs are saved in the folder *results*.
* results.py
  * Plots the accuracy and loss value of the model based on the epochs. It also plots the top 5 most influential features used to determine the class. The graphs are saved in a the folder *results*.
* train.py
  * Trains the initialized model with user-defined epoch and model value if preferred.Default is 1000 epochs and base_model.pth(if it exists, else train a new one). Also saves the train/test accuracy and loss values each epoch to file.
* test.py
  * Tests the model on the train data
* utils.py
  * Different helper functions to save code space. Functions such as save_model, load_model, read_file and save file.
* main.py
  * The main program. Uses all the files above to create a user-friendly approach for the user to use the program.

## Results

### Activity simulation

### Accuracy and Loss

### Top 5 features
