import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by removing rows with NaN values and resetting the index.
    
    Args:
        df (pd.DataFrame): The DataFrame to be cleaned.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame with NaN values removed and index reset.
    """
    # Remove rows with NaN values
    df_cleaned = df.dropna()
    
    # Reset the index of the DataFrame
    df_cleaned.reset_index(drop=True, inplace=True)
    
    return df_cleaned

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "UCI HAR Dataset")

X_test_path = os.path.join(DATASET_DIR, "test", "X_test.txt")
Y_test_path = os.path.join(DATASET_DIR, "test", "Y_test.txt")
X_train_path = os.path.join(DATASET_DIR, "train", "X_train.txt")
Y_train_path = os.path.join(DATASET_DIR, "train", "Y_train.txt")
features_path = os.path.join(DATASET_DIR, "features.txt")

def setup_data() -> pd.DataFrame:
    """
    Cleans the input DataFrame by removing rows with NaN values and resetting the index.
    
    Args:
        df (pd.DataFrame): The DataFrame to be cleaned.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame with NaN values removed and index reset.
    """

    X_test = pd.read_csv(X_test_path, sep='\s+', header=None)
    Y_test = pd.read_csv(Y_test_path, sep='\s+', header=None)

    X_train = pd.read_csv(X_train_path, sep='\s+', header=None)
    Y_train = pd.read_csv(Y_train_path, sep='\s+', header=None)


    # Concatenate features and labels for test and train sets
    test = pd.concat([X_test, Y_test], axis=1)
    train = pd.concat([X_train, Y_train], axis=1)


    # Add feature names to the DataFrame based on features.txt
    features = pd.read_csv(features_path, sep='\s+', header=None)
    feature_names = features[1].tolist()
    test.columns = feature_names + ['Activity']
    train.columns = feature_names + ['Activity']

    # Clean the data
    test_cleaned = clean_data(test)
    train_cleaned = clean_data(train)

    st = StandardScaler()
    # Scale the features
    train_cleaned.iloc[:, :-1] = st.fit_transform(train_cleaned.iloc[:, :-1])
    test_cleaned.iloc[:, :-1] = st.transform(test_cleaned.iloc[:, :-1])
    
    return train_cleaned, test_cleaned