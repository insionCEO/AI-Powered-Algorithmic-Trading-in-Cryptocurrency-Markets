import os 

import pandas as pd
import matplotlib.pyplot as plt

def load_data(folder_path):
    """
    This function loads all the data from the specified folder and returns it as a list of DataFrames.

    Parameters:
    folder_path (str): the path to the folder containing the data files

    Returns:
    data (list): a list of pandas DataFrames, one for each data file
    """
    # Get a list of all the non-hidden files in the folder
    dataset_files = [f for f in os.listdir(folder_path) if not f.startswith('.')]

    # Remove the ".dvc" file from the list for now
    for file in dataset_files:
        if file.endswith('.dvc'):
            dataset_files.remove(file)

    # Initialize an empty list to store the data
    data = []

    # Iterate through the files
    for file in dataset_files:
        # Construct the file path
        file_path = "data/raw/" + file

        # Load the data from the file and append it to the list
        data.append(pd.read_csv(file_path))

    # Return the list of data
    return data

def visualize_data(data_frame, column_name):
    data = data_frame[column_name].values
    plt.plot(data)
    plt.show()
