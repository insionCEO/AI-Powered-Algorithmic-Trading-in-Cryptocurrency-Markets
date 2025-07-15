import os

import pandas as pd
import plotly.express as px


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
    dataset_files.remove("BTC-USD.csv.dvc")

    # Initialize an empty list to store the data
    data = []

    # Iterate through the files
    for file in dataset_files:
        # Construct the file path
        file_path = "../../datasets/" + file

        # Load the data from the file and append it to the list
        data.append(pd.read_csv(file_path))

    # Return the list of data
    return data


def plot_line_graph(data_df, date_label, price_label):
    """
    This function plots a time-series line graph for price over time

    Parameters:
    data_df (pandas DataFrame): a DataFrame containing the data to be plotted
    date_label (str): the label for the column to be used as the x-axis
    price_label (str): the label for the column to be used as the y-axis

    Returns:
    None
    """
    labels = {date_label: "Date", price_label: "Price"}
    fig = px.line(data_df, x=date_label, y=price_label, labels=labels)

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15min", step="minute", stepmode="backward"),
                dict(count=30, label="30m", step="minute", stepmode="backward"),
                dict(count=1, label="1hr", step="hour", stepmode="backward"),
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show()


def convert_data(data_df):
    """
    This function converts the bids/ask distance values in the Keggle dataset to actual
    prices at the bid/ask levels. It also creates four new DataFrames: bids_prices, bids_volume, ask_prices,
    and ask_volume.

    Parameters:
    BTC_Data (pandas DataFrame): a DataFrame containing the bids distance values to be converted

    Returns:
    bids_prices (pandas DataFrame): a DataFrame containing the converted bids prices
    bids_volume (pandas DataFrame): a DataFrame containing the bids volume values
    ask_prices (pandas DataFrame): a DataFrame containing the converted ask prices
    ask_volume (pandas DataFrame): a DataFrame containing the ask volume values
    """
    # Initialize empty DataFrames for the bids prices, bids volume, ask prices, and ask volume
    bids_prices = pd.DataFrame()
    bids_volume = pd.DataFrame()
    ask_prices = pd.DataFrame()
    ask_volume = pd.DataFrame()

    # Iterate through the columns of BTC_Data
    for col in data_df:
        if col.startswith('bids_distance'):
            bids_prices[col] = data_df[col] * 100 + data_df.midpoint
        if col.startswith('asks_distance'):
            ask_prices[col] = data_df[col] * 100 + data_df.midpoint
        if col.startswith('bids_limit_notional'):
            bids_volume[col] = data_df[col]
        if col.startswith('asks_limit_notional'):
            ask_volume[col] = data_df[col]

    return bids_prices, bids_volume, ask_prices, ask_volume


def plot_order_book(combined_prices, combined_volume, time_stamp):
    """
    This function generates a bar chart showing the order book trades for a given timestamp.

    Parameters:
    combined_prices (pandas DataFrame): a DataFrame containing the bid/ask prices
    combined_volume (pandas DataFrame): a DataFrame containing the bid/ask volumes
    time_stamp (str): the timestamp to filter the data by

    Returns:
    None
    """
    # Filter the combined_prices and combined_volume DataFrames by the given timestamp
    bid_ask = combined_prices.loc[combined_prices["date"] == time_stamp]
    volume = combined_volume.loc[combined_volume["date"] == time_stamp]

    # Create a new DataFrame with the bid/ask prices and volumes
    output = pd.DataFrame()
    output["bid_ask_price"] = bid_ask.values[0][:-1]
    output["bid_ask_volume"] = volume.values[0][:-1]

    # Create a list of colors for the bars
    colors = ['Bid', ] * 15
    colors = colors + ['Ask', ] * 15

    # Create the bar chart using Plotly express
    fig = px.bar(output,
                 x='bid_ask_price', y='bid_ask_volume',
                 color=colors,
                 color_discrete_sequence=["#008000", "#FF0000"],
                 labels={"bid_ask_price": "Price", "bid_ask_volume": "Volume"},
                 title="Order Book Trades")

    # Update the x-axis to be a categorical axis
    fig.update_xaxes(type='category', categoryorder="category ascending", visible=False)

    # Display the chart
    fig.show()


def extract_time_lob_data(df):
    """
    Extract ask and bid prices and their corresponding volume from a DataFrame and return a merged DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to extract data from.

    Returns:
    pandas.DataFrame: A merged DataFrame containing the extracted ask and bid data.
    """
    ask_bid_values = pd.melt(df, id_vars=["date"], value_vars=[col for col in df.columns if
                                                               col.startswith("bids_distance") or col.startswith(
                                                                   "asks_distance")],
                             var_name="distance_type", value_name="distance_value")

    ask_bid_volume = pd.melt(df, id_vars=["date"], value_vars=[col for col in df.columns if
                                                               col.startswith("bids_limit") or col.startswith(
                                                                   "asks_limit")],
                             var_name="limit_type", value_name="limit_value")

    ask_bid_values["distance_num"] = ask_bid_values["distance_type"].str.extract(r'(\d+)')

    # Extract the numeric part of the "limit_type" column
    ask_bid_volume["limit_num"] = ask_bid_volume["limit_type"].str.extract(r'(\d+)')

    # Extract the "bid" or "ask" part of the "distance_type" column
    ask_bid_values["distance_level"] = ask_bid_values["distance_type"].str.extract(r'(bid|ask)')

    # Extract the "bid" or "ask" part of the "limit_type" column
    ask_bid_volume["limit_level"] = ask_bid_volume["limit_type"].str.extract(r'(bid|ask)')

    # Merge the two data frames on the "date", "limit_num", and "limit_level" columns
    merged_df = pd.merge(ask_bid_values, ask_bid_volume, left_on=["date", "distance_num", "distance_level"],
                         right_on=["date", "limit_num", "limit_level"])

    return merged_df
