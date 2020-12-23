# Libraries
import pandas as pd
import numpy as np

# ENVIRONMENT VARIABLES

# Number of Modes in the dataset
# #1: car
# #2: pass
# #3: bus
# #4: train
# #5: walk
# #6: bike
NUM_OF_MODES = 6
MODES = ["car", "pass", "bus", "train", "walk", "bike"] 



# Reading CSV file
def read_csv(filepath):
    return pd.read_csv(filepath, sep=";")


# Function to normalize the matrix
def normalize(x):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    return (x - xmin) / (xmax - xmin)

# Fetch attribute
def get_attributes(df, model):
        # Initialize the variables
        cost  = np.zeros((len(df), NUM_OF_MODES))
        time  = np.zeros((len(df), NUM_OF_MODES))
        avail = np.zeros((len(df), NUM_OF_MODES))
        tw    = np.zeros((len(df), NUM_OF_MODES))
        tg    = np.zeros((len(df), NUM_OF_MODES))

        for i in range(NUM_OF_MODES):
            # COST
            if MODES[i] + "_cost" in df.columns:
                cost[:, i] = df[MODES[i] + "_cost"]
            # TIME
            if (MODES[i] + "_time" in df.columns) and (MODES[i] + "_w_time" not in df.columns) and (MODES[i] + "_g_time" not in df.columns):
                time[:, i] = df[MODES[i] + "_time"]
            # AVAILABILITY
            if MODES[i] + "_ok" in df.columns:
                avail[:, i] = df[MODES[i] + "_ok"]
            # WAITING TIME FOR MODE
            if MODES[i] + "_w_time" in df.columns:
                tw[:, i] = df[MODES[i] + "_w_time"]
            # WALKING TIME FOR MODE
            if MODES[i] + "_g_time" in df.columns:
                tg[:, i] = df[MODES[i] + "_g_time"]

        # MODE CHOICE - ONE HOT SHOT MATRIX
        mode_choosen  = df["mode"].values
        mode_choosen = mode_choosen - 1

        # Mark all the zeros as NaN in avail matrix
        avail = np.where(avail==0, np.NaN, avail)

        # ATTRIBUTES for "BASE MODEL".
        if model == "base":
            attributes = [cost, time, avail, tw, tg, mode_choosen]
            return attributes
        # ATTRIBUTES for "REDUCED MODEL".
        if model == "reduced":
            time = time + tw + tg
            attributes = [cost, time, avail, mode_choosen]
            return attributes
        
        else:
            return None


if __name__ == "__main__":
    # Reading data
    filename = "data/modeData.csv"
    df = read_csv(filename)

    cost, time, avail, tw, tg, mode_choosen = get_attributes(df, model="base")