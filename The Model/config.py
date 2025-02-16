import os

# File paths
DATA_DIR = r"D:\Ai courses\ML mostafa saad\projects\taxi duration"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# Model saving path (saved in the same directory as the script)
MODEL_PATH = os.path.join(os.getcwd(), "saved_models", "model.pkl")

# Features
NUMERIC_COLUMNS = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 
                   'distance_short', 'dirction', 'mnhattan_short_path', 'pick_hour','is_holiday']

CATEGORICAL_COLUMNS = ['pick_season', 'pick_time_period', 'is_weekend']
