import pandas as pd
import numpy as np
from utils import get_distance_km, calculate_direction, manhattan_distance, month_to_season, time_period, is_holiday

def prepare(df, country='US'):
    df['trip_duration'] = np.log1p(df['trip_duration'])
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['distance_short'] = df.apply(get_distance_km, axis=1)
    df['dirction'] = df.apply(calculate_direction, axis=1)
    df['mnhattan_short_path'] = df.apply(manhattan_distance, axis=1)
    
    df['distance_short'] = np.log1p(df['distance_short'])
    df['mnhattan_short_path'] = np.log1p(df['mnhattan_short_path'])
    df['dirction'] = np.log1p(df['dirction'])
    
    df['pick_hour'] = df['pickup_datetime'].dt.hour
    df['pick_day_of_week'] = df['pickup_datetime'].dt.day_of_week
    df['pick_day'] = df['pickup_datetime'].dt.day
    df['pick_month'] = df['pickup_datetime'].dt.month
    df['is_weekend'] = df['pickup_datetime'].dt.dayofweek >= 5
    df['pick_season'] = df['pick_month'].apply(month_to_season)
    df['pick_time_period'] = df['pick_hour'].apply(time_period)
    
    # Add holiday feature
    df['is_holiday'] = df['pickup_datetime'].apply(lambda x: is_holiday(x, country))
    
    df.drop(columns=['pickup_datetime', 'id'], inplace=True)
    return df
