import numpy as np
import pandas as pd
from geopy import distance
from geopy.point import Point
import math
import holidays
from scipy import stats

def get_distance_km(df):
    pickup_coords = (df['pickup_latitude'], df['pickup_longitude'])
    dropoff_coords = (df['dropoff_latitude'], df['dropoff_longitude'])
    return distance.geodesic(pickup_coords, dropoff_coords).km

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'unknown'

def time_period(hour):
    if 5 <= hour < 12:  
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    else:
        return 'Night'

def calculate_direction(row):
    pickup_coordinates = Point(row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coordinates = Point(row['dropoff_latitude'], row['dropoff_longitude'])
    delta_longitude = dropoff_coordinates[1] - pickup_coordinates[1]
    y = math.sin(math.radians(delta_longitude)) * math.cos(math.radians(dropoff_coordinates[0]))
    x = math.cos(math.radians(pickup_coordinates[0])) * math.sin(math.radians(dropoff_coordinates[0])) - \
        math.sin(math.radians(pickup_coordinates[0])) * math.cos(math.radians(dropoff_coordinates[0])) * \
        math.cos(math.radians(delta_longitude))
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing

def manhattan_distance(df):
    lat_distance = abs(df['pickup_latitude'] - df['dropoff_latitude']) * 111  
    lon_distance = abs(df['pickup_longitude'] - df['dropoff_longitude']) * 111 * math.cos(math.radians(df['pickup_latitude']))
    return lat_distance + lon_distance

def remove_outlier(df, feature_column, target_column, factor=3):
    zscore = np.abs(stats.zscore(df[[feature_column, target_column]]))
    filtered_rows = (zscore < factor).all(axis=1)
    df_cleaned = df[filtered_rows].copy()
    return df_cleaned

def is_holiday(date, country='US'):
    us_holidays = holidays.country_holidays(country)
    return date in us_holidays

