# Library import
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import math
import pickle
from scipy.special import inv_boxcox
import requests_cache
from retry_requests import retry
import openmeteo_requests
import pytz
import time


# Helper Functions
def great_circle_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    distance = math.acos(
        math.sin(lat1) * math.sin(lat2) +
        math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * 6371
    return distance

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(dLon))
    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def get_weather_data(latitude, longitude, date_time):
    url = 'https://historical-forecast-api.open-meteo.com/v1/forecast'
    latitude = round(latitude, 2)
    longitude = round(longitude, 2)
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': date_time.strftime('%Y-%m-%d'),
        'end_date': date_time.strftime('%Y-%m-%d'),
        'hourly': [
            'surface_pressure',
            'precipitation',
            'rain',
            'sunshine_duration',
        ]
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        timestamps = pd.to_datetime(
            hourly.Time(), unit="s", utc=True).tz_convert(pst)
        time_diffs = np.abs(timestamps - date_time)
        idx = np.argmin(time_diffs)
        weather_data = {
            'surface_pressure': hourly.Variables(0).ValuesAsNumpy()[idx],
            'precipitation': hourly.Variables(1).ValuesAsNumpy()[idx],
            'rain': hourly.Variables(2).ValuesAsNumpy()[idx],
            'sunshine_duration': hourly.Variables(3).ValuesAsNumpy()[idx],
        }
        return weather_data
    except Exception as e:
        print(f"Error fetching data for "
              f"{date_time}, {latitude}, {longitude}: {e}")
        return None

def transform_duration_to_text(duration):
    hours = int(duration)
    minutes = round((duration - hours) * 60)
    if hours > 0:
        return (f"{hours} hr{'s' if hours > 1 else ''} "
                f"{minutes} min{'s' if minutes != 1 else ''}")
    else:
        return f"{minutes} min{'s' if minutes != 1 else ''}"


# Title Display
st.title('Truck Cycle Time Estimator App')
st.write('\n')


# Date Section
st.subheader('Date', divider='gray')
input_date = st.date_input('Select a date', value=None)
st.write('Confirmed: ', input_date)
st.write('\n')


# Basis Section
time_basis_input = 'TIMEOUT'


# Generate Estimates Section
st.subheader('Compute for TCT', divider='gray')
if st.button('Compute'):
    st.write('Computing...')
    st.write('\n')

    
    # Data Reading
    sheet_name = time_basis_input.replace(' ', '%20')
    sheet_id = '1Pl3_F9QY721amRYWj8aUzT63hYDy836SJ8jQoGv86gE'
    url = (f"https://docs.google.com/spreadsheets/d/{sheet_id}"
           f"/gviz/tq?tqx=out:csv&sheet={sheet_name}")
    df = pd.read_csv(url, parse_dates=[time_basis_input])
    st.write(df[time_basis_input])
    df[time_basis_input] = df[time_basis_input].dt.time
    
    
    # Data Filtering
    df_filtered = df.copy()
    df_filtered = df_filtered.iloc[:, 1:].dropna(how='all')
    st.subheader('Trips', divider='gray')
    st.write(df_filtered)
    st.write('\n')
    keep_cols = [
        'RUNNING VOL',
        time_basis_input,
        'PLACEMENT FAMILY',
        'STRUCTURE FAMILY',
        'LATITUDE',
        'LONGITUDE',
    ]
    df_filtered = df_filtered[keep_cols]

    
    # Feature Engineering
    # Estimated Plant Out/Departure Time & Hour
    df_features = df_filtered.copy()
    df_features['EST PLANT OUT'] = (
        df_features.apply(
            lambda row: datetime.datetime.combine(
                input_date, row[time_basis_input]
            ), axis=1
        ))
    df_features['EST PLANT OUT'] = (
        df_features['EST PLANT OUT'] + pd.to_timedelta(5, unit='m'))
    df_features['dt_hour'] = df_features['EST PLANT OUT'].dt.hour

    # Radial Distance & Angle
    ref_lat, ref_lon = 14.902746, 120.838638
    df_features['radial_distance'] = df_features.apply(
        lambda row: great_circle_distance(
            row['LATITUDE'], row['LONGITUDE'], ref_lat, ref_lon), axis=1
    )
    df_features['angle'] = df_features.apply(
        lambda row: calculate_bearing(
            row['LATITUDE'], row['LONGITUDE'], ref_lat, ref_lon), axis=1
    )
    
    # Structure & Placement
    structure_dict = {
        'SUSPENDED SLAB (11TH FLOOR & UP)': 'SUSPENDED SLAB (2ND-10TH FLOOR)',
        'DECK STITCH': 'PRECAST/RETAINING WALL/ABUTMENT',
        'SHEAR KEY/SHEAR BLOCK': 'BEAM'
    }
    df['STRUCTURE FAMILY'] = df['STRUCTURE FAMILY'].replace(structure_dict)
    with open(
        'pickle_files/label_encoders/placement_label_encoder.pkl',
        'rb') as f:
        ple = pickle.load(f)
    with open(
        'pickle_files/label_encoders/structure_label_encoder.pkl',
        'rb') as f:
        sle = pickle.load(f)
    df_features['PLACEMENT FAMILY'] = ple.transform(df_features['PLACEMENT FAMILY'])
    df_features['STRUCTURE FAMILY'] = sle.transform(df_features['STRUCTURE FAMILY'])    

    # Feature Renaming
    df_features.rename(columns={
        'RUNNING VOL': 'running_volume',
        'PLACEMENT FAMILY': 'placement_type',
        'STRUCTURE FAMILY': 'structure_type',
    }, inplace=True)

    
    # Weather API Session Set-up
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)    
    pst = pytz.timezone('Asia/Manila')
    weather_cols = [
        'surface_pressure',
        'precipitation',
        'rain',
        'sunshine_duration',
    ]


    # Model 1: Travel Time from Plant to Client Jobsite
    # Data Preparation
    df_model_1 = df_features.copy()
    for col in weather_cols:
        df_model_1[col] = None
    for idx, row in df_model_1.iterrows():
        date_time = pd.to_datetime(row['EST PLANT OUT']).tz_localize(pst)
        latitude = ref_lat
        longitude = ref_lon
        weather_data = get_weather_data(latitude, longitude, date_time)
        if weather_data:
            for col in weather_cols:
                df_model_1.at[idx, col] = weather_data[col]
        # Adding a small delay to avoid hitting API rate limits
        time.sleep(0.01)
    model_1_features = [
        'dt_hour',
        'placement_type',
        'running_volume',
        'surface_pressure',
        'precipitation',
        'rain',
        'sunshine_duration',
        'radial_distance',
        'angle',
    ]
    df_model_1 = df_model_1[model_1_features]

    # Model Inference
    with open('pickle_files/models/model_1.pkl', 'rb') as f:
        pickle_1 = pickle.load(f)
        model_1 = pickle_1['model']
        scaler_1 = pickle_1['scaler']
        lambda_boxcox_1 = pickle_1['lambda_boxcox']
    df_model_1_scaled = scaler_1.transform(df_model_1)
    duration_1_pred_boxcox = model_1.predict(df_model_1_scaled)
    duration_1_pred = inv_boxcox(duration_1_pred_boxcox, lambda_boxcox_1) - 1e-6
    df_features['EST ARRIVAL AT JOBSITE'] = (
        df_features['EST PLANT OUT'] + pd.to_timedelta(duration_1_pred, unit='h')
    ).dt.floor('min')
    df_features['pred_atc_hour'] = df_features['EST ARRIVAL AT JOBSITE'].dt.hour
    df_features['pred_atc_day'] = df_features['EST ARRIVAL AT JOBSITE'].dt.weekday


    # Model 2: Time Spent/Stay at Client Jobsite
    # Data Preparation
    df_model_2 = df_features.copy()
    for col in weather_cols:
        df_model_2[col] = None
    for idx, row in df_model_2.iterrows():
        date_time = pd.to_datetime(row['EST ARRIVAL AT JOBSITE']).tz_localize(pst)
        latitude = row['LATITUDE']
        longitude = row['LONGITUDE']
        weather_data = get_weather_data(latitude, longitude, date_time)
        if weather_data:
            for col in weather_cols:
                df_model_2.at[idx, col] = weather_data[col]
        time.sleep(0.01)
    model_2_features = [
        'structure_type',
        'placement_type',
        'pred_atc_hour',
        'running_volume',
        'pred_atc_day',
        'surface_pressure',
        'sunshine_duration',
        'radial_distance',
        'angle',
    ]
    df_model_2 = df_model_2[model_2_features]
    
    # Model Inference
    with open('pickle_files/models/model_2.pkl', 'rb') as f:
        pickle_2 = pickle.load(f)
        model_2 = pickle_2['model']
        scaler_2 = pickle_2['scaler']
        lambda_boxcox_2 = pickle_2['lambda_boxcox']
    df_model_2_scaled = scaler_2.transform(df_model_2)
    duration_2_pred_boxcox = model_2.predict(df_model_2_scaled)
    duration_2_pred = inv_boxcox(duration_2_pred_boxcox, lambda_boxcox_2) - 1e-6
    df_features['EST DEPARTURE FROM JOBSITE'] = (
        df_features['EST ARRIVAL AT JOBSITE'] +
        pd.to_timedelta(duration_2_pred, unit='h')).dt.floor('min')
    df_features['pred_dfc_hour'] = df_features['EST DEPARTURE FROM JOBSITE'].dt.hour


    # Model 3: Travel Time of Return from Client Jobsit to Plant
    # Data Preparation
    df_model_3 = df_features.copy()
    for col in weather_cols:
        df_model_3[col] = None
    for idx, row in df_model_3.iterrows():
        date_time = pd.to_datetime(
            row['EST DEPARTURE FROM JOBSITE']).tz_localize(pst)
        latitude = row['LATITUDE']
        longitude = row['LONGITUDE']
        weather_data = get_weather_data(latitude, longitude, date_time)
        if weather_data:
            for col in weather_cols:
                df_model_3.at[idx, col] = weather_data[col]
        time.sleep(0.01)
    model_3_features = [
        'placement_type',
        'running_volume',
        'surface_pressure',
        'precipitation',
        'rain',
        'sunshine_duration',
        'pred_dfc_hour',
        'radial_distance',
        'angle',
    ]
    df_model_3 = df_model_3[model_3_features]

    # Model Inference
    with open('pickle_files/models/model_3.pkl', 'rb') as f:
        pickle_3 = pickle.load(f)
        model_3 = pickle_3['model']
        scaler_3 = pickle_3['scaler']
        lambda_boxcox_3 = pickle_3['lambda_boxcox']
    df_model_3_scaled = scaler_3.transform(df_model_3)
    duration_3_pred_boxcox = model_3.predict(df_model_3_scaled)
    duration_3_pred = inv_boxcox(duration_3_pred_boxcox, lambda_boxcox_3) - 1e-6
    df_features['EST PLANT IN'] = (
        df_features['EST DEPARTURE FROM JOBSITE'] +
        pd.to_timedelta(duration_3_pred, unit='h')).dt.floor('min')


    # Timestamp Results
    df_results = df_features.copy()
    results_cols = [
        'EST PLANT OUT',
        'EST ARRIVAL AT JOBSITE',
        'EST DEPARTURE FROM JOBSITE',
        'EST PLANT IN',
    ]
    df_results = df_results[results_cols]
    st.subheader('Estimate Times', divider='gray')
    st.write(df_results)
    st.write('\n')
    
    
    # Duration Results
    duration_dict = {
        'TRAVEL DURATION TO JOBSITE':
        [transform_duration_to_text(d) for d in duration_1_pred],
        'DURATION OF STAY AT JOBSITE':
        [transform_duration_to_text(d) for d in duration_2_pred],
        'RETURN DURATION BACK TO PLANT':
        [transform_duration_to_text(d) for d in duration_3_pred],
    }
    df_duration = pd.DataFrame(duration_dict)
    st.subheader('Estimate Durations', divider='gray')
    st.write(df_duration)
    st.write('\n')
