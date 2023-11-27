import numpy as np
import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    # Load JSON data from a file
    with open(file_path) as f:
        data = json.load(f)

    # Convert the JSON data to a DataFrame
    df = pd.json_normalize(data)
        
    return df

def preprocess_data(data, name):
    
    # Sort by timestamp
    data = data.sort_values(by='timestamp')
    
    # Dropping features to simplify analisys or because there is no correlation
    data.drop(['hash', 'height', 'fee_symbol'], axis=1, inplace=True)
    
    # Drop rows with missing values
    data.dropna(inplace=True)
        
    # Extract relevant information from 'sub_transactions'
    # Convert 'amount' to numeric before multiplication 
    data['total_amount_usd'] = data['sub_transactions'].apply(lambda x: sum([float(output['amount']) * x[0]['unit_price_usd'] for output in x[0]['outputs']]))
    
    # Convert fee to float
    data['fee'] = data['fee'].astype(float)
    
    # Creation of the lag features
    # Specify the number of lag steps you want to include
    num_lags = 50
    # Create lag features for relevant columns
    for i in range(1, num_lags + 1):
        for col in ['total_amount_usd']:
            data[f'{col}_lag_{i}'] = data[col].shift(i)
            
    # Define a Binary Target Variable
    # Our binary target variable will be whether a transaction is a whale transaction
    # As referenced in our presentation, we categorize whale transactions according to its size
    # Scaling our whales
    scale = {'DOGE' : 1, 'BTC' : 8, 'SOL' : 1, 'ETH' : 1, 'LTC' : 1, 'ALGO' : 1}
    conditions = [
        (data['total_amount_usd'] < float(scale[name]) * 100e3),  # Tiny Fish < 100m USD
        (data['total_amount_usd'] >= float(scale[name]) * 100e3) & (data['total_amount_usd'] < float(scale[name]) * 200e3),  # Small Fish  < 200m USD
        (data['total_amount_usd'] >= float(scale[name]) * 200e3) & (data['total_amount_usd'] < float(scale[name]) * 500e3),  # Medium Fish < 500m USD
        (data['total_amount_usd'] >= float(scale[name]) * 500e3) & (data['total_amount_usd'] < float(scale[name]) * 1e6),  # Big Fish  < 1M USD
        (data['total_amount_usd'] >= float(scale[name]) * 1e6) & (data['total_amount_usd'] < float(scale[name]) * 5e6),  # Small Whale  < 5M USD
        (data['total_amount_usd'] >= float(scale[name]) * 5e6) & (data['total_amount_usd'] < float(scale[name]) * 10e6),  # Medium Whale  < 10M USD
        (data['total_amount_usd'] >= float(scale[name]) * 10e6) & (data['total_amount_usd'] < float(scale[name]) * 20e6),  # Big Whale  < 20M USD
        (data['total_amount_usd'] >= float(scale[name]) * 20e6) & (data['total_amount_usd'] < float(scale[name]) * 50e6),  # Huge Whale  < 50M USD
        (data['total_amount_usd'] >= float(scale[name]) * 50e6)  # Mega Whale > 50M
    ]
    choices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    data['whale_class'] = np.select(conditions, choices, default=0)
    # Drop Original Sub-Transactions Column
    data.drop(['sub_transactions'], axis=1, inplace=True)
    

    # Create datetime col
    data['timestamp_datetime'] = pd.to_datetime(data['timestamp'], unit='s')  # Convert to datetime if not already
    # Create hour col and drop timestamp_datetime
    data['hour'] = data['timestamp_datetime'].dt.hour
    data.drop(['timestamp_datetime'], axis=1, inplace=True)
    # Create max_whale_class_next_hour
    data['max_whale_class_next_hour'] = data.groupby('hour')['whale_class'].shift(-1).groupby(data['hour']).transform('max')
    # Convert the column to integer type
    data['max_whale_class_next_hour'] = data['max_whale_class_next_hour'].astype(int)
    
    # Drop rows with NaN values resulting from lag creation
    data = data.dropna()
    
    return data


def export_npz(df, name):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['max_whale_class_next_hour'], axis=1), df['max_whale_class_next_hour'], test_size=0.3, random_state=42)
    # Save the compressed data
    np.savez_compressed('data/'+name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

def get_info(df):
    print("Shape: ", df.shape)
    print(df.describe())
    print(df.info())
    print(df)
    print(df['max_whale_class_next_hour'].value_counts())
    print(df['whale_class'].value_counts())

if __name__ == '__main__':
    file_paths_names = [
        ('data/txs_dogecoin_2023-01-03.json', 'DOGE'),
        ('data/txs_bitcoin_2023-01-03.json', 'BTC'),
        #('data/txs_solana_2023-01-03.json', 'SOL'),
        #('data/txs_ethereum_2023-01-03.json', 'ETH'),
        ('data/txs_litecoin_2023-01-03.json', 'LTC'),
        #('data/txs_algorand_2023-01-03.json', 'ALGO'),  
    ]
    
    print(f'\n\n4th Dataton Cryptocurrencies FGV EESP')
    print(f'HAS Whale predictor\n\n')
    
    for file_path, name in file_paths_names:
        print(f'\nLoading {name}...\n')
        df = load_data(file_path)
        df_preprocessed = preprocess_data(df, name)
        export_npz(df_preprocessed, name)
        get_info(df_preprocessed)