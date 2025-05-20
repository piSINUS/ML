import time
import cryptocompare
import numpy as np
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random


coin_list = cryptocompare.get_coin_list()
coins = list(coin_list.keys())

random_coins = ['BTC'] + random.sample(coins, min(99, len(coins)-1))


def get_prices(ticker, timeframe):
    time.sleep(0.5)  
    try:
        if timeframe == 'day':
            return cryptocompare.get_historical_price_day(ticker, currency='USD', limit=30)
        elif timeframe == 'hour':
            return cryptocompare.get_historical_price_hour(ticker, currency='USD', limit=72)
        elif timeframe == 'minute':
            return cryptocompare.get_historical_price_minute(ticker, currency='USD', limit=60)
    except:
        return None


prices_daily = {coin: get_prices(coin, 'day') for coin in random_coins}
prices_hourly = {coin: get_prices(coin, 'hour') for coin in random_coins}
prices_minutely = {coin: get_prices(coin, 'minute') for coin in random_coins}


common_length = min(
    min((len(v) for v in prices_daily.values() if v is not None), default=9999),
    min((len(v) for v in prices_hourly.values() if v is not None), default=9999),
    min((len(v) for v in prices_minutely.values() if v is not None), default=9999)
)

def trim_prices(prices_dict, length):
    return {coin: data[:length] for coin, data in prices_dict.items() if data is not None and len(data) >= length}

prices_daily = trim_prices(prices_daily, common_length)
prices_hourly = trim_prices(prices_hourly, common_length)
prices_minutely = trim_prices(prices_minutely, common_length)


def extract_prices(prices_dict):
    return {coin: [entry['close'] for entry in data] if data else [] for coin, data in prices_dict.items()}

df_daily = pd.DataFrame(extract_prices(prices_daily)).T
df_hourly = pd.DataFrame(extract_prices(prices_hourly)).T
df_minutely = pd.DataFrame(extract_prices(prices_minutely)).T


df_daily = df_daily.apply(lambda x: x.fillna(x.mean()), axis=1)
df_hourly = df_hourly.apply(lambda x: x.fillna(x.mean()), axis=1)
df_minutely = df_minutely.apply(lambda x: x.fillna(x.mean()), axis=1)


scaler = StandardScaler()
df_daily_scaled = pd.DataFrame(scaler.fit_transform(df_daily), index=df_daily.index)
df_hourly_scaled = pd.DataFrame(scaler.fit_transform(df_hourly), index=df_hourly.index)
df_minutely_scaled = pd.DataFrame(scaler.fit_transform(df_minutely), index=df_minutely.index)


num_clusters = 5
kmeans_daily = KMeans(n_clusters=num_clusters, random_state=42).fit(df_daily_scaled)
kmeans_hourly = KMeans(n_clusters=num_clusters, random_state=42).fit(df_hourly_scaled)
kmeans_minutely = KMeans(n_clusters=num_clusters, random_state=42).fit(df_minutely_scaled)

df_clusters = pd.DataFrame(index=df_daily.index)
df_clusters['Cluster_Daily'] = kmeans_daily.labels_
df_clusters['Cluster_Hourly'] = kmeans_hourly.labels_
df_clusters['Cluster_Minutely'] = kmeans_minutely.labels_


dtw_kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", verbose=False, random_state=42)
dtw_clusters = dtw_kmeans.fit_predict(df_daily_scaled)

df_clusters['Cluster_DTW'] = dtw_clusters


btc_cluster = df_clusters.loc['BTC', 'Cluster_Daily']
outliers = df_clusters[df_clusters['Cluster_Daily'] != btc_cluster]


print("Монеты в кластере BTC:")
print(df_clusters[df_clusters['Cluster_Daily'] == btc_cluster])

print("\nМонеты в других кластерах:")
print(outliers)
