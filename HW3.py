import pandas as pd
import numpy as np
import random
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tqdm import tqdm
import datetime
from sklearn.metrics import precision_score, recall_score


interactions = pd.read_csv("data_kion/interactions_df.csv")
users = pd.read_csv("data_kion/users.csv")
items = pd.read_csv("data_kion/items.csv")



random_users = random.sample(interactions['user_id'].unique().tolist(), 2000)
interactions = interactions[interactions['user_id'].isin(random_users)]


interactions['last_watch_dt'] = pd.to_datetime(interactions['last_watch_dt'])


test_threshold = interactions['last_watch_dt'].max() - pd.Timedelta(days=7)
train = interactions[interactions['last_watch_dt'] < test_threshold]
test = interactions[interactions['last_watch_dt'] >= test_threshold]

train = train.dropna(subset=['user_id', 'item_id', 'watched_pct'])
train = train[train['watched_pct'] > 0]

train['watched_pct'] = np.log1p(train['watched_pct'])

# Маппинг пользователей и айтемов
user_ids = train['user_id'].unique()
item_ids = train['item_id'].unique()
user_id_map = {id_: idx for idx, id_ in enumerate(user_ids)}
item_id_map = {id_: idx for idx, id_ in enumerate(item_ids)}

train['user_idx'] = train['user_id'].map(user_id_map)
train['item_idx'] = train['item_id'].map(item_id_map)

# Статистики user/item
user_stats = train.groupby('user_id')['watched_pct'].agg(['mean', 'count']).rename(columns={'mean': 'user_mean', 'count': 'user_count'})
item_stats = train.groupby('item_id')['watched_pct'].agg(['mean', 'count']).rename(columns={'mean': 'item_mean', 'count': 'item_count'})

train = train.merge(user_stats, on='user_id', how='left')
train = train.merge(item_stats, on='item_id', how='left')

# Построение sparse-матрицы
rows, cols = train['user_idx'].values, train['item_idx'].values
data = train['watched_pct'].values
matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))

# Проверка на NaN и пустые строки
assert not np.isnan(matrix.data).any(), "NaN in matrix"
assert matrix.getnnz(axis=1).min() > 0, "User with 0 interactions"
assert matrix.getnnz(axis=0).min() > 0, "Item with 0 interactions"

# Обучение ALS
als = AlternatingLeastSquares(factors=32, regularization=0.1, iterations=10)
als.fit(matrix)

# Подсчёт bpr_score
bpr_score = als.user_factors @ als.item_factors.T


def get_recommendations(model, user_idx, N=10):
    return model.recommend(user_idx, matrix[user_idx], N=N)[0]


def precision_at_k(model, test_df, train_df, k=20):
    hits = []
    user_ids_test = test_df['user_id'].unique()

    for user in tqdm(user_ids_test):
        if user not in user_id_map:
            continue

        uid = user_id_map[user]
        recs = get_recommendations(model, uid, N=k)
        test_items = test_df[test_df['user_id'] == user]['item_id'].map(item_id_map).dropna().astype(int).values
        hit_count = len(set(recs).intersection(test_items))
        hits.append(hit_count / k)

    return np.mean(hits)

prec = precision_at_k(als, test, train, k=20)
print(f"Precision@20: {prec:.4f}")
