import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.decomposition import PCA


df = pd.read_csv("OnlineRetail.csv")

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df = df.dropna(subset=["CustomerID"])
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["Date"] = df["InvoiceDate"].dt.date


unique_customers = df["CustomerID"].unique()
selected_customers = np.random.choice(unique_customers, size=1000, replace=False)
df = df[df["CustomerID"].isin(selected_customers)]


customer_daily = df.groupby(["CustomerID", "Date"]).agg({
    "TotalPrice": "sum",
    "InvoiceNo": "nunique"
}).reset_index()


tsfresh_df = customer_daily.rename(columns={"CustomerID": "id", "Date": "time", "TotalPrice": "value"})
features = extract_features(tsfresh_df, column_id="id", column_sort="time", column_value="value", n_jobs=4)


features = features.dropna(axis=1) 
features = features.loc[:, features.nunique() > 1]  
features = features.loc[:, features.var() > 0]  

impute(features)


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

time_series_data = to_time_series_dataset(features_scaled)

# def dtw_elbow_method(data, max_clusters=6):
#     inertia = []
#     cluster_range = range(2, max_clusters + 1)

#     for k in cluster_range:
#         model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42, n_init=1)
#         model.fit(data)
#         inertia.append(model.inertia_)

#     plt.figure(figsize=(8, 5))
#     plt.plot(cluster_range, inertia, marker="o", linestyle="--")
#     plt.xlabel("Количество кластеров (k)")
#     plt.ylabel("Инерция (DTW)")
#     plt.title("Метод локтя (DTW)")
#     plt.show()

# dtw_elbow_method(time_series_data)

optimal_clusters = 4  
model_dtw = TimeSeriesKMeans(n_clusters=optimal_clusters, metric="dtw", random_state=42, n_init=1)
clusters_dtw = model_dtw.fit_predict(time_series_data)


features_scaled_df = pd.DataFrame(features_scaled, index=features.index)
features_scaled_df["Cluster"] = clusters_dtw

cluster_summary = features_scaled_df.groupby("Cluster").mean()
cluster_summary["CustomerCount"] = features_scaled_df["Cluster"].value_counts()

print("Характеристики каждого кластера:")
print(cluster_summary)


# plt.figure(figsize=(12, 6))
# sns.barplot(x=cluster_summary.index, y=cluster_summary["CustomerCount"], palette="viridis")
# plt.xlabel("Кластер")
# plt.ylabel("Количество клиентов")
# plt.title("Распределение клиентов по кластерам (DTW)")
# plt.show()

pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=clusters_dtw, palette="viridis", alpha=0.7)
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.title("Визуализация кластеров с PCA")
plt.legend(title="Кластер")
plt.show()