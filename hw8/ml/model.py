import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix 
from imblearn.pipeline import Pipeline
from joblib import dump
import os

df = pd.read_csv("hw8/ml/archive/heart.csv")
# print(df.head())
# print(df.columns)
# print(df.shape)
# print(df.info())
# print(df.describe())
df =df.drop_duplicates()
# print(df.duplicated().sum())
# print(Counter(df['target']))  


X = df.drop('target', axis=1) 
y = df['target']   

print("Before SMOTE:", Counter(y))
smote = SMOTE(sampling_strategy={0: 500, 1: 500}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("After SMOTE:", Counter(y_resampled))

X_res_df = pd.DataFrame(X_resampled, columns=X.columns)
y_res_series = pd.Series(y_resampled, name='target')
df_resampled = pd.concat([X_res_df, y_res_series], axis=1)
df_resampled.head()
print( df_resampled['target'].value_counts())
df = df.drop_duplicates()

cat_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
for col in cat_columns:
    sns.countplot(data=df, x=col,palette = 'Purples')
    plt.title(f'Distribution of {col}')
    plt.show()

num_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for col in num_columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns
  
    sns.histplot(df[col], kde=True, color='purple', ax=axes[0])
    axes[0].set_title(f'Histogram of {col}')

    sns.boxplot(x=df[col], color='purple', ax=axes[1])
    axes[1].set_title(f'Boxplot of {col}')
    
    plt.tight_layout()
    plt.show()


cols = ['oldpeak', 'trestbps', 'chol']
def drop_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df
df_no_outliers = drop_outliers_iqr(df, cols)

print("Original rows:", len(df))
print("Rows after dropping outliers:", len(df_no_outliers))

correlations = df_no_outliers.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlations, annot=True, cmap='Purples')
plt.title("Correlation Heatmap")
plt.show()

scaler = StandardScaler()
df_no_outliers[num_columns] = scaler.fit_transform(df_no_outliers[num_columns])
df_no_outliers.head()

X = df_no_outliers.drop('target', axis=1)
y = df_no_outliers['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X train shape: {X_train.shape}")
print(f"y train shape: {y_train.shape}")
print(f"X test shape: {X_test.shape}")
print(f"y test shape: {y_test.shape}")


pipeline = Pipeline([
    ('smote', SMOTE()),
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"))])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)
print(f"\nTraining Accuracy: {train_score:.4f}")
print(f"Testing Accuracy:  {test_score:.4f}\n")

# model = XGBClassifier()
# y_pred = model.predict(X_test)
# train_score = model.score(X_train, y_train)
# test_score = model.score(X_test, y_test)

conf_matrix = confusion_matrix(y_test, y_pred)
conf_df = pd.DataFrame(conf_matrix, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
plt.figure(figsize=(6, 4))
sns.heatmap(conf_df, annot=True,cmap="Purples",fmt="d")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

print("Classification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# top_features = feat_importances.nlargest(10)
# colors = sns.color_palette("Purples", n_colors=10)
# top_features.plot(kind='barh', color=colors)
# plt.title("Top 10 Feature Importances")
# plt.gca().invert_yaxis()  
# plt.tight_layout()
# plt.show()

# Save the model to a file
BASE_DIR = 'hw8/ml'
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
dump(pipeline, MODEL_PATH)
print("Model saved to 'ml/model.pkl'")  # Confirmation message for saving the