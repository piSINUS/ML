import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier
from sklearn.multioutput import MultiOutputClassifier



df = pd.read_csv("hw9/jigsaw-toxic-comment-classification-challenge/train.csv")

X = df["comment_text"].fillna(" ")
y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Базовая модель
# baseline = Pipeline([
#     ("tfidf", TfidfVectorizer(max_features=50000, stop_words="english")),
#     ("clf", OneVsRestClassifier(LogisticRegression(max_iter=200)))
# ])

# baseline.fit(X_train, y_train)
# y_pred = baseline.predict(X_test)

# print(classification_report(y_test, y_pred, target_names=y.columns))

# Уменьшаем размер выборки для ускорения работы AutoML
df_small = df.sample(n = 1000, random_state=42)  
X_small = df_small["comment_text"].fillna(" ")
y_small = df_small[y.columns]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42
)

# TPOT с плотным конфигом
# tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
# X_train_tfidf = tfidf.fit_transform(X_train_s)
# X_test_tfidf = tfidf.transform(X_test_s)

# tpot = TPOTClassifier(
#     generations=2,
#     population_size=5,
#     verbosity=2,
#     max_time_mins= 3,
#     random_state=42,
#     config_dict="TPOT sparse"
# )

# automl = MultiOutputClassifier(tpot)
# automl.fit(X_train_tfidf, y_train_s)

# y_pred_auto = automl.predict(X_test_tfidf)
# print(classification_report(y_test_s, y_pred_auto, target_names=y.columns))

# H2O с плотным конфигом
tfidf = TfidfVectorizer(max_features=500)
X_train_vec = tfidf.fit_transform(X_train_s).toarray()
X_test_vec = tfidf.transform(X_test_s).toarray()

train_df = pd.DataFrame(X_train_vec)
train_df[y_train_s.columns] = y_train_s.reset_index(drop=True)

test_df = pd.DataFrame(X_test_vec)
test_df[y_train_s.columns] = y_test_s.reset_index(drop=True)


h2o.init(max_mem_size="2G")
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

x_cols = list(map(str, range(X_train_vec.shape[1])))  # имена признаков

models = {}
for target in y_train_s.columns:
    aml = H2OAutoML(max_runtime_secs=60, seed=42, max_models=5)
    aml.train(x=x_cols, y=target, training_frame=train_h2o)
    models[target] = aml.leader

preds = []
for target in y_train_s.columns:
    pred = models[target].predict(test_h2o).as_data_frame()["predict"]
    preds.append(pred)

y_pred_auto = pd.DataFrame(preds).T
y_pred_auto.columns = y_train_s.columns
y_pred_auto = (y_pred_auto > 0.5).astype(int)

print("Micro F1:", f1_score(y_test_s, y_pred_auto, average="micro"))
print("Macro F1:", f1_score(y_test_s, y_pred_auto, average="macro"))