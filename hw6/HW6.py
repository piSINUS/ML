import pymc as pm
import numpy as np
import arviz  as az
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv("hw6/train.csv")
test = pd.read_csv("hw6/test .csv")

# print(train.columns)

median_price = train['SalePrice'].median()
train["target"] = (train['SalePrice'] > median_price).astype(int)

features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "1stFlrSF"]
X = train[features].fillna(0).values
y = train["target"].values
X_test = test[features].fillna(0).values

with pm.Model() as logistic_model:
    w = pm.Normal("w", mu=0, sigma=5, shape=X.shape[1])
    b = pm.Normal("b",mu=0, sigma=5)
    logits = pm.math.dot(X,w) + b
    p = pm.Deterministic("p", pm.math.sigmoid(logits))
    y_obs = pm.Bernoulli("y_obs", p=p, observed = y)
    trace = pm.sample(1000, tune=1000, return_inferencedata=True, target_accept = 0.9)

w_samples = trace.posterior["w"].stack(draws = ("chain","draw")).values.T
b_samples = trace.posterior["b"].stack(draws = ("chain", "draw")).values.flatten()


logits_test_samples = np.dot(X_test, w_samples.T) + b_samples
p_test_samples = 1 / (1 + np.exp(-logits_test_samples))
p_test_mean = p_test_samples.mean(axis=1)
p_test_hdi = az.hdi(p_test_samples.T, hdi_prob=0.95)

y_test_pred = (p_test_mean > 0.5).astype(int)

w_mean = trace.posterior["w"].mean(dim=("chain", "draw")).values
b_mean = trace.posterior["b"].mean(dim=("chain", "draw")).values
print(az.summary(trace, var_names=["w", "b"], hdi_prob=0.95))
print("Mean weights:", w_mean)
print("Mean bias:", b_mean)

# feature_contribs = X_test * w_mean

# interpretation_df = pd.DataFrame(feature_contribs, columns=[f"{f} (effect)" for f in features])
# interpretation_df.insert(0, "Id", test["Id"])
# interpretation_df["ProbExpensive"] = p_test_mean
# interpretation_df["HDI_lower"] = p_test_hdi[:, 0]
# interpretation_df["HDI_upper"] = p_test_hdi[:, 1]
# interpretation_df["IsExpensive"] = y_test_pred

# interpretation_df.to_csv("interpretable_results.csv", index=False)

