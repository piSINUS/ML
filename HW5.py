import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az


N = 10      
k = 7       


with pm.Model() as model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    y = pm.Binomial('y', n=N, p=theta, observed=k)
    trace = pm.sample(2000, return_inferencedata=True, random_seed=42)
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['theta', 'y'], random_seed=42)


az.plot_trace(trace, var_names=['theta'])
plt.suptitle('Апостериорное распределение θ', fontsize=16)
plt.tight_layout()
plt.show()

az.plot_posterior(trace, var_names=['theta'], hdi_prob=0.95)
plt.title('Постериорная плотность θ с 95% HDI')
plt.show()

future_theta_samples = trace.posterior['theta'].values.flatten()
future_y_samples = np.random.binomial(n=20, p=future_theta_samples)


sns.histplot(future_y_samples, bins=np.arange(21)-0.5, stat='density', kde=False, color='skyblue')
plt.title('Posterior Predictive Distribution\n(Ожидаемое число орлов в 20 бросках)')
plt.xlabel('Число орлов')
plt.ylabel('Плотность')
plt.grid(True)
plt.show()


sns.histplot(posterior_predictive['y'], bins=np.arange(N+2)-0.5, kde=False, color='lightgreen', edgecolor='black')
plt.title('Posterior Predictive по наблюдаемым данным\n(число орлов из 10 подбрасываний)')
plt.xlabel('Число орлов (k)')
plt.ylabel('Частота')
plt.grid(True)
plt.show()