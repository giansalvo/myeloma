# https://towardsdatascience.com/introduction-to-survival-analysis-the-kaplan-meier-estimator-94ec5812a97a
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines import KaplanMeierFitter
from lifelines.statistics import (logrank_test,
                                  pairwise_logrank_test,
                                  multivariate_logrank_test,
                                  survival_difference_at_fixed_point_in_time_test)



df = pd.read_csv('./telco_customer_churn.csv')
df['churn'] = [1 if x == 'Yes' else 0 for x in df['Churn']]

T = df['tenure']
E = df['churn']

ax = plt.subplot(111)

kmf = KaplanMeierFitter()

for payment_method in df['PaymentMethod'].unique():
    flag = df['PaymentMethod'] == payment_method

    kmf.fit(T[flag], event_observed=E[flag], label=payment_method)
    kmf.plot_survival_function(ax=ax)

plt.title("Survival curves by payment methods")
plt.show()