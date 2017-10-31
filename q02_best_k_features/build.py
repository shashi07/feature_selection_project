# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df,k=20):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    selector = SelectPercentile(f_regression,k)
    selector.fit(X, y)
    t = df.iloc[:,selector.get_support(True)]
    fs = list(t)
    #feature_names = t.columns.values
    feature_names = list(X)
    ordered = [feature_names[i] for i in np.argsort(selector.scores_)[::-1]]
    feature_names_ks = []
    for i in ordered:
        if i in fs:
            feature_names_ks.append(i)
    return feature_names_ks
