# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here

def rf_rfe(df):
    model = RandomForestClassifier()
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    rfe = RFE(model, df.shape[1]/2)
    rfe = rfe.fit(X, y)

    t = df.iloc[:,rfe.support_]
    fs = list(t)
    feature_names = list(X)
    #ordered = [feature_names[i] for i in np.argsort(rfe.ranking_)[::-1]]
    #print fs
    #print ordered
    return fs
