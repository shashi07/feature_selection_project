# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')
np.random.seed(9)

# Your solution code here
def select_from_model(df):
    model = RandomForestClassifier(random_state=9)
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    model.fit(X,y)
    m = SelectFromModel(model, prefit=True)
    return list(X.iloc[:,m.get_support()])
