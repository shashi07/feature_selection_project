# Default imports
import pandas as pd
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap

data = pd.read_csv('data/house_prices_multivariate.csv')


# Write your solution here:
def plot_corr(df,size=11):
    corr = data.corr()
    fig, ax = subplots(figsize=(size,size))
    im = ax.imshow(corr, cmap="YlOrRd")
