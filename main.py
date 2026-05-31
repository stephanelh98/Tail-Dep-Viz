import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import sklearn.manifold
import itertools

import functions as fn
import classes as cl


# --- Set the path to main directory of the project --- #

PROJECT_PATH = '/Users/stephanelhaut/Tail-Dep-Viz'
os.chdir(PROJECT_PATH)


### SOME EXAMPLES

# --- 1. Load the financial data and preprocess --- #

DATA_PATH = 'datasets/30_Industry_Portfolios_Daily.csv'

data = fn.load_data(path=DATA_PATH,
                 header=5,
                 nrows=26211+1,
                 na_values=[-99,-999])

data.info()
data

# we take the same data as Cooley 2019 or Wan 2020 -> 1950 - 2015
# we must have n = 16,694 rows. We also consider loss -> * (-1)

years = data.index.values // 10000
mask = (years >= 1950) & (years <= 2015)
data_filtered = (-1) * data[mask]

# --- Compute the TPDM --- #

tpdm = fn.get_TPDM(data=data_filtered, 
                   estimator='pairwise',
                   p_upper=0.008)

#tpdm.iloc[0:5,0:5]
#sns.heatmap(tpdm, annot=False, cmap='viridis')


# --- Perform the desired dimension reduction --- #

dim_red = cl.DimReducer(method="mds",n_components=2)
emb = dim_red.fit_transform(0.5-tpdm)
fn.plot_embedding(embedding=emb,
                  labels=data_filtered.columns,
                  method_name="mds")






