import os
import pandas as pd
import functions as fn
import classes as cl


# --- Set the path to main directory of the project --- #

PROJECT_PATH = '/Users/stephanelhaut/Tail-Dep-Viz'
os.chdir(PROJECT_PATH)

# --- Load the data --- #

DATA_PATH = 'datasets/30_industry_losses_1950-2015.csv'
data=pd.read_csv(DATA_PATH)


# --- Compute the TPDM --- #

tpdm = fn.get_TPDM(data=data, 
                   estimator='pairwise',
                   p_upper=0.008)

#sns.heatmap(tpdm, annot=False, cmap='viridis')


# --- Perform the desired dimension reduction --- #

dim_red = cl.DimReducer(method="mds",n_components=2)
emb = dim_red.fit_transform(0.5-tpdm)
fn.plot_embedding(embedding=emb,
                  labels=data.columns,
                  method_name="mds")