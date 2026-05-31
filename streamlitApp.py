import streamlit as st
import pandas as pd
import io
import functions as fn
import classes as cl
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("# Tail visualisation based on Tail Pairwise Dependence Matrices")


# Dataset choice
st.markdown("## Dataset choice")

DATASETS = {
    "30 industries daily losses": "datasets/30_industry_losses_1950-2015.csv",
}
choix = st.selectbox("Pick a dataset", ["-- Pick --"] + list(DATASETS.keys()))

if choix == "-- Pick --":
    st.stop()
df = pd.read_csv(DATASETS[choix])

## some viz

col1, col2 = st.columns(2)

with col1:
    st.dataframe(df)

with col2:
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())


# TPDM fit and viz
st.markdown('## Fit the TPDM')

estimator = st.radio("Which estimator?", 
                     ["pairwise", "full"], 
                     horizontal=True)
n = len(df)
p_upper = st.slider(
    "Proportion of upper observation used (ranked by L2-norm on Z scale)",
    min_value=round(1/n, 6),
    max_value=1.0,
    value=0.008,
    step=round(1/n, 6),
    format="%.6f"
)

if st.button("Compute TPDM"):
    with st.spinner("Computing TPDM..."):
        st.session_state.tpdm = fn.get_TPDM(data=df, estimator=estimator, p_upper=p_upper)
    st.success("TPDM computed!")

if "tpdm" not in st.session_state:
    st.stop()

tpdm = st.session_state.tpdm

col1, col2 = st.columns(2)

with col1:
    st.dataframe(tpdm)
with col2:
    fig, ax = plt.subplots()
    sns.heatmap(tpdm, annot=False, cmap='viridis', ax=ax)
    st.pyplot(fig)


# perform dimesion reduction
st.markdown('## Visualisation of the tail')

# Choix de la méthode
method = st.radio("Method", ["mds", "tsne", "isomap"], horizontal=True)

method_kwargs = {}
if method == "tsne":
    method_kwargs["perplexity"] = st.slider("Perplexity", 2.0, 50.0, 5.0, step=0.5)
    method_kwargs["early_exaggeration"] = st.slider("Early exaggeration", 1.0, 50.0, 12.0, step=0.5)
elif method == "mds":
    method_kwargs["metric_mds"] = st.checkbox("Metric MDS", value=False)
    method_kwargs["n_init"] = st.slider("Number of initializations", 1, 10, 4)
elif method == "isomap":
    method_kwargs["n_neighbors"] = st.slider("Number of neighbors", 2, 20, 5)

plot_type = st.radio("Plot type", ["pairwise", "univariate", "both"], horizontal=True)

if st.button("Run dimensionality reduction"):
    with st.spinner("Running dimensionality reduction..."):
        dim_red = cl.DimReducer(method=method, n_components=2, method_kwargs=method_kwargs)
        emb = dim_red.fit_transform(0.5 - tpdm)

    fn.plot_embedding(embedding=emb, labels=df.columns, method_name=method, plot_type=plot_type)
    st.pyplot(plt.gcf())