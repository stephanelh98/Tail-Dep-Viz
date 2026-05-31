import numpy as np
import pandas as pd
from scipy.stats import rankdata
import itertools
import matplotlib.pyplot as plt

# --- Loading the data --- #

def load_data(path=None, sep=",", header=0, index_col=0, nrows=None, na_values=None):
    if not path.endswith(".csv"):
        raise ValueError(f"File should be in CSV format, got : {path}")
        
    if nrows is None and na_values is None:
        return pd.read_csv(filepath_or_buffer=path,
                    sep=sep,
                    header=header,
                    index_col=index_col)
    if nrows is not None and na_values is None:
        return pd.read_csv(filepath_or_buffer=path,
                    sep=sep,
                    header=header,
                    index_col=index_col,
                    nrows=nrows)
    if nrows is None and na_values is not None:
        return pd.read_csv(filepath_or_buffer=path,
                    sep=sep,
                    header=header,
                    index_col=index_col,
                    na_values=na_values)
    if nrows is not None and na_values is not None:
        return pd.read_csv(filepath_or_buffer=path,
                    sep=sep,
                    header=header,
                    index_col=index_col,
                    nrows=nrows,
                    na_values=na_values)

# --- Compute the TPDM between columns --- #

def get_TPDM(data=None, estimator='pairwise',p_upper=None):
    if data is None:
        raise ValueError("Please provide a non-empyty dataset")
    elif not isinstance(data, pd.DataFrame):
        raise TypeError(f"data should be a pandas.DataFrame, got {type(data)}")
    
    if p_upper is None:
        p_upper=0.05
    
    n_upper = int(p_upper * data.shape[0])
    
    # select numeric columns
    data_num = data.select_dtypes('number')
    n_col = data_num.shape[1]
    if n_col < 2:
        raise ValueError(f"Your dataset does contain only {n_col} numeric columns but should contain at least two")
    
    # transform to RV(2) 
    def Z_transform(x):
        n = len(x)
        return (n+1)/(n+1-rankdata(x))
    data_num_rv = data_num.apply(Z_transform)
    
    # compute the TPDM
    tpdm = np.eye(n_col) / 2
    
    if estimator == 'pairwise':
        def coef_pair(x,y):
            pair = pd.concat([x,y], axis = 1)
            radii = pair.apply(lambda x: np.linalg.norm(x,ord=2), 
                               axis="columns")
            angles = pair.div(radii, axis=0)
            angles_large = angles.loc[radii.sort_values().index[-n_upper:]]
            return sum(angles_large.prod(axis=1)) / n_upper
        
        for i in range(n_col):
            for j in range(i,n_col):
                if i == j:
                    pass
                else:
                    tpdm[i,j] = coef_pair(data_num_rv.iloc[:,i],
                                          data_num_rv.iloc[:,j])
                    tpdm[j,i] = tpdm[i,j]
        
    
    elif estimator == 'full':
        print("To be implemented. Sorry")
        return None
    

    def is_positive_definite(mat):
        eigenvalues = np.linalg.eigvals(mat)
        return bool(np.all(eigenvalues > 0))
    
    
    if is_positive_definite(tpdm):
        return pd.DataFrame(tpdm, columns=data_num.columns,index=data_num.columns)
    else:
        # Source - https://stackoverflow.com/a/10940283
        # Posted by segasai, modified by community. See post 'Timeline' for change history
        # Retrieved 2026-05-28, License - CC BY-SA 3.0

        def _getAplus(A):
            eigval, eigvec = np.linalg.eig(A)
            Q = np.matrix(eigvec)
            xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
            return Q*xdiag*Q.T
        
        def _getPs(A, W=None):
            W05 = np.matrix(W**.5)
            return  W05.I * _getAplus(W05 * A * W05) * W05.I
        
        def _getPu(A, W=None):
            Aret = np.array(A.copy())
            Aret[W > 0] = np.array(W)[W > 0]
            return np.matrix(Aret)
        
        def nearPD(A, nit=10):
            n = A.shape[0]
            W = np.identity(n) 
            # W is the matrix used for the norm (assumed to be Identity matrix here)
            # the algorithm should work for any diagonal W
            deltaS = 0
            Yk = A.copy()
            for k in range(nit):
                Rk = Yk - deltaS
                Xk = _getPs(Rk, W=W)
                deltaS = Xk - Rk
                Yk = _getPu(Xk, W=W)
            return Yk

        return pd.DataFrame(nearPD(tpdm), columns=data_num.columns,index=data_num.columns)
    
    

# --- Plot the dimension reduction --- #



def plot_embedding(embedding, labels=None, method_name="MDS", plot_type="pairwise", groups=None, group_col=None):
    """
    embedding   : array (n_samples, n_components) — sortie de DimReducer.fit_transform()
    labels      : liste de noms pour annoter les points (ex: data_filtered.columns)
                  si None, pas d'annotations
    method_name : str, pour les titres ("MDS", "TSNE", "Isomap")
    plot_type   : "pairwise", "univariate", ou "both"
    groups      : array (n_samples,) optionnel — valeurs de groupe pour colorer les points
    group_col   : nom de la variable de groupe (pour la légende)
    """
    n_components = embedding.shape[1]
    method_name = method_name.upper()

    # Palette de couleurs si groupes
    if groups is not None:
        unique_groups = np.unique(groups)
        palette = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        color_map = {g: palette[i] for i, g in enumerate(unique_groups)}
        colors = [color_map[g] for g in groups]
    else:
        colors = ["steelblue"] * embedding.shape[0]
        unique_groups = None

    def _scatter(ax, x, y, xi, yi):
        if unique_groups is not None:
            for g in unique_groups:
                mask = groups == g
                ax.scatter(x[mask], y[mask], alpha=0.7, label=str(g), color=color_map[g])
            ax.legend(title=group_col, fontsize=7)
        else:
            ax.scatter(x, y, alpha=0.7, color=colors)

        if labels is not None:
            for k, name in enumerate(labels):
                ax.annotate(name, (x[k], y[k]), fontsize=7, ha='right')

        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.set_xlabel(f"{method_name}{xi+1}")
        ax.set_ylabel(f"{method_name}{yi+1}")
        ax.set_title(f"{method_name}{xi+1} vs {method_name}{yi+1}")

    def _plot_pairwise():
        pairs = list(itertools.combinations(range(n_components), 2))
        n_cols = 3
        n_rows = int(np.ceil(len(pairs) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten()
        for idx, (i, j) in enumerate(pairs):
            _scatter(axes[idx], embedding[:, i], embedding[:, j], i, j)
        for idx in range(len(pairs), len(axes)):
            axes[idx].set_visible(False)
        plt.suptitle(f"{method_name} — Paires de composantes", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    def _plot_univariate():
        n_cols = min(n_components, 3)
        n_rows = int(np.ceil(n_components / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()
        for i in range(n_components):
            ax = axes[i]
            if unique_groups is not None:
                for g in unique_groups:
                    mask = groups == g
                    ax.hist(embedding[mask, i], alpha=0.6, label=str(g), color=color_map[g], bins=15)
                ax.legend(title=group_col, fontsize=7)
            else:
                ax.hist(embedding[:, i], bins=15, color="steelblue", alpha=0.7)
            ax.set_xlabel(f"{method_name}{i+1}")
            ax.set_title(f"Distribution {method_name}{i+1}")
        for idx in range(n_components, len(axes)):
            axes[idx].set_visible(False)
        plt.suptitle(f"{method_name} — Composantes univariées", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    if plot_type in ("pairwise", "both"):
        _plot_pairwise()
    if plot_type in ("univariate", "both"):
        _plot_univariate()