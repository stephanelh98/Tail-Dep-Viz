from sklearn.manifold import MDS, Isomap, TSNE

class DimReducer:

    _DEFAULTS = {
        "tsne":   {"metric": 'precomputed',
                   "perplexity": 5.0,
                   "early_exaggeration": 12.0,
                   "random_state": 42,},
        "mds":    {"metric": 'precomputed',
                   "metric_mds": False, 
                   "n_init": 4,
                   "init": 'random',
                   "random_state": 42},
        "isomap": {"metric": 'precomputed',
                   "n_neighbors": 5},
    }
    _CLASSES = {"tsne": TSNE, "mds": MDS, "isomap": Isomap}

    def __init__(self, method="mds", n_components=2, method_kwargs=None):
        self.method = method.lower()
        self.n_components = n_components
        self.method_kwargs = method_kwargs if method_kwargs is not None else {}

        if self.method not in self._CLASSES:
            raise ValueError(f"Unknown method: {self.method}. Available: {list(self._CLASSES)}")

    def fit_transform(self, X):
        cls = self._CLASSES[self.method]

        params = {
            "n_components": self.n_components,
            **self._DEFAULTS[self.method],
            **self.method_kwargs,
        }

        self.embedding_ = cls(**params).fit_transform(X)
        return self.embedding_

    def __repr__(self):
        return f"DimReducer(method={self.method!r}, n_components={self.n_components}, method_kwargs={self.method_kwargs})"