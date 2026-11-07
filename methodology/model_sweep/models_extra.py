"""Course-notes models missed in stage 1: feature mapping and k-means-based classifiers."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import PolynomialFeatures
from models import grid

N_STRUCT = 31  # first 31 design-matrix columns are structured (26 parsed + 5 missing flags)

class PolyMap(BaseEstimator, ClassifierMixin):
    """Degree-2 feature mapping of the structured columns + raw text columns, then a linear model."""
    def __init__(self, C=0.3, base="softmax"):
        self.C, self.base = C, base
    def _map(self, X):
        return np.hstack([self.pf_.transform(X[:, :N_STRUCT]), X[:, N_STRUCT:]])
    def fit(self, X, y):
        self.pf_ = PolynomialFeatures(2, include_bias=False).fit(X[:, :N_STRUCT])
        Z = self._map(X)
        self.m_ = (LogisticRegression(C=self.C, max_iter=5000) if self.base == "softmax"
                   else RidgeClassifier(alpha=self.C)).fit(Z, y)
        self.classes_ = self.m_.classes_; return self
    def decision_function(self, X): return self.m_.decision_function(self._map(X))
    def predict(self, X): return self.m_.predict(self._map(X))

class KMeansPerClass(BaseEstimator, ClassifierMixin):
    """k-means with K clusters inside each class; predict the class of the nearest centroid."""
    def __init__(self, K=3): self.K = K
    def fit(self, X, y):
        self.classes_ = np.unique(y); C, L = [], []
        for i, c in enumerate(self.classes_):
            km = KMeans(self.K, n_init=10, random_state=0).fit(X[y == c])
            C.append(km.cluster_centers_); L += [i] * self.K
        self.C_, self.L_ = np.vstack(C), np.array(L); return self
    def decision_function(self, X):
        d = ((X[:, None, :] - self.C_[None]) ** 2).sum(-1)
        return np.column_stack([-d[:, self.L_ == i].min(1) for i in range(len(self.classes_))])
    def predict(self, X): return self.classes_[self.decision_function(X).argmax(1)]

class KMeansFeatures(BaseEstimator, ClassifierMixin):
    """Unsupervised k-means on all rows; distances to centroids appended as features for softmax."""
    def __init__(self, K=20, C=0.3): self.K, self.C = K, C
    def fit(self, X, y):
        self.km_ = KMeans(self.K, n_init=10, random_state=0).fit(X)
        self.m_ = LogisticRegression(C=self.C, max_iter=5000).fit(self._z(X), y)
        self.classes_ = self.m_.classes_; return self
    def _z(self, X):
        D = self.km_.transform(X); return np.hstack([X, (D - D.mean(1, keepdims=True)) / (D.std(1, keepdims=True) + 1e-9)])
    def predict_proba(self, X): return self.m_.predict_proba(self._z(X))
    def predict(self, X): return self.m_.predict(self._z(X))

EXTRA = {
    "PolyFeatures+Softmax": (lambda p: PolyMap(p["C"], "softmax"), grid(C=[0.03, 0.1, 0.3, 1])),
    "PolyFeatures+LinReg": (lambda p: PolyMap(p["a"], "ridge"), grid(a=[10, 30, 100, 300])),
    "kMeans-per-class(nearest centroid)": (lambda p: KMeansPerClass(p["K"]), grid(K=[1, 2, 3, 5, 8])),
    "kMeans-features+Softmax": (lambda p: KMeansFeatures(p["K"], p["C"]), grid(K=[10, 30], C=[0.1, 0.3])),
}
