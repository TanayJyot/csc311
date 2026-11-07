"""Every CSC311 model family that can do 3-class classification, with search grids."""
import itertools
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin


class GMMClassifier(BaseEstimator, ClassifierMixin):
    """Generative classifier: one Gaussian mixture (fit with EM) per class on PCA features."""
    def __init__(self, n_components=2, n_pca=20, cov="diag"):
        self.n_components, self.n_pca, self.cov = n_components, n_pca, cov
    def fit(self, X, y):
        self.pca_ = PCA(min(self.n_pca, X.shape[1]), random_state=0).fit(X)
        Z = self.pca_.transform(X)
        self.classes_ = np.unique(y)
        self.g_ = [GaussianMixture(self.n_components, covariance_type=self.cov, reg_covar=1e-3,
                                   random_state=0).fit(Z[y == c]) for c in self.classes_]
        self.lp_ = np.log([np.mean(y == c) for c in self.classes_])
        return self
    def predict_proba(self, X):
        Z = self.pca_.transform(X)
        L = np.column_stack([g.score_samples(Z) for g in self.g_]) + self.lp_
        L -= L.max(1, keepdims=True); P = np.exp(L)
        return P / P.sum(1, keepdims=True)
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*kw.values())]

nonneg = lambda: FunctionTransformer(lambda X: X - X.min(0))  # for MultinomialNB

FAMILIES = {
    "kNN": (lambda p: KNeighborsClassifier(p["k"], weights=p["w"], metric=p["m"]),
            grid(k=[1, 3, 5, 9, 15, 25, 41, 61], w=["uniform", "distance"], m=["euclidean", "cosine"])),
    "DecisionTree": (lambda p: DecisionTreeClassifier(criterion=p["c"], max_depth=p["d"],
                                                       min_samples_leaf=p["l"], random_state=0),
            grid(c=["gini", "entropy"], d=[3, 5, 8, 12, None], l=[1, 5, 10, 20])),
    "BaggedTrees": (lambda p: BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=p["l"]),
                                                 n_estimators=p["n"], random_state=0),
            grid(n=[100, 300], l=[1, 3, 5])),
    "RandomForest": (lambda p: RandomForestClassifier(p["n"], max_features=p["f"], max_depth=p["d"],
                                                       min_samples_leaf=p["l"], random_state=0),
            grid(n=[300], f=["sqrt", 0.1, 0.3], d=[None, 10], l=[1, 3])),
    "AdaBoost": (lambda p: AdaBoostClassifier(DecisionTreeClassifier(max_depth=p["d"]),
                                               n_estimators=p["n"], learning_rate=p["lr"], random_state=0),
            grid(d=[1, 2, 3], n=[100, 300], lr=[0.1, 0.5, 1.0])),
    "GradientBoosting": (lambda p: GradientBoostingClassifier(n_estimators=p["n"], learning_rate=p["lr"],
                                                               max_depth=p["d"], subsample=0.8, random_state=0),
            grid(n=[100, 300], lr=[0.03, 0.1], d=[2, 3])),
    "LeastSquares(Ridge)": (lambda p: RidgeClassifier(alpha=p["a"]),
            grid(a=[0.1, 1, 10, 30, 100, 300])),
    "SoftmaxRegression": (lambda p: LogisticRegression(C=p["C"], penalty=p["pen"],
                                                        solver="saga" if p["pen"] == "l1" else "lbfgs",
                                                        max_iter=5000),
            grid(C=[0.01, 0.03, 0.1, 0.3, 1, 3, 10], pen=["l2", "l1"])),
    "Perceptron": (lambda p: Perceptron(alpha=p["a"], penalty=p["pen"], max_iter=2000, random_state=0),
            grid(a=[1e-4, 1e-3, 1e-2], pen=[None, "l2"])),
    "LinearSVM": (lambda p: LinearSVC(C=p["C"], max_iter=20000),
            grid(C=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1])),
    "KernelSVM(RBF)": (lambda p: SVC(C=p["C"], gamma=p["g"], kernel="rbf"),
            grid(C=[0.3, 1, 3, 10, 30], g=["scale", 0.001, 0.003, 0.01])),
    "KernelSVM(poly)": (lambda p: SVC(C=p["C"], kernel="poly", degree=p["deg"], gamma="scale"),
            grid(C=[0.1, 1, 10], deg=[2, 3])),
    "BernoulliNB": (lambda p: BernoulliNB(alpha=p["a"], binarize=0.0),
            grid(a=[0.1, 0.3, 1, 3])),
    "GaussianNB": (lambda p: GaussianNB(var_smoothing=p["v"]),
            grid(v=[1e-9, 1e-3, 1e-2, 1e-1])),
    "MultinomialNB": (lambda p: make_pipeline(MinMaxScaler(), MultinomialNB(alpha=p["a"])),
            grid(a=[0.03, 0.1, 0.3, 1, 3])),
    "LDA": (lambda p: LinearDiscriminantAnalysis(solver="lsqr", shrinkage=p["s"]),
            grid(s=["auto", 0.1, 0.3, 0.5, 0.8])),
    "QDA(PCA)": (lambda p: make_pipeline(PCA(p["k"], random_state=0),
                                          QuadraticDiscriminantAnalysis(reg_param=p["r"])),
            grid(k=[5, 10, 20, 40], r=[0.1, 0.5, 0.9])),
    "GMM-classifier(EM)": (lambda p: GMMClassifier(p["c"], p["k"]),
            grid(c=[1, 2, 3], k=[10, 20, 40])),
    "PCA+Softmax": (lambda p: make_pipeline(PCA(p["k"], random_state=0), LogisticRegression(C=p["C"], max_iter=5000)),
            grid(k=[10, 20, 40, 80], C=[0.1, 1])),
    "PCA+kNN": (lambda p: make_pipeline(PCA(p["k"], random_state=0), KNeighborsClassifier(p["n"], weights="distance")),
            grid(k=[5, 10, 20, 40], n=[9, 25])),
    "MLP": (lambda p: MLPClassifier(p["h"], activation=p["act"], solver=p["s"], alpha=p["a"],
                                    learning_rate_init=0.05 if p["s"] == "sgd" else 1e-3,
                                    batch_size=32, max_iter=400, early_stopping=True, random_state=0),
            grid(h=[(16,), (64,), (64, 32)], act=["relu", "tanh"], s=["sgd", "adam"], a=[1e-3, 1e-1, 1.0])),
}
