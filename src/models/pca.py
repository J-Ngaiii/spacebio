import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


class PCARidge:
    """Standardize genes, keep the top principal components, then fit ridge regression from them to every protein."""

    def __init__(self, n_components=5, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0), random_state=0):
        self.n_components = n_components
        self.alphas = alphas
        self.random_state = random_state

    def fit(self, X, Y):
        # PCA cannot find more directions than there are samples minus one.
        self.n_components_ = min(self.n_components, len(X) - 1)
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=self.n_components_, random_state=self.random_state).fit(self.scaler.transform(X))
        Z = self.pca.transform(self.scaler.transform(X))

        # Proteins measured in the same samples (e.g. the same TMT plex) share one fit; NaN targets are left out.
        # Proteins with fewer than 3 measured samples keep zero weights and predict 0, the training mean after z-scoring.
        self.coef_ = np.zeros((Y.shape[1], self.n_components_))
        self.intercept_ = np.zeros(Y.shape[1])
        self.alpha_ = np.full(Y.shape[1], np.nan)
        patterns, groups = np.unique(~np.isnan(Y).T, axis=0, return_inverse=True)
        for group, measured in enumerate(patterns):
            if measured.sum() < 3:
                continue
            columns = np.flatnonzero(groups.ravel() == group)
            # RidgeCV picks the penalty by leave-one-out within these training samples only.
            ridge = RidgeCV(alphas=self.alphas).fit(Z[measured], Y[np.ix_(measured, columns)])
            self.coef_[columns], self.intercept_[columns], self.alpha_[columns] = ridge.coef_, ridge.intercept_, ridge.alpha_
        return self

    def predict(self, X):
        return self.pca.transform(self.scaler.transform(X)) @ self.coef_.T + self.intercept_
