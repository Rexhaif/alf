import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from modAL.models import ActiveLearner


def random_forest_max_std(regressor: ActiveLearner, X: np.ndarray, batch_size: int = 10):
    predictions = [tree.predict(X) for tree in regressor.estimator.estimators_]
    predictions = np.vstack(predictions).T
    predictions_std = np.std(predictions, axis=-1)
    idxs = np.argsort(predictions_std)[::-1]
    idxs = idxs[:batch_size]
    return idxs, X[idxs]


def gaussian_process_max_std(regressor: ActiveLearner, X: np.ndarray, batch_size: int = 10):
    _, std = regressor.predict(X, return_std=True)
    idxs = np.argsort(std)[::-1]
    idxs = idxs[:batch_size]
    return idxs, X[idxs]


MODELS = {
    'random_forest': (RandomForestRegressor(n_jobs=-1), random_forest_max_std),
    'gaussian_process': (
        GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
            + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))),
        gaussian_process_max_std
    )
}
