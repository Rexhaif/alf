import logging

import pandas as pd
import numpy as np

from modAL.models import ActiveLearner
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from rich.logging import RichHandler

from typing import *

from .parameter import ParameterSpace

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("surrogate")


class SurrogateModel:

    def __init__(
            self, estimator: BaseEstimator, query_strategy: Callable,
            parameter_space: ParameterSpace, seed_size: int = 20,
            n_exploration_iterations: int = 10, n_runs_per_iter: int = 10,
            exploration_multiplier: int = 10, exploitation_eval_size: int = 10,
            exploitation_multiplier: int = 10, verbose: int = 0

    ):
        """
        Constructs new optimizer backend
        :param estimator: Base extimator for surrogate model, could be
        RandomForestRegressor or GaussianProcessRegressor
        :param query_strategy: function, which take regressor and unlabeled
        set of parameters and selects most informative
        :param parameter_space: instance of ParameterSpace
        :param seed_size: number of parameters to be scored for seeding stage
        :param n_exploration_iterations: number of iterations for exploration stage,
        i.e training surrogate model
        :param n_runs_per_iter: number of parameter evaluations per iteration
        :param exploration_multiplier: multiplier for sampling parameters for exploration
        :param exploitation_eval_size: number of parameter evaluations for exploitation stage
        :param exploitation_multiplier: multiplier for parameter sampling for exploitation
        """
        self.surrogate = ActiveLearner(estimator, query_strategy)
        self.parameter_space = parameter_space

        self.seed_size = seed_size
        self.n_exploration_iterations = n_exploration_iterations
        self.n_runs_per_iter = n_runs_per_iter
        self.exploration_multiplier = exploration_multiplier
        self.exploitation_eval_size = exploitation_eval_size
        self.exploitation_multiplier = exploitation_multiplier

        self.__history_parameters = []
        self.__history_scores = []

        self.verbose = verbose
        log.setLevel(verbose)

    def __sample_batch(self, batch_size: int = 10):
        return pd.concat([x for x in self.parameter_space.sample(batch_size)])

    def __update_history(self, params: pd.DataFrame, scores: np.ndarray):
        self.__history_parameters.append(params)
        self.__history_scores += list(scores)

    def seeding(self, objective_evaluator):
        """
        Runs seed stage of optimizer
        :param objective_evaluator: function which takes sequence of parameter dicts
         and returns array of scores
        :return: NoReturn
        """
        params = self.__sample_batch(self.seed_size)
        scores = objective_evaluator(params.to_dict(orient='records'))

        self.__update_history(params, scores)

        self.surrogate.fit(X=params.values, y=scores)

        if self.verbose > 0:
            mae = self.score_on_history()
            log.info(f"Seed stage MAE = {mae:.4f}")

    def exploration(self, objective_evaluator):
        """
        Runs exploration stage of optimizer - training optimizer to pred
        :param objective_evaluator: function which takes sequence of parameter dicts
         and returns array of scores
        :return: NoReturn
        """
        for i in range(self.n_exploration_iterations):
            space = self.__sample_batch(self.n_runs_per_iter * self.exploration_multiplier)
            idxs, _ = self.surrogate.query(space.values, batch_size=self.n_runs_per_iter)
            params = space.iloc[idxs]

            scores = objective_evaluator(params.to_dict(orient='records'))

            self.__update_history(params, scores)
            self.surrogate.teach(params.values, scores)

            if self.verbose:
                mae = self.score_on_history()
                log.info(f"Exploration iter {i} MAE = {mae:.4f}")

    def score_on_history(self):
        """
        Scores model on existing hostorical evaluations
        :return: mean absolute error score
        """
        X = pd.concat(self.__history_parameters)
        y_true = np.array(self.__history_scores)
        y_pred = self.surrogate.predict(X.values)
        mae = mean_absolute_error(y_true, y_pred)
        return mae

    def exploitation(self, objective_evaluator):
        """
        Runs exploitatio stage
        :param objective_evaluator: function which takes sequence of parameter dicts
         and returns array of scores
        :return: final mode mae on best params
        """
        space = self.__sample_batch(self.exploitation_eval_size * self.exploitation_multiplier)
        scores_pred = self.surrogate.predict(space.values)
        idxs = np.argsort(scores_pred)[::-1]
        idxs = idxs[:self.exploitation_eval_size]
        params = space.iloc[idxs]
        scores_pred = scores_pred[idxs]

        scores_true = objective_evaluator(params.to_dict(orient='records'))
        mae = mean_absolute_error(scores_true, scores_pred)
        self.__update_history(params, scores_true)
        return mae

    @property
    def best_params(self):
        """
        Best params so far
        :return: dictionary with parameters
        """
        params = pd.concat(self.__history_parameters)
        scores = np.array(self.__history_scores)

        idx = np.argmax(scores)
        return params.to_dict(orient='records')[idx]





