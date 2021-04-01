import logging

import numpy as np

from sklearn.model_selection._search import BaseSearchCV
from rich.logging import RichHandler

from .surrogate import SurrogateModel
from .model import MODELS
from .parameter import ParameterSpace

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("search")


class AlfSearchCV(BaseSearchCV):

    def __init__(self, estimator, param_distributions, *,
                 seed_size: int = 10, n_iter: int = 25,
                 n_runs_per_iter: int = 10,
                 score_surrogate: bool = True,
                 surrogate_model: str = "random_forest",
                 exploraion_multiplier: int = 10,
                 exploitation_multiplier: int = 10,
                 exploitation_eval_size: int = 10,
                 scoring=None, n_jobs=None, refit=True,
                 cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score=np.nan,
                 return_train_score=False):

        self.random_state = random_state
        self.score_surrogate = score_surrogate
        self.param_distributions = param_distributions

        self.seed_size = seed_size
        self.n_iter = n_iter
        self.n_runs_per_iter = n_runs_per_iter
        self.exploraion_multiplier = exploraion_multiplier
        self.exploitation_multiplier = exploitation_multiplier
        self.exploitation_eval_size = exploitation_eval_size
        self.surrogate_model = surrogate_model

        self._estimator, self._query_strategy = MODELS[surrogate_model]
        self.__surrogate = None

        super().__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    def _run_search(self, evaluate_candidates):
        log.setLevel(self.verbose)

        self.__surrogate = SurrogateModel(
            self._estimator, self._query_strategy, ParameterSpace.from_dict(self.param_distributions),
            seed_size=self.seed_size, n_exploration_iterations=self.n_iter,
            n_runs_per_iter=self.n_runs_per_iter, exploration_multiplier=self.exploraion_multiplier,
            exploitation_multiplier=self.exploitation_multiplier, exploitation_eval_size=self.exploitation_eval_size
        )

        def objective_evaluator(params):
            result = evaluate_candidates(params)
            return result['mean_test_score'][-len(params):]

        log.info("Running seeding stage")
        self.__surrogate.seeding(objective_evaluator)

        log.info("Running exploration stage")
        self.__surrogate.exploration(objective_evaluator)

        if self.score_surrogate:
            log.info("Scoring on historical data")
            mae = self.__surrogate.score_on_history()
            log.info(f"Surrogate MAE on historical data: {mae:.4f}")

        log.info("Running exploitation - searching for best params")
        final_mae = self.__surrogate.exploitation(objective_evaluator)
        log.info(f"Final surrogate mae on best params: {final_mae:.4f}")
