import numpy as np
import pandas as pd

from collections import namedtuple
from typing import *

ParameterData = namedtuple("ParameterData", ['is_categorical', 'size', 'fn'])


class ParameterDistribution:
    """
    Named interface to underlying parameter distribution
    """

    def __init__(self, name: str, data: ParameterData, random_state: int = None):
        """
        Creates new parameter distribution interface
        :param name: name of parameter
        :param data: parameter data, which describes distribution and provides sampling function
        :param random_state: random state, used to initialize random number generator
        """
        self.__name = name
        self.__sampling_fn = data.fn
        self.__rng = np.random.default_rng(np.random.default_rng(random_state))
        self.__is_categorical = data.is_categorical
        self.__size = data.size

    @property
    def size(self):
        return self.__size

    @property
    def is_categorical(self):
        return self.__is_categorical

    @property
    def name(self):
        return self.__name

    def sample(self):
        """
        Samples single value from underlying parameter distribution
        :return: dict {parameter name -> sampled value}
        """
        param_value = self.__sampling_fn(self.__rng)
        return {
            self.__name: param_value
        }


class ParameterSpace:
    """
    Wraps several distributions into parameter space
    """
    def __init__(self, params: List[ParameterDistribution]):
        """
        Constructs new parameter space from given parameter distributions
        :param params: list of parameter distribution
        """
        self.__params = params
        self.__names = [p.name for p in self.__params]
        self.__categoricals = {p.name: p.size for p in self.__params if p.is_categorical}

    def __to_dataframe(self, params: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(params, index=[0], columns=self.__names)

    def sample(self, size: int = 10) -> pd.DataFrame:
        """
        Jointly samples from underlying parameter distirbutions
        :param size: number of parameter combinations to sample
        :return: dataframe with sampled parameter
        """
        for i in range(size):
            param_sample = {}
            for p in self.__params:
                param_sample.update(p.sample())

            yield self.__to_dataframe(param_sample)

    @property
    def categoricals(self):
        return self.__categoricals

    @classmethod
    def from_dict(cls, param_space: Dict[str, ParameterData], random_state: int = None):
        """
        Constructs parameter space from sklearn-like dict definition
        :param param_space: dict {param name -> parameter data}
        :param random_state: random state to initialize rng
        :return: instance of parameter space
        """
        param_list = []
        for name, param_data in param_space.items():
            dist = ParameterDistribution(name, param_data, random_state)

            param_list.append(dist)

        return cls(param_list)
