import numpy as np

from typing import *

from .parameter import ParameterData


def uniform(start: float = 0.0, end: float = 0.0):
    """
    Provides continuous uniform distribution
    :param start: lowest bound for values
    :param end: highest bound for values
    :return: parameter data
    """

    assert start <= end, "start must be lower than end"

    def sampling_fn(rng: np.random.Generator):
        return rng.uniform(low=start, high=end, size=1)

    return ParameterData(False, None, sampling_fn)


def categorical(values: Union[List, np.ndarray]):
    """
    Wraps provided array into discrete uniform distribution
    :param values: array with values to be sampled
    :return: parameter data
    """

    assert len(values) > 0, "You must supply non-empty array as values"

    def sampling_fn(rng: np.random.Generator):
        return rng.choice(values, size=1)

    return ParameterData(True, len(values), sampling_fn)


def weighted_categorical(values: Union[List[Tuple[float, Any]], Dict[Any, float]]):
    """
    Wraps provided array of tuples or dict into custom destribution with provided weights
    :param values: array of tuples: (value, weight) or dict {value -> weight}
    :return: parameter data
    """
    if isinstance(values, list):
        a = [x[1] for x in values]
        p = [x[0] for x in values]
    elif isinstance(values, dict):
        a = list(values.keys())
        p = [values[k] for k in a]
    else:
        raise ValueError("values must be a list of tuples or dict")

    def sampling_fn(rng: np.random.Generator):
        return rng.choice(a, size=1, p=p)

    return ParameterData(True, len(a), sampling_fn)


def normal(loc: float = 0.0, scale: float = 1.0):
    """
    Provides normal distribution ~N[loc, scale]
    :param loc: mean of distribution
    :param scale: std of distribution
    :return: parameter data
    """

    assert scale >= 0, "scale must be >= 0"

    def sampling_fn(rng: np.random.Generator):
        return rng.normal(loc=loc, scale=scale, size=1)

    return ParameterData(False, None, sampling_fn)


def lognormal(mean: float = 0.0, sigma: float = 0.0):
    """
    Provides log-scaled normal distribution
    :param mean: mean of underlying normal distribution
    :param sigma: std(~scale) of underlying normal distribution
    :return: parameter data
    """
    def sampling_fn(rng: np.random.Generator):
        return rng.lognormal(mean=mean, sigma=sigma, size=1)

    return ParameterData(False, None, sampling_fn)


def exponential(scale: float = 0.0):
    """
    Provides exponential distribution
    :param scale: beta-parameter, must be >= 0
    :return: parameter data
    """

    assert scale >= 0, 'Scale must be >= 0'

    def sampling_fn(rng: np.random.Generator):
        return rng.exponential(scale=scale, size=1)

    return ParameterData(False, None, sampling_fn)