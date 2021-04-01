import sys
sys.path.append("../")

import alf


def test_consistent_naming():
    param_space = alf.ParameterSpace.from_dict({
        'a1': alf.categorical([1, 2, 3]),
        'a2': alf.categorical([3, 4, 5]),
        'a3': alf.categorical([5, 6, 7])
    })
    column_name_1 = tuple(next(param_space.sample(1)).columns)
    for i in range(1000):
        frame = next(param_space.sample(1))
        column_names = tuple(frame.columns)
        assert column_names == column_name_1, "Column names must be the same"


def test_correct_categoricals():
    param_space = alf.ParameterSpace.from_dict({
        'c1': alf.categorical([1, 2, 3]),
        'c2': alf.categorical([3, 4, 5]),
        'c3': alf.weighted_categorical({1: 0.1, 2: 0.1, 3: 0.8}),
        'n1': alf.normal(0.0, 1.0),
        'n2': alf.lognormal(0.0, 1.0),
        'n3': alf.exponential(10)
    })

    cat_names = set(param_space.categoricals.keys())
    assert len(cat_names.difference({'c1', 'c2', 'c3'})) == 0, "categorical must include all cat variables"
