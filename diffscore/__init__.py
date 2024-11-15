
registry = {}


def register(id, obj=None):
    def _register(id, obj):
        registry[id] = obj

    if obj is None:
        def decorator(obj):
            _register(id, obj)
            return obj
        return decorator
    else:
        _register(id, obj)


def make(id, *args, **kwargs):
    return registry[id]


class Env:
    def __new__(cls, env_id: str, *args, **kwargs):
        return make(f"env/{env_id}", *args, **kwargs)


class Measure:
    def __new__(cls, measure_id: str, *args, **kwargs):
        return make(f"measure/{measure_id}", *args, **kwargs)
    

class Dataset:
    def __new__(cls, dataset_id: str, *args, **kwargs):
        return make(f"dataset/{dataset_id}", *args, **kwargs)()


# important to place imports after class definitions because use the classes in the imports
from diffscore import analysis
from diffscore import nn
from diffscore import model  # backward compatibility
from diffscore import dataset
from diffscore import env
# have to import training after env because use env in training
from diffscore import training

from diffscore.training import fit_measure, optimize, OptimResult
from diffscore.analysis import pipeline_optim_score
