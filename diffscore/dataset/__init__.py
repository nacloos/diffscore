from . import toy
from .toy import exp_gaussian, ultrametric, toy2d

from diffscore import register


register("dataset.diffscore.toy2d", toy2d)
register("dataset.diffscore.exp-gaussian", exp_gaussian)
register("dataset.diffscore.ultrametric", ultrametric)
# register("dataset.mante13-var99", )
# register("dataset.siegel15", )



@register("dataset.diffscore.MajajHong2015")
def majajhong2015():
    import numpy as np
    import brainscore_vision
    from brainscore_vision.benchmark_helpers.neural_common import average_repetition

    neural_data = brainscore_vision.load_dataset("MajajHong2015.public")
    neural_data = neural_data.transpose('time_bin', 'presentation', 'neuroid')

    neural_data = neural_data.sel(region='IT')
    neural_data = average_repetition(neural_data)
    neural_data = neural_data.squeeze('time_bin')

    X = neural_data.values
    # convert to numerical
    categories = np.unique(neural_data["category_name"].values)
    conditions = [
        {
            "category": np.where(categories == cat)[0][0],
        } for cat, obj in zip(neural_data["category_name"].values, neural_data["object_name"].values)
    ]

    # TODO: temp for debugging
    # X = X[:300]
    # conditions = conditions[:300]
    return X, conditions
