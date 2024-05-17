from functools import partial

from . import toy
from .toy import exp_gaussian, ultrametric, toy2d

from diffscore import register


register("dataset.toy2d", toy2d)
register("dataset.exp-gaussian", exp_gaussian)
register("dataset.exp-gaussian-dim50", partial(exp_gaussian, dim=50))
register("dataset.ultrametric", ultrametric)


@register("dataset.Hatsopoulos2007")
def hatsopoulos2007():
    from .hatsopoulos2007 import process_data
    # activity during movement period
    data = process_data(condition_average=False, dt=50, align_range=(0, 500), align_event="mvt_onset")
    return data["activity"], data["trial_info"]


@register("dataset.Mante2013")
def mante2013(for_decoding=True):
    import numpy as np
    from .mante2013 import process_data

    data = process_data(dt=50)

    # TODO: temp
    if not for_decoding:
        return data

    # preprocess motion and color coherence values for decoding
    conditions = data["trial_info"]
    coh_mod1_values = np.unique([cond["coh_mod1"] for cond in conditions])
    coh_mod2_values = np.unique([cond["coh_mod2"] for cond in conditions])

    for cond in conditions:
        # convert to categorical values (replace by index in unique values)
        # don't use this because low decoding accuracy (around 0.23%)
        # cond["coh_mod1"] = np.where(cond["coh_mod1"] == coh_mod1_values)[0][0]
        # cond["coh_mod2"] = np.where(cond["coh_mod2"] == coh_mod2_values)[0][0]

        # convert to binary values (0 if lower than median, 1 if higher)
        cond["coh_mod1"] = int(np.where(cond["coh_mod1"] < np.median(coh_mod1_values), 0, 1))
        cond["coh_mod2"] = int(np.where(cond["coh_mod2"] < np.median(coh_mod2_values), 0, 1))
    return data["activity"], conditions


@register("dataset.Mante2013-var95")
def mante2013_var95():
    from sklearn.decomposition import PCA
    X, conditions = mante2013(for_decoding=True)

    pca = PCA(n_components=0.95)
    X = pca.fit_transform(X.reshape(X.shape[0]*X.shape[1], -1)).reshape(X.shape[0], X.shape[1], -1)
    return X, conditions


@register("dataset.Mante2013-var99")
def mante2013_var99():
    from sklearn.decomposition import PCA
    X, conditions = mante2013(for_decoding=True)

    pca = PCA(n_components=0.99)
    X = pca.fit_transform(X.reshape(X.shape[0]*X.shape[1], -1)).reshape(X.shape[0], X.shape[1], -1)
    return X, conditions


@register("dataset.MajajHong2015")
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

    return X, conditions


@register("dataset.FreemanZiemba2013")
def freemanziemba2013(bin_size=150):
    import numpy as np
    import brainscore_vision
    from brainscore_vision.benchmark_helpers.neural_common import average_repetition

    neural_data = brainscore_vision.load_dataset("FreemanZiemba2013.public")
    neural_data = neural_data.transpose('time_bin', 'presentation', 'neuroid')
    neural_data = average_repetition(neural_data)

    X = neural_data.values
    # average over time bins
    X = np.mean(X.reshape(X.shape[0] // bin_size, bin_size, *X.shape[1:]), axis=1)

    conditions = [{"texture_type": texture_type} for texture_type in neural_data["texture_type"].values]
    # convert to numerical
    texture_types = np.unique(neural_data["texture_type"].values)
    conditions = [{
        "texture_type": np.where(texture_types == texture_type)[0][0]
    } for texture_type in neural_data["texture_type"].values]
    return X, conditions
