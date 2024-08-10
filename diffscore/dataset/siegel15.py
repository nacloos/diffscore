from copy import deepcopy
from functools import cache, partial
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from multisys.dataset import Siegel15Dataset
from multisys.dataset.preprocessing import TimeRestriction, RegionRestricter
from multisys.analysis.decoding.siegel15 import decode_label

from diffscore import make, register


REGIONS = ["FEF", "V4", "IT", "MT", "LIP", "PFC"]

def siegel15_decoding():
    # labels = ['context', 'direction', 'color', 'response']
    labels = ['context', 'response']

    methods = ['pev', 'mutual_information', 'decoding']
    method = 'pev'
    method = "decoding"

    dataset = Siegel15Dataset(dt=50, load=True)
    # dataset = Siegel15Dataset()

    TimeRestriction(t=dataset.t, t_min=-1500, t_max=500).transform(dataset)


    for r in dataset.get_region_names():
        print(r, dataset.get_activity(region=r).shape)

    # TODO: decoding not working for color and direction (bin small and large values?)
    print(np.unique(dataset.get_trial_info(label='color')))
    print(np.unique(dataset.get_trial_info(label='direction')))
    print(np.unique(dataset.get_trial_info(label='context')))
    print(np.unique(dataset.get_trial_info(label='response')))

    # for trial in range(dataset.get_activity().shape[1]):
    #     trial_info = dataset.get_trial_info(trial=trial)
    #     print(trial_info)

    for i, label in enumerate(labels):
        legend = True if i == 0 else False
        legend = False
        decode_label(dataset, label=label, method=method, legend=legend, save_path=Path('figures') / 'decoding' / method)


# @register("dataset.siegel15")
# def siegel15_dataset(dt=50, load=True):
#     return Siegel15Dataset(dt=dt, load=load)


@cache
def _make_siegel15_dataset_cached(**kwargs):
    return Siegel15Dataset(**kwargs)


@register("dataset.siegel15")
def siegel15_data(dt=50, subject="paula", region=None, t_min=-1500, t_max=500, pca_components=None, use_cache=False):
    if use_cache:
        dataset = _make_siegel15_dataset_cached(dt=dt, load=True)
        dataset = deepcopy(dataset)
    else:
        dataset = Siegel15Dataset(dt=dt, subject=subject, load=True)

    TimeRestriction(t=dataset.t, t_min=t_min, t_max=t_max).transform(dataset)

    if region is not None:
        dataset = RegionRestricter(region).transform(dataset)

    data = dataset.get_activity()
    conditions = dataset.get_trial_info()
    print("Original data shape:", data.shape)

    if pca_components is not None:
        pca = PCA(n_components=pca_components)
        data = pca.fit_transform(data.reshape(data.shape[0] * data.shape[1], -1)).reshape(data.shape[0], data.shape[1], -1)

    for cond in conditions:
        del cond["timing"]
        # remove some labels
        del cond["correct"]
        del cond["gt_choice"]
        del cond["context_cue"]

    color_values = np.unique([cond["color"] for cond in conditions])
    direction_values = np.unique([cond["direction"] for cond in conditions])
    for cond in conditions:
        # convert to binary values (0 if lower than median, 1 if higher)
        cond["color"] = np.where(cond["color"] < np.median(color_values), 0, 1)
        cond["direction"] = np.where(cond["direction"] < np.median(direction_values), 0, 1)
        cond["color"] = int(cond["color"])
        cond["direction"] = int(cond["direction"])

    return data, conditions


register("dataset.siegel15-FEF", partial(siegel15_data, region="FEF"))
register("dataset.siegel15-V4", partial(siegel15_data, region="V4"))
register("dataset.siegel15-IT", partial(siegel15_data, region="IT"))
register("dataset.siegel15-stim_period-var99", partial(siegel15_data, t_min=0, t_max=500, pca_components=0.99))


for region in REGIONS:
    register(
        f"dataset.siegel15-{region}-stim_period",
        partial(
            siegel15_data,
            t_min=0,
            t_max=500,
            region=region
        )
    )
    register(f"dataset.siegel15-{region}-stim_period#paula", partial(siegel15_data, subject="paula", t_min=0, t_max=500, region=region))
    register(f"dataset.siegel15-{region}-stim_period#rex", partial(siegel15_data, subject="rex", t_min=0, t_max=500, region=region))

    register(
        f"dataset.siegel15-{region}-stim_period-var99",
        partial( 
            siegel15_data,
            t_min=0,
            t_max=500,
            region=region,
            pca_components=0.99
        )
    )

    register(
        f"dataset.siegel15-{region}-var99",
        partial(
            siegel15_data,
            region=region,
            pca_components=0.99
        )
    )


if __name__ == '__main__':
    import diffscore_exp.analysis

    data, conditions = make(
        "dataset.siegel15",
        region="V4",
        pca_components=0.99
    )
    # TODO: can't decode color if take all the regions (but work if take V4)

    label = "color"
    make(
        "decoder.spynal",
        data=data,
        labels=[cond[label] for cond in conditions],
        method="decode",
        plot=True
    )
    plt.show()

    metric_id = "procrustes-angular"
    stop_crit = 0.1

    res = make("pipeline.fit_metric", data=data, stop_crit=stop_crit, metric_id=metric_id, lr=1e-1)

    plt.figure()
    plt.plot(res["scores"])
    plt.show()
