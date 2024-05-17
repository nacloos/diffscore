from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

import similarity
import diffscore
from diffscore import Measure
from diffscore.training import fit_measure


def _test_measure_equality(measure1, measure2, n_tries=10):
    for _ in range(n_tries):
        X = np.random.rand(10, 15, 5)
        Y = np.random.rand(10, 15, 5)
        # TODO: do that automatically
        import torch
        score2 = measure2(X, Y)
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        score1 = measure1(X, Y)
        if not np.allclose(score1, score2):
            # print abs max diff
            print("Scores are not equal", score1, score2)
            return False
    return True


# TODO: some of the measures don't give some scores for diffscore and other backends
def test_measures():
    measures = Measure("*")
    for id1, measure1 in measures.items():
        # same measure but different backend implementations
        measures2 = similarity.make(f"measure.*.{id1.split('.')[-1]}")
        for id2, measure2 in measures2.items():
            backend = id2.split('.')[1]
            if backend == "diffscore":
                continue

            print(id1, id2)
            _test_measure_equality(measure1, measure2)


def test_measure_optim():
    measure = Measure("procrustes-angular-score")

    X = np.random.rand(5, 10)
    fit_measure(dataset=X, measure=measure, stop_crit=0.5)


def measure_backend_consistency(save_dir=None):
    def test_measure_equal(measure1, measure2, n_repeats):
        scores1 = []
        scores2 = []
        for i in range(n_repeats):
            X = np.random.rand(20, 10, 50)
            Y = np.random.rand(20, 10, 50)
            score1 = measure1(X, Y)
            score2 = measure2(X, Y)
            scores1.append(score1)
            scores2.append(score2)

        # return relative error
        # return np.mean(np.abs(np.array(scores1) - np.array(scores2)) / np.array(scores1))
        return np.mean(np.abs(np.array(scores1) - np.array(scores2)))


    measures = Measure("*")

    results = defaultdict(list)
    for measure_id, measure in measures.items():
        print("Measure:", measure_id)
        other_backends = diffscore.make(f"measure.*.{measure_id.split('.')[-1]}")
        for other_measure_id, other_measure in other_backends.items():
            print(other_measure_id)
            relative_error = test_measure_equal(measure, other_measure, 50)
            print("Relative error:", relative_error)
            results["measure"].append(measure_id.split('.')[-1])
            results["backend"].append(other_measure_id.split('.')[1])
            results["relative_error"].append(relative_error)

    results_df = pd.DataFrame(results)

    # backend = index, measure = columns, relative_error = values (add NaN for missing values)
    results_df = results_df.pivot(index="backend", columns="measure", values="relative_error")
    # order columns alphabetically
    results_df = results_df.reindex(sorted(results_df.columns), axis=1)
    # order rows alphabetically
    results_df = results_df.reindex(sorted(results_df.index), axis=0)

    import seaborn as sns
    import matplotlib.pyplot as plt
    print(len(results_df.columns), len(results_df.index))

    plt.figure(figsize=(10, 5), dpi=500)
    sns.heatmap(results_df, cmap="viridis_r", cbar=True, linewidths=2,
                linecolor='white', cbar_kws={"shrink": 0.5, "label": "Mean Absolute Error"})
    plt.ylabel("Backends", fontsize=10)
    plt.xlabel("Measures", fontsize=10)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left', fontsize=8)
    plt.yticks(rotation=0, va='center', fontsize=8)
    plt.axis('scaled')
    plt.tight_layout()


    if save_dir:
        results_df.to_csv(save_dir / "backends_matrix.csv", index=False)
        plt.savefig(save_dir / "backends_matrix.png")
    print(results_df)

    # data for d3 heatmap
    # replace nans with -1
    results_df = results_df.fillna(-1)

    # retransform into columns table
    results_df = results_df.stack().reset_index()
    # results_df.columns = ["backend", "measure", "relative_error"]
    results_df.columns = ["variable", "group", "value"]
    if save_dir:
        results_df.to_csv(save_dir / "backends.csv", index=False)


if __name__ == "__main__":
    save_path = Path(__file__).parent / "results"
    save_path.mkdir(exist_ok=True)
    # TODO
    measure_backend_consistency(save_dir=save_path)

    test_measures()
    test_measure_optim()
