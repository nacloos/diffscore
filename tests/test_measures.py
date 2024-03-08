import numpy as np

import similarity
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
    fit_measure(data=X, measure=measure, stop_crit=0.5)


if __name__ == "__main__":
    test_measures()
    test_measure_optim()

