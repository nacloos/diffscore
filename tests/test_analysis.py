import numpy as np
import matplotlib.pyplot as plt

from diffscore import pipeline_optim_score, optim_square


def test_pipeline_optim_score():
    dataset = "ultrametric"
    pipeline_optim_score(
        dataset=dataset,
        measure="procrustes-angular-score",
        stop_score=0.7
    )


def test_optim_square():
    dataset = "ultrametric"
    measure1 = "procrustes-angular-score"
    measure2 = "cka-angular-score"
    optim_square(measure1, measure2, dataset, N=2, n_iter=10)


if __name__ == "__main__":
    test_pipeline_optim_score()
    test_optim_square()

