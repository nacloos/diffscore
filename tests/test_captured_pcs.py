import numpy as np
import matplotlib.pyplot as plt

from diffscore import pipeline_optim_score



if __name__ == "__main__":
    dataset = "ultrametric"

    pipeline_optim_score(
        dataset=dataset,
        measure="procrustes-angular-score",
        stop_score=0.7
    )

    plt.show()
