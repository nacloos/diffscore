import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict


def decoder_logistic(data, labels, penalty='l2', max_iter=200, cv="StratifiedKFold", n_splits=5, plot=False, **kwargs):
    labels = np.array(labels)

    decoder = LogisticRegression(penalty=penalty, max_iter=max_iter, **kwargs)
    if cv == "StratifiedKFold":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    else:
        cv = None

    if data.ndim == 2:
        # assume no time dimension
        data = data[None]

    res = defaultdict(list)
    for t in range(data.shape[0]):
        X = data[t]
        y = labels

        if cv is not None:
            scores = []
            for (train_idxs, test_idxs) in cv.split(X, y):
                decoder = decoder.fit(X[train_idxs], y[train_idxs])
                score = decoder.score(X[test_idxs], y[test_idxs])
                scores.append(score)
            res["score"].append(np.mean(scores))
            res["score_std"].append(np.std(scores))
        else:
            decoder = decoder.fit(X, y)
            score = decoder.score(X, y)
            res["score"].append(score)

    if plot:
        plt.figure()
        plt.plot(res["score"])
        if "score_std" in res:
            plt.fill_between(
                range(len(res["score"])),
                np.array(res["score"]) - np.array(res["score_std"]),
                np.array(res["score"]) + np.array(res["score_std"]),
                alpha=0.2
            )
        plt.ylabel("Score")
        plt.xlabel("Time")
        # plt.show()

    return dict(res)
