import numpy as np
import torch
import torch.nn as nn

from diffscore import Dataset, Measure


class MeasureOptim:
    def __init__(
            self,
            measure,
            stop_crit,
            optimizer="Adam",
            is_dist=False,
            lr=1e-1,
            n_iter=1000,
            log_steps=10
    ):
        self.measure = measure
        self.stop_crit = stop_crit
        self.is_dist = is_dist
        self.lr = lr
        self.n_iter = n_iter
        self.log_steps = log_steps
        self.optimizer = optimizer

    def fit(self, X, Y0=None):
        self.X = X = torch.Tensor(X)
        if Y0 is None:
            Y = nn.Parameter(torch.randn(X.shape))
        else:
            Y = nn.Parameter(torch.Tensor(Y0))
        self.Y0 = torch.clone(Y)
        self.Ys = [self.Y0]
        self.scores = [self.measure(X, self.Y0)]

        if self.optimizer == "Adam":
            optim = torch.optim.Adam([Y], lr=self.lr)
        elif self.optimizer == "SGD":
            optim = torch.optim.SGD([Y], lr=self.lr)
        else:
            raise NotImplementedError
        # print("Optimizing with", optim)

        resample = False
        for i in range(self.n_iter):
            optim.zero_grad()

            if resample:
                _Y = Y + torch.randn_like(Y) * 0.01
            else:
                _Y = Y

            score = self.measure(X, _Y)
            # print(score)

            if i % self.log_steps == 0:
                print("Iter {}, similarity: {}".format(i, score))

            self.Ys.append(torch.clone(Y))
            self.scores.append(score)
            if (not self.is_dist) and (score > self.stop_crit):
                break
            if self.is_dist and (score < self.stop_crit):
                break

            # min dist, max similarity
            loss = score if self.is_dist else -score
            loss.backward()

            # prevent nan grad with cka angular
            if torch.isnan(Y.grad).any():
                resample = True
                print("Nan grad, resampling")
                continue

            optim.step()

        self.scores = np.array([score.detach().numpy() for score in self.scores])
        print("Final score {}".format(score))
        self.Y = Y
        return self


def fit_measure(
    dataset,
    measure=None,
    metric_id=None,
    init_data_fit=None,
    optimizer="Adam",
    lr=1e-1,
    n_iter=1000,
    log_steps=10,
    **kwargs
):
    """
    Optimize similarity measure between datasets.

    Args:
        dataset: array X or dataset id (str). 
            X: np.ndarray of shape (n_steps, n_conditions, n_neurons)
            condition: list[dict] of length n_conditions
        measure: str or Measure object
        metric_id: used for backward compatibility
        init_data_fit: initial data to fit
        optimizer: optimizer name
        lr: learning rate
        n_iter: max number of iterations
        log_steps: log every n steps

    Returns:
        dict with keys:
            reference_dataset: X
            fitted_data: Y
            fitted_datasets: list of Ys
            scores: list of scores
            score: final score
            metric_id: measure.id
    """
    if isinstance(dataset, str):
        data, conditions = Dataset(dataset)
    else:
        data = dataset

    # metric_id for backward compatibility
    if measure is None:
        measure = metric_id
    measure = Measure(measure) if isinstance(measure, str) else measure

    if init_data_fit is not None:
        # prevent inplace modification
        init_data_fit = init_data_fit.copy()

    optim = MeasureOptim(measure, optimizer=optimizer, lr=lr, n_iter=n_iter, log_steps=log_steps, **kwargs)
    optim.fit(data, Y0=init_data_fit)

    return {
        "reference_dataset": data,
        "fitted_data": optim.Y.detach().numpy(),
        "fitted_datasets": [Y.detach().numpy() for Y in optim.Ys],
        "scores": optim.scores,
        "score": optim.scores[-1],
    }
