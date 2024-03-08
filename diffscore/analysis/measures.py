from functools import partial
import numpy as np
from sklearn.model_selection import KFold

import torch

import similarity
from similarity import register, make


class RegCCA:
    """
    Code adapted from https://github.com/ahwillia/netrep/blob/main/netrep/metrics/linear.py
    """
    def __init__(self, alpha=1, zero_pad=True, scoring_method='angular'):
        self.alpha = alpha
        self.zero_pad = zero_pad
        self.scoring_method = scoring_method

    def fit(self, X, Y):
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0]*X.shape[1], -1).double()
            Y = Y.reshape(Y.shape[0]*Y.shape[1], -1).double()
        else:
            X = X.double()
            Y = Y.double()
        # zero padding
        X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)

        # centering
        self.mx = torch.mean(X, dim=0)
        self.my = torch.mean(Y, dim=0)
        X = X - self.mx
        Y = Y - self.my

        Xw, Zx = whiten(X, self.alpha)
        Yw, Zy = whiten(Y, self.alpha)

        _, sigma, _ = torch.linalg.svd(Xw.T @ Yw)
        U, _, Vt = torch.linalg.svd(Xw.T @ Yw)

        Zx = Zx.double()
        Zy = Zy.double()
        self.Wx = Zx @ U
        self.Wy = Zy @ Vt.T

        return self

    def score(self, X, Y):
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0]*X.shape[1], -1).double()
            Y = Y.reshape(Y.shape[0]*Y.shape[1], -1).double()
        else:
            X = X.double()
            Y = Y.double()
        # zero padding
        X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)
        # centering
        X = X - self.mx
        Y = Y - self.my
        # rotational alignment
        X = X @ self.Wx
        Y = Y @ self.Wy
        # scoring
        if self.scoring_method == 'angular':
            normalizer = torch.linalg.norm(X.ravel()) * torch.linalg.norm(Y.ravel())
            dist = torch.dot(X.ravel(), Y.ravel()) / normalizer
            dist = torch.arccos(dist)
        elif self.scoring_method == 'euclidean':
            dist = torch.linalg.norm(X - Y, ord='fro')
        else:
            raise NotImplementedError
        return dist

    def eval(self, activities):
        act1, act2 = activities
        X = act1.reshape(act1.shape[0]*act1.shape[1], -1)
        Y = act2.reshape(act2.shape[0]*act2.shape[1], -1)
        X = X.double()
        Y = Y.double()
        # print("X, Y", X.shape, Y.shape)

        # metric = LinearMetric(alpha=self.alpha)
        # metric.fit(X.detach().numpy(), Y.detach().numpy())
        # gt = metric.score(X.detach().numpy(), Y.detach().numpy())
        # Xtsf, Ytsf = metric.transform(X.detach().numpy(), Y.detach().numpy())

        # zero padding
        X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)
        # pad = torch.zeros(60, 2)
        # X = torch.concat([X, pad], dim=-1)
        # print("After padding", X.shape, Y.shape)

        # centering
        X = X - torch.mean(X, dim=0)
        Y = Y - torch.mean(Y, dim=0)

        Xw, Zx = whiten(X, self.alpha)
        Yw, Zy = whiten(Y, self.alpha)
        # assert np.allclose(Xw.detach().numpy(), metric.Xw), np.max(np.abs(Xw.detach().numpy() - metric.Xw))
        # assert np.allclose(Yw.detach().numpy(), metric.Yw), np.max(np.abs(Yw.detach().numpy() - metric.Yw))

        _, sigma, _ = torch.linalg.svd(Xw.T @ Yw)

        U, _, Vt = torch.linalg.svd(Xw.T @ Yw)

        # assert np.allclose(U.detach().numpy(), metric.U), np.max(np.abs(U.detach().numpy() - metric.U))
        # assert np.allclose(Vt.T.detach().numpy(), metric.V)
        X = Xw @ U
        Y = Yw @ Vt.T

        # assert np.allclose(X.detach().numpy(), Xtsf, atol=1e-6), np.max(np.abs(X.detach().numpy() - Xtsf))
        # assert np.allclose(Y.detach().numpy(), Ytsf, atol=1e-6)

        if self.scoring_method == 'angular':
            normalizer = torch.linalg.norm(X.ravel()) * torch.linalg.norm(Y.ravel())
            dist_bis = torch.sum(sigma) / normalizer
            dist_bis = torch.arccos(dist_bis)

            dist = torch.dot(X.ravel(), Y.ravel()) / normalizer
            dist = torch.arccos(dist)
            # assert np.allclose(dist.detach().numpy(), gt, atol=1e-6), "{} != {}".format(dist.detach().numpy(), gt)
            # assert torch.allclose(dist, dist_bis, atol=1e-6), "{} != {}".format(dist.detach().numpy(), dist_bis)

        elif self.scoring_method == 'euclidean':
            dist = torch.linalg.norm(X - Y, ord='fro')

        else:
            raise NotImplementedError
        return dist


def whiten(X, alpha, preserve_variance=True, eigval_tol=1e-7):
    # Return early if regularization is maximal (no whitening).
    if alpha > (1 - eigval_tol):
        return X, torch.eye(X.shape[1])

    # Compute eigendecomposition of covariance matrix
    lam, V = torch.linalg.eigh(X.T @ X)
    lam = torch.maximum(lam, torch.tensor(eigval_tol))

    # Compute diagonal of (partial) whitening matrix.
    #
    # When (alpha == 1), then (d == ones).
    # When (alpha == 0), then (d == 1 / sqrt(lam)).
    d = alpha + (1 - alpha) * lam ** (-1 / 2)

    # Rescale the whitening matrix.
    if preserve_variance:
        # Compute the variance of the transformed data.
        #
        # When (alpha == 1), then new_var = sum(lam)
        # When (alpha == 0), then new_var = len(lam)
        new_var = torch.sum(
            (alpha ** 2) * lam
            + 2 * alpha * (1 - alpha) * (lam ** 0.5)
            + ((1 - alpha) ** 2) * torch.ones_like(lam)
        )

        # Now re-scale d so that the variance of (X @ Z)
        # will equal the original variance of X.
        d *= torch.sqrt(torch.sum(lam) / new_var)

    # Form (partial) whitening matrix.
    Z = (V * d[None, :]) @ V.T

    # An alternative regularization strategy would be:
    #
    # lam, V = np.linalg.eigh(X.T @ X)
    # d = lam ** (-(1 - alpha) / 2)
    # Z = (V * d[None, :]) @ V.T

    # Returned (partially) whitened data and whitening matrix.
    return X @ Z, Z


def check_equal_shapes(X, Y, nd=2, zero_pad=False):
    if (X.ndim != nd) or (Y.ndim != nd):
        raise ValueError(
            "Expected {}d arrays, but shapes were {} and "
            "{}.".format(nd, X.shape, Y.shape)
        )

    if X.shape != Y.shape:

        if zero_pad and (X.shape[:-1] == Y.shape[:-1]):

            # Number of padded zeros to add.
            n = max(X.shape[-1], Y.shape[-1])

            # Padding specifications for X and Y.
            px = torch.zeros((nd, 2), dtype=torch.int)
            py = torch.zeros((nd, 2), dtype=torch.int)
            # torch pad different from numpy pad!
            px[0, -1] = n - X.shape[-1]
            py[0, -1] = n - Y.shape[-1]

            # Pad X and Y with zeros along final axis.
            X = torch.nn.functional.pad(X, tuple(px.flatten()))
            Y = torch.nn.functional.pad(Y, tuple(py.flatten()))

        else:
            raise ValueError(
                "Expected arrays with equal dimensions, "
                "but got arrays with shapes {} and {}."
                "".format(X.shape, Y.shape))

    return X, Y


class CKA:
    def __init__(self, arccos=False):
        self.arccos = arccos

    def eval(self, activities):
        # activity: time x trial x neuron
        act1, act2 = activities
        X1 = act1.reshape(act1.shape[0]*act1.shape[1], -1)
        X2 = act2.reshape(act2.shape[0]*act2.shape[1], -1)
        # score = linear_CKA(X1, X2)
        # assert torch.allclose(cka_svd(X1@X1.T, X2@X2.T), score)
        score = cka_svd(X1@X1.T, X2@X2.T)
        return score if not self.arccos else torch.arccos(score)


class RSA:
    def __init__(self, arccos=False):
        self.arccos = arccos

    def eval(self, activities):
        # activity: time x trial x neuron
        act1, act2 = activities
        X = act1.reshape(act1.shape[0]*act1.shape[1], -1)
        Y = act2.reshape(act2.shape[0]*act2.shape[1], -1)

        XX, YY = centering(X@X.T), centering(Y@Y.T)
        score = torch.sum(XX*YY)/(torch.linalg.norm(XX.reshape(-1))*torch.linalg.norm(YY.reshape(-1)))
        return score if not self.arccos else torch.arccos(score)


def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n], dtype=K.dtype)
    I = torch.eye(n, dtype=K.dtype)
    H = I - unit / n

    return (H @ K) @ H  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def linear_HSIC(X, Y):
    L_X = X @ X.T
    L_Y = Y @ Y.T
    return torch.sum(centering(L_X) * centering(L_Y))  # <vec(XX.T, vec(YY.T)>


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def cka_svd(XX, YY):
    XX, YY = centering(XX), centering(YY)
    # assert torch.allclose(XX, XX.T, atol=1e-5), torch.max(torch.abs(XX - XX.T))
    # assert torch.allclose(YY, YY.T, atol=1e-5), torch.max(torch.abs(YY - YY.T))

    lambX, uX = torch.linalg.eigh(XX)
    lambY, uY = torch.linalg.eigh(YY)
    uX_uY = uX.T @ uY

    lambX_norm = torch.sqrt(torch.sum(lambX**2))
    lambY_norm = torch.sqrt(torch.sum(lambY**2))

    return torch.sum(torch.outer(lambX, lambY) * uX_uY * uX_uY) / (lambX_norm*lambY_norm)


@register("measure.pytorch-cka-angular")
def cka_angular():
    cka = CKA(arccos=True)
    def _fit_score(X, Y):
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        return cka.eval([X, Y])
    return _fit_score


@register("measure.pytorch-cka-angular-score")
def _():
    cka = make("measure.pytorch-cka-angular")
    def _fit_score(X, Y):
        return 1 - cka(X, Y) / (np.pi/2)
    return _fit_score


@register("measure.pytorch-procrustes-angular")
def procrustes_angular():
    cca = RegCCA(alpha=1)
    def _fit_score(X, Y):
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        return cca.eval([X, Y])
    return _fit_score


@register("measure.pytorch-procrustes-angular-score")
def _():
    proc = make("measure.pytorch-procrustes-angular")
    def _fit_score(X, Y):
        return 1 - proc(X, Y) / (np.pi/2)
    return _fit_score


@register("measure.pytorch-procrustes-angular-cv")
def procrustes_angular_cv(n_splits=5, fit_ratio=0.8):
    cca = RegCCA(alpha=1)
    def _fit_score(X, Y):
        # cca.fit(X, Y)
        # score1 = cca.score(X, Y)
        # score2 = make("measure.pytorch-procrustes-angular")(X, Y)
        # assert torch.allclose(score1, score2), f"{score1} != {score2}"
        # print("Check passed")

        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)

        # cross val over conditions
        n_conditions = X.shape[1]
        n_fit = int(n_conditions * fit_ratio)

        scores = torch.zeros(n_splits)
        for i in range(n_splits):
            # torch doesn't have setdiff1d
            indices = torch.randperm(n_conditions)
            fit_conditions = indices[:n_fit]
            val_conditions = indices[n_fit:]
            assert len(fit_conditions) + len(val_conditions) == n_conditions

            fit_X = X[:, fit_conditions, :]
            val_X = X[:, val_conditions, :]

            fit_Y = Y[:, fit_conditions, :]
            val_Y = Y[:, val_conditions, :]

            cca.fit(fit_X, fit_Y)
            score = cca.score(val_X, val_Y)
            scores[i] = score
        return torch.mean(scores)

    return _fit_score


@register("measure.pytorch-procrustes-angular-cv-score")
def _():
    proc = make("measure.pytorch-procrustes-angular-cv")
    def _fit_score(X, Y):
        return 1 - proc(X, Y) / (np.pi/2)
    return _fit_score


@register("measure.pytorch-procrustes-euclidean")
def _():
    cca = RegCCA(alpha=1, scoring_method="euclidean")
    def _fit_score(X, Y):
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        return cca.eval([X, Y])
    return _fit_score


@register("measure.pytorch-cca-angular")
def cca_angular():
    cca = RegCCA(alpha=0)
    def _fit_score(X, Y):
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        return cca.eval([X, Y])
    return _fit_score


@register("measure.pytorch-cca-angular-score")
def _():
    cca = make("measure.pytorch-cca-angular")
    def _fit_score(X, Y):
        return 1 - cca(X, Y) / (np.pi/2)
    return _fit_score


# @register("measure.pytorch-svcca-angular")
# TODO: match dim wth pca instead of zero padding?

@register("measure.pytorch-linreg")
def linreg(arccos=False, zero_pad=True):
    # X: neural data, Y: model data
    # ref: https://arxiv.org/pdf/1905.00414.pdf
    # R2 = 1 - min_B || X - YB ||_F^2 / || X ||_F^2 = || Q_Y.T X ||_F^2 / || X ||_F^2
    def _fit_score(X, Y):
        n_steps, n_trials, n_neurons = X.shape
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        X = X.reshape(n_steps * n_trials, n_neurons)
        Y = Y.reshape(n_steps * n_trials, Y.shape[-1])

        # zero padding
        X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)

        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)

        Q, R = torch.linalg.qr(Y)
        R2 = torch.linalg.norm(Q.T @ X) ** 2 / torch.linalg.norm(X) ** 2

        if arccos:
            if torch.abs(R2 - 1) < 1e-5:
                # arccos of 1 gives nan
                return torch.tensor(0.)
            R2 = torch.arccos(R2)
        return R2
    return _fit_score


register("measure.pytorch-linreg-angular", partial(linreg, arccos=True))


@register("measure.pytorch-linreg-cv")
def linreg_cv(arccos=False, zero_pad=True, n_splits=5, fit_ratio=0.8):
    class LinRegScore:
        def __init__(self, arccos=False, zero_pad=True):
            self.arccos = arccos
            self.zero_pad = zero_pad

        def fit(self, X, Y):
            X = torch.as_tensor(X).reshape(X.shape[0]*X.shape[1], X.shape[2])
            Y = torch.as_tensor(Y).reshape(Y.shape[0]*Y.shape[1], Y.shape[2])
            # zero padding
            X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)

            self.mx = X.mean(axis=0)
            self.my = Y.mean(axis=0)
            X = X - self.mx
            Y = Y - self.my

            # self.Q, self.R = torch.linalg.qr(Y)
            # self.B = torch.linalg.solve(self.R, self.Q.T @ X)
            self.B = torch.linalg.lstsq(Y, X).solution

        def score(self, X, Y):
            X = torch.as_tensor(X).reshape(X.shape[0]*X.shape[1], X.shape[2])
            Y = torch.as_tensor(Y).reshape(Y.shape[0]*Y.shape[1], Y.shape[2])
            # zero padding
            X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)

            X = X - self.mx
            Y = Y - self.my

            X_pred = Y @ self.B
            R2 = 1 - torch.linalg.norm(X - X_pred) ** 2 / torch.linalg.norm(X) ** 2
            if self.arccos:
                if torch.abs(R2 - 1) < 1e-5:
                    return torch.tensor(0.)
                # TODO: nan if R2 is too large
                R2 = torch.arccos(R2)
            return R2

    linreg = LinRegScore(arccos=arccos, zero_pad=zero_pad)

    def _fit_score(X, Y):
        # linreg.fit(X, Y)
        # score1 = linreg.score(X, Y)
        # score2 = make("measure.pytorch-linreg", arccos=arccos)(X, Y)
        # assert torch.allclose(score1, score2), f"{score1} != {score2}"
        # print("Check passed")

        # cross val over conditions
        n_conditions = X.shape[1]
        n_fit = int(n_conditions * fit_ratio)

        scores = torch.zeros(n_splits)
        for i in range(n_splits):
            indices = torch.randperm(n_conditions)
            fit_conditions = indices[:n_fit]
            val_conditions = indices[n_fit:]
            assert len(fit_conditions) + len(val_conditions) == n_conditions

            fit_X = X[:, fit_conditions, :]
            val_X = X[:, val_conditions, :]

            fit_Y = Y[:, fit_conditions, :]
            val_Y = Y[:, val_conditions, :]

            linreg.fit(fit_X, fit_Y)
            score = linreg.score(val_X, val_Y)
            scores[i] = score
        score = torch.mean(scores)
        return score

    return _fit_score


register("measure.pytorch-linreg-angular-cv", partial(linreg_cv, arccos=True))


@register("measure.diffscore.linreg-r2#5folds_cv")
def measure_linreg(zero_pad=True, alpha=0, n_splits=5):
    class LinRegScore:
        def fit(self, X, Y):
            # zero padding
            X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)

            self.mx = X.mean(axis=0)
            self.my = Y.mean(axis=0)
            X = X - self.mx
            Y = Y - self.my

            # fit mapping from Y to X
            if alpha > 0:
                self.B = torch.linalg.lstsq(Y.T @ Y + alpha * torch.eye(Y.shape[1]), Y.T @ X).solution

            else:
                # self.Q, self.R = torch.linalg.qr(Y)
                # self.B = torch.linalg.solve(self.R, self.Q.T @ X)
                self.B = torch.linalg.lstsq(Y, X).solution

        def score(self, X, Y):
            # zero padding
            X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)

            X = X - self.mx
            Y = Y - self.my

            X_pred = Y @ self.B
            R2 = 1 - torch.linalg.norm(X - X_pred) ** 2 / torch.linalg.norm(X) ** 2
            return R2

    linreg = LinRegScore()

    def _fit_score(X, Y):
        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)

        if n_splits is None:
            linreg.fit(X, Y)
            score = linreg.score(X, Y)
            return score

        # cross val over time and conditions concatenated
        kfold = KFold(n_splits=n_splits, shuffle=False)
        scores = torch.zeros(n_splits)
        for i, (train_index, test_index) in enumerate(kfold.split(X)):
            fit_X = X[train_index]
            val_X = X[test_index]

            fit_Y = Y[train_index]
            val_Y = Y[test_index]

            linreg.fit(fit_X, fit_Y)
            score = linreg.score(val_X, val_Y)
            scores[i] = score
        score = torch.mean(scores)
        return score

    return _fit_score


register(
    "measure.diffscore.linreg-r2#no_cv",
    partial(measure_linreg, n_splits=None)
)
register(
    "measure.diffscore.ridge-lambda1-r2#5folds_cv",
    partial(measure_linreg, alpha=1)
)
register(
    "measure.diffscore.ridge-lambda10-r2#5folds_cv",
    partial(measure_linreg, alpha=10)
)
register(
    "measure.diffscore.ridge-lambda100-r2#5folds_cv",
    partial(measure_linreg, alpha=100)
)
register(
    "measure.diffscore.ridge-lambda1-r2#no_cv",
    partial(measure_linreg, alpha=1, n_splits=None)
)
register(
    "measure.diffscore.ridge-lambda10-r2#no_cv",
    partial(measure_linreg, alpha=10, n_splits=None)
)
register(
    "measure.diffscore.ridge-lambda100-r2#no_cv",
    partial(measure_linreg, alpha=100, n_splits=None)
)


def kfold_crossval(measure, n_splits=5):
    def _fit_score(X, Y):
        print(X.shape, Y.shape)
        X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
        Y = Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2])
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        print(X.shape, Y.shape)

        if n_splits is None:
            measure.fit(X, Y)
            score = measure.score(X, Y)
            return score

        # cross val over time and conditions concatenated
        kfold = KFold(n_splits=n_splits, shuffle=False)
        scores = torch.zeros(n_splits)
        for i, (train_index, test_index) in enumerate(kfold.split(X)):
            fit_X = X[train_index]
            val_X = X[test_index]

            fit_Y = Y[train_index]
            val_Y = Y[test_index]

            measure.fit(fit_X, fit_Y)
            score = measure.score(val_X, val_Y)
            scores[i] = score
        score = torch.mean(scores)
        return score

    return _fit_score


@register("measure.diffscore.procrustes-angular#5folds_cv")
def _():
    measure = RegCCA(alpha=1)
    return kfold_crossval(measure=measure, n_splits=5)


@register("measure.diffscore.procrustes-angular-score#5folds_cv")
def _():
    measure = RegCCA(alpha=1, scoring_method="euclidean")
    def _fit_score(X, Y):
        return 1 - measure(X, Y) / (np.pi/2)
    return _fit_score


@register("measure.diffscore.procrustes-euclidean-score")
def _():
    measure = RegCCA(alpha=1)
    def _fit_score(X, Y):
        return 1 - kfold_crossval(measure=measure, n_splits=5)(X, Y) / (np.pi/2)
    return _fit_score


def test_pytorch_metric(dataset, metric_id):
    data = dataset.get_activity()
    cond_avg = data.mean(axis=1, keepdims=True).repeat(data.shape[1], axis=1)

    metric = make(f"measure.pytorch-{metric_id}")
    metric2 = similarity.make(f"measure.{metric_id}")

    score = metric(data, cond_avg)
    score2 = metric2.fit_score(data, cond_avg)
    assert np.allclose(score.numpy(), score2)
    print("Test passed for", metric_id)


@register("measure.pytorch-nbs-squared")
def _():
    def _fit_score(X, Y):
        X = torch.as_tensor(X.reshape(X.shape[0]*X.shape[1], X.shape[2]))
        Y = torch.as_tensor(Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2]))
        # centering
        X = X - torch.mean(X, dim=0)
        Y = Y - torch.mean(Y, dim=0)

        sXY = torch.linalg.svdvals(X.T @ Y)
        sXX = torch.linalg.svdvals(X @ X.T)
        sYY = torch.linalg.svdvals(Y @ Y.T)

        nbs_squared = torch.sum(sXY)**2 / (torch.sum(sXX) * torch.sum(sYY))
        return nbs_squared
    return _fit_score


@register("measure.pytorch-nbs")
def _():
    nbs_square = make("measure.pytorch-nbs-squared")
    def _fit_score(X, Y):
        return torch.sqrt(nbs_square(X, Y))
    return _fit_score

@register("measure.pytorch-nbs-angular")
def _():
    nbs = make("measure.pytorch-nbs")
    def _fit_score(X, Y):
        return torch.acos(nbs(X, Y))
    return _fit_score


@register("measure.pytorch-nbs-angular-score")
def _():
    nbs = make("measure.pytorch-nbs-angular")
    def _fit_score(X, Y):
        return 1 - nbs(X, Y) / (np.pi/2)
    return _fit_score

# TODO: what about nbs-squared-angular? (don't think it is a metric like nbs-angular = procrustes-angular metric)


@register("measure.pytorch-cka")
def _():
    def _fit_score(X, Y):
        X = torch.as_tensor(X.reshape(X.shape[0]*X.shape[1], X.shape[2]))
        Y = torch.as_tensor(Y.reshape(Y.shape[0]*Y.shape[1], Y.shape[2]))
        # centering
        X = X - torch.mean(X, dim=0)
        Y = Y - torch.mean(Y, dim=0)

        sXY = torch.linalg.svdvals(X.T @ Y)
        sXX = torch.linalg.svdvals(X @ X.T)
        sYY = torch.linalg.svdvals(Y @ Y.T)

        cka = torch.sum(sXY**2) / (torch.sqrt(torch.sum(sXX**2)) * torch.sqrt(torch.sum(sYY**2)))
        return cka
    return _fit_score


measure_ids = make(id="measure.pytorch-*").keys()
for measure_id in measure_ids:
    register(
        f"measure.diffscore.{measure_id.split('pytorch-')[1]}",
        partial(make, measure_id)
    )


# def test_pytorch_measures():
#     for i in range(10):
#         X = np.random.randn(15, 72, 100)
#         Y = np.random.randn(15, 72, 100)
#         pytorch_cka = make("measure.pytorch-cka")
#         cka = similarity.make("measure.cka")

#         score = pytorch_cka(X, Y).numpy()
#         score2 = cka.fit_score(X, Y)
#         assert np.allclose(score, score2)

#         pytorch_nbs = make("measure.pytorch-nbs-squared")
#         procrustes = similarity.make("measure.procrustes-angular")

#         score = pytorch_nbs(X, Y).numpy()
#         score2 = procrustes.fit_score(X, Y)
#         assert np.allclose(score, np.cos(score2)**2)
#     print("Test passed for pytorch metrics")


# def test_diffscore_measures():
#     measure_ids = make(id="measure.pytorch-*").keys()

#     for measure_id in measure_ids:
#         print("Testing", measure_id)
#         print("diffscore id:", f"measure.diffscore.{measure_id.split('pytorch-')[1]}")
#         X = np.random.randn(15, 72, 100)
#         Y = np.random.randn(15, 72, 100)

#         measure = similarity.make(f"measure.diffscore.{measure_id.split('pytorch-')[1]}")
#         pytorch_measure = make(measure_id)
#         score = measure(X, Y)
#         score2 = pytorch_measure(X, Y).numpy()

#         assert np.allclose(score, score2), np.max(np.abs(score - score2))
#     print("Test passed for diffscore measures")


# if __name__ == "__main__":
#     test_diffscore_measures()
