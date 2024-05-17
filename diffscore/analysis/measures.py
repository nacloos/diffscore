from functools import partial
import numpy as np
from sklearn.model_selection import KFold

import torch

import similarity
from diffscore import register, make


def reshape2d(X, Y, to_tensor=True):
    if to_tensor:
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)

    # convert to same dtype (some measures raise error if dtype is different)
    X = X.double()
    Y = Y.double()

    if len(X.shape) == 3:
        X = X.reshape(X.shape[0]*X.shape[1], -1)
    if len(Y.shape) == 3:
        Y = Y.reshape(Y.shape[0]*Y.shape[1], -1)
    return X, Y


class RegCCA:
    """
    Code adapted from https://github.com/ahwillia/netrep/blob/main/netrep/metrics/linear.py
    """
    def __init__(self, alpha=1, zero_pad=True, scoring_method='angular'):
        self.alpha = alpha
        self.zero_pad = zero_pad
        self.scoring_method = scoring_method

    def fit(self, X, Y):
        X, Y = reshape2d(X, Y)
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

        U, _, Vt = torch.linalg.svd(Xw.T @ Yw)

        Zx = Zx.double()
        Zy = Zy.double()
        self.Wx = Zx @ U
        self.Wy = Zy @ Vt.T

        return self

    def score(self, X, Y):
        X, Y = reshape2d(X, Y)
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

    def fit_score(self, X, Y):
        return self.fit(X, Y).score(X, Y)

    def __call__(self, X, Y):
        return self.fit_score(X, Y)


class CKA:
    def __init__(self, arccos=False):
        self.arccos = arccos

    def score(self, X, Y):
        # X: time x trial x neuron
        X, Y = reshape2d(X, Y)
        # score = linear_CKA(X1, X2)
        # assert torch.allclose(cka_svd(X1@X1.T, X2@X2.T), score)
        score = cka_svd(X@X.T, Y@Y.T)
        return score if not self.arccos else torch.arccos(score)

    def __call__(self, X, Y):
        return self.score(X, Y)


class RSA:
    def __init__(self, arccos=False):
        self.arccos = arccos

    def score(self, X, Y):
        # X: time x trial x neuron
        X, Y = reshape2d(X, Y)

        XX, YY = centering(X@X.T), centering(Y@Y.T)
        score = torch.sum(XX*YY)/(torch.linalg.norm(XX.reshape(-1))*torch.linalg.norm(YY.reshape(-1)))
        return score if not self.arccos else torch.arccos(score)

    def __call__(self, X, Y):
        return self.score(X, Y)


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


@register("measure.cka-angular")
def cka_angular():
    cka = CKA(arccos=True)
    def _fit_score(X, Y):
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        return cka(X, Y)
    return _fit_score


@register("measure.cka-angular-score")
def _():
    cka = make("measure.cka-angular")
    def _fit_score(X, Y):
        return 1 - cka(X, Y) / (np.pi/2)
    return _fit_score


# TODO: correlation-corr correct?
register("measure.rsa-correlation-corr", partial(RSA, arccos=False))


@register("measure.procrustes-angular")
def procrustes_angular():
    cca = RegCCA(alpha=1)
    def _fit_score(X, Y):
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        return cca(X, Y)
    return _fit_score


@register("measure.procrustes-angular-score")
def _():
    proc = make("measure.procrustes-angular")
    def _fit_score(X, Y):
        return 1 - proc(X, Y) / (np.pi/2)
    return _fit_score


@register("measure.procrustes-angular-cv")
def procrustes_angular_cv(n_splits=5, fit_ratio=0.8):
    cca = RegCCA(alpha=1)
    def _fit_score(X, Y):
        # cca.fit(X, Y)
        # score1 = cca.score(X, Y)
        # score2 = make("measure.procrustes-angular")(X, Y)
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


@register("measure.procrustes-angular-cv-score")
def _():
    proc = make("measure.procrustes-angular-cv")
    def _fit_score(X, Y):
        return 1 - proc(X, Y) / (np.pi/2)
    return _fit_score


@register("measure.procrustes-euclidean")
def _():
    cca = RegCCA(alpha=1, scoring_method="euclidean")
    def _fit_score(X, Y):
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        return cca(X, Y)
    return _fit_score


@register("measure.cca-angular")
def cca_angular():
    cca = RegCCA(alpha=0)
    def _fit_score(X, Y):
        X = torch.as_tensor(X)
        Y = torch.as_tensor(Y)
        return cca(X, Y)
    return _fit_score


@register("measure.cca-angular-score")
def _():
    cca = make("measure.cca-angular")
    def _fit_score(X, Y):
        return 1 - cca(X, Y) / (np.pi/2)
    return _fit_score


# @register("measure.svcca-angular")
# TODO: match dim wth pca instead of zero padding?

@register("measure.linreg")
def linreg(arccos=False, zero_pad=True):
    # X: neural data, Y: model data
    # ref: https://arxiv.org/pdf/1905.00414.pdf
    # R2 = 1 - min_B || X - YB ||_F^2 / || X ||_F^2 = || Q_Y.T X ||_F^2 / || X ||_F^2
    def _fit_score(X, Y):
        # n_steps, n_trials, n_neurons = X.shape
        X, Y = reshape2d(X, Y)
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


register("measure.linreg-angular", partial(linreg, arccos=True))


@register("measure.linreg-cv")
def linreg_cv(arccos=False, zero_pad=True, n_splits=5, fit_ratio=0.8):
    class LinRegScore:
        def __init__(self, arccos=False, zero_pad=True):
            self.arccos = arccos
            self.zero_pad = zero_pad

        def fit(self, X, Y):
            X, Y = reshape2d(X, Y)
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
            X, Y = reshape2d(X, Y)
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
        # score2 = make("measure.linreg", arccos=arccos)(X, Y)
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


register("measure.linreg-angular-cv", partial(linreg_cv, arccos=True))


@register("measure.linreg-r2#5folds_cv")
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
        X, Y = reshape2d(X, Y)

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
    "measure.linreg-r2#no_cv",
    partial(measure_linreg, n_splits=None)
)
register(
    "measure.ridge-lambda1-r2#5folds_cv",
    partial(measure_linreg, alpha=1)
)
register(
    "measure.ridge-lambda10-r2#5folds_cv",
    partial(measure_linreg, alpha=10)
)
register(
    "measure.ridge-lambda100-r2#5folds_cv",
    partial(measure_linreg, alpha=100)
)
register(
    "measure.ridge-lambda1-r2#no_cv",
    partial(measure_linreg, alpha=1, n_splits=None)
)
register(
    "measure.ridge-lambda1-r2",
    partial(measure_linreg, alpha=1, n_splits=None)
)
register(
    "measure.ridge-lambda10-r2#no_cv",
    partial(measure_linreg, alpha=10, n_splits=None)
)
register(
    "measure.ridge-lambda100-r2#no_cv",
    partial(measure_linreg, alpha=100, n_splits=None)
)
register(
    "measure.ridge-lambda10-r2",
    partial(measure_linreg, alpha=10, n_splits=None)
)
register(
    "measure.ridge-lambda100-r2",
    partial(measure_linreg, alpha=100, n_splits=None)
)


def kfold_crossval(measure, n_splits=5):
    def _fit_score(X, Y):
        X, Y = reshape2d(X, Y)

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


@register("measure.procrustes-angular#5folds_cv")
def _():
    measure = RegCCA(alpha=1)
    return kfold_crossval(measure=measure, n_splits=5)


@register("measure.procrustes-angular-score#5folds_cv")
def _():
    measure = RegCCA(alpha=1, scoring_method="euclidean")
    def _fit_score(X, Y):
        return 1 - measure(X, Y) / (np.pi/2)
    return _fit_score


@register("measure.procrustes-euclidean-score")
def _():
    measure = RegCCA(alpha=1)
    def _fit_score(X, Y):
        return 1 - kfold_crossval(measure=measure, n_splits=5)(X, Y) / (np.pi/2)
    return _fit_score


def test_pytorch_metric(dataset, metric_id):
    data = dataset.get_activity()
    cond_avg = data.mean(axis=1, keepdims=True).repeat(data.shape[1], axis=1)

    metric = make(f"measure.{metric_id}")
    metric2 = similarity.make(f"measure.{metric_id}")

    score = metric(data, cond_avg)
    score2 = metric2.fit_score(data, cond_avg)
    assert np.allclose(score.numpy(), score2)
    print("Test passed for", metric_id)


@register("measure.nbs-squared")
def _():
    def _fit_score(X, Y):
        X, Y = reshape2d(X, Y)
        # centering
        X = X - torch.mean(X, dim=0)
        Y = Y - torch.mean(Y, dim=0)

        sXY = torch.linalg.svdvals(X.T @ Y)
        sXX = torch.linalg.svdvals(X @ X.T)
        sYY = torch.linalg.svdvals(Y @ Y.T)

        nbs_squared = torch.sum(sXY)**2 / (torch.sum(sXX) * torch.sum(sYY))
        return nbs_squared
    return _fit_score


@register("measure.nbs")
def _():
    nbs_square = make("measure.nbs-squared")
    def _fit_score(X, Y):
        return torch.sqrt(nbs_square(X, Y))
    return _fit_score

@register("measure.nbs-angular")
def _():
    nbs = make("measure.nbs")
    def _fit_score(X, Y):
        return torch.acos(nbs(X, Y))
    return _fit_score


@register("measure.nbs-angular-score")
def _():
    nbs = make("measure.nbs-angular")
    def _fit_score(X, Y):
        return 1 - nbs(X, Y) / (np.pi/2)
    return _fit_score

# TODO: what about nbs-squared-angular? (don't think it is a metric like nbs-angular = procrustes-angular metric)


@register("measure.cka")
def _():
    def _fit_score(X, Y):
        X, Y = reshape2d(X, Y)
        # centering
        X = X - torch.mean(X, dim=0)
        Y = Y - torch.mean(Y, dim=0)

        sXY = torch.linalg.svdvals(X.T @ Y)
        sXX = torch.linalg.svdvals(X @ X.T)
        sYY = torch.linalg.svdvals(Y @ Y.T)

        cka = torch.sum(sXY**2) / (torch.sqrt(torch.sum(sXX**2)) * torch.sqrt(torch.sum(sYY**2)))
        return cka
    return _fit_score


@register("measure.ensd")
def _():
    def _fit_score(X, Y):
        X, Y = reshape2d(X, Y)
        # centering
        X = X - torch.mean(X, dim=0)
        Y = Y - torch.mean(Y, dim=0)

        # https://www.biorxiv.org/content/10.1101/2023.07.27.550815v1.full.pdf
        YtX = Y.T @ X
        XtY = YtX.T
        XtX = X.T @ X
        YtY = Y.T @ Y
        score = torch.trace(YtX @ XtY) * torch.trace(XtX) * torch.trace(YtY) / (torch.trace(XtX @ XtX) * torch.trace(YtY @ YtY))

        return score
    return _fit_score


@register("measure.cka-hsic_song")
def cka_hsic_song():
    """
    Code adapted from https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
    Convert numpy to pytorch
    """
    def center_gram(gram):
        n = gram.shape[0]
        gram.fill_diagonal_(0)
        means = torch.sum(gram, 0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram.fill_diagonal_(0)
        return gram

    def _fit_score(X, Y):
        X, Y = reshape2d(X, Y)

        gram_x = X @ X.T
        gram_y = Y @ Y.T

        gram_x = center_gram(gram_x)
        gram_y = center_gram(gram_y)

        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
        normalization_x = torch.linalg.norm(gram_x)
        normalization_y = torch.linalg.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)

    return _fit_score


if __name__ == "__main__":
    import similarity
    X, Y = np.random.rand(10, 15, 5), np.random.rand(10, 15, 5)
    cka = similarity.make("measure.kornblith19.cka-hsic_song")
    cka2 = similarity.make("measure.diffscore.cka-hsic_song")

    print(cka(X, Y))
    print(cka2(X, Y))
