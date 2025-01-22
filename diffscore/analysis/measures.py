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
    # TODO: raise RuntimeError: Index out of range
    # X = X.double()
    # Y = Y.double()

    Y = Y.to(X.dtype)

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


@register("measure/cka-angular")
def cka_angular(X, Y):
    cka = CKA(arccos=True)
    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y)
    return cka(X, Y)


@register("measure/cka-angular-score")
def cka_angular_score(X, Y):
    return 1 - cka_angular(X, Y) / (np.pi/2)


@register("measure/rsa-correlation-corr")
def rsa_correlation_corr(X, Y):
    rsa = RSA(arccos=False)
    return rsa(X, Y)


def kernel_to_rdm(K):
    return torch.diag(K)[:, None] + torch.diag(K)[None, :] - 2 * K


def linear_kernel(X):
    return X @ X.T


@register("measure/rsa-euclidean-cosine")
def rsa_euclidean_cosine(X, Y):
    X, Y = reshape2d(X, Y)
    rdmX = kernel_to_rdm(linear_kernel(X))
    rdmY = kernel_to_rdm(linear_kernel(Y))
    return torch.trace(rdmX@rdmY) / (torch.linalg.norm(rdmX)*torch.linalg.norm(rdmY))


@register("measure/rsa-euclidean-centered-cosine")
def rsa_euclidean_centered_cosine(X, Y):
    X, Y = reshape2d(X, Y)
    rdmX = kernel_to_rdm(linear_kernel(X))
    rdmY = kernel_to_rdm(linear_kernel(Y))
    rdmX = centering(rdmX)
    rdmY = centering(rdmY)
    return torch.trace(rdmX@rdmY) / (torch.linalg.norm(rdmX)*torch.linalg.norm(rdmY))

@register("measure/rsa-correlation-cosine")
def rsa_correlation_cosine(X, Y):
    def corr_rdm(X):
        X_centered = X - X.mean(dim=1, keepdim=True)
        X_centered = X_centered / torch.sqrt(torch.clip(torch.sum(X_centered**2, dim=1, keepdim=True), min=1e-8))
        return 1 - X_centered @ X_centered.T

    X, Y = reshape2d(X, Y)
    rdmX = corr_rdm(X)
    rdmY = corr_rdm(Y)
    return torch.trace(rdmX@rdmY) / (torch.linalg.norm(rdmX)*torch.linalg.norm(rdmY))


@register("measure/rsa-euclidean-corr")
def rsa_euclidean_corr(X, Y):
    X, Y = reshape2d(X, Y)
    R1 = kernel_to_rdm(linear_kernel(X))
    R2 = kernel_to_rdm(linear_kernel(Y))

    triu_indices = torch.triu_indices(R1.shape[0], R1.shape[1], offset=1)
    r1 = R1[triu_indices[0], triu_indices[1]]
    r2 = R2[triu_indices[0], triu_indices[1]]
    r1_mean = r1.mean()
    r2_mean = r2.mean()
    # center the vectors
    r1 = r1 - r1_mean
    r2 = r2 - r2_mean
    # compute correlation
    return torch.sum(r1*r2) / (torch.linalg.norm(r1) * torch.linalg.norm(r2))



@register("measure/procrustes-angular")
def procrustes_angular(X, Y):
    cca = RegCCA(alpha=1)
    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y)
    return cca(X, Y)


@register("measure/procrustes-angular-score")
def procrustes_angular_score(X, Y):
    return 1 - procrustes_angular(X, Y) / (np.pi/2)


@register("measure/procrustes-angular-cv")
def procrustes_angular_cv(X, Y, n_splits=5, fit_ratio=0.8):
    cca = RegCCA(alpha=1)
    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y)

    # cross val over conditions
    n_conditions = X.shape[1]
    n_fit = int(n_conditions * fit_ratio)

    scores = torch.zeros(n_splits)
    for i in range(n_splits):
        indices = torch.randperm(n_conditions)
        fit_conditions = indices[:n_fit]
        val_conditions = indices[n_fit:]

        fit_X = X[:, fit_conditions, :]
        val_X = X[:, val_conditions, :]
        fit_Y = Y[:, fit_conditions, :]
        val_Y = Y[:, val_conditions, :]

        cca.fit(fit_X, fit_Y)
        scores[i] = cca.score(val_X, val_Y)
    return torch.mean(scores)


@register("measure/procrustes-angular-cv-score")
def procrustes_angular_cv_score(X, Y):
    return 1 - procrustes_angular_cv(X, Y) / (np.pi/2)


@register("measure/procrustes-euclidean")
def procrustes_euclidean(X, Y):
    cca = RegCCA(alpha=1, scoring_method="euclidean")
    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y)
    return cca(X, Y)


@register("measure/cca-angular")
def cca_angular(X, Y):
    cca = RegCCA(alpha=0)
    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y)
    return cca(X, Y)


@register("measure/cca-angular-score")
def cca_angular_score(X, Y):
    return 1 - cca_angular(X, Y) / (np.pi/2)


# @register("measure/svcca-angular")
# TODO: match dim wth pca instead of zero padding?

@register("measure/linreg")
def linreg(X, Y, arccos=False, zero_pad=True):
    X, Y = reshape2d(X, Y)
    X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)

    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    Q, R = torch.linalg.qr(Y)
    R2 = torch.linalg.norm(Q.T @ X) ** 2 / torch.linalg.norm(X) ** 2

    if arccos:
        if torch.abs(R2 - 1) < 1e-5:
            return torch.tensor(0.)
        R2 = torch.arccos(R2)
    return R2


@register("measure/linreg-angular")
def linreg_angular(X, Y, zero_pad=True):
    return linreg(X, Y, arccos=True, zero_pad=zero_pad)


@register("measure/linreg-cv")
def linreg_cv(X, Y, arccos=False, zero_pad=True, n_splits=5, fit_ratio=0.8):
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
    n_conditions = X.shape[1]
    n_fit = int(n_conditions * fit_ratio)

    scores = torch.zeros(n_splits)
    for i in range(n_splits):
        indices = torch.randperm(n_conditions)
        fit_conditions = indices[:n_fit]
        val_conditions = indices[n_fit:]

        fit_X = X[:, fit_conditions, :]
        val_X = X[:, val_conditions, :]
        fit_Y = Y[:, fit_conditions, :]
        val_Y = Y[:, val_conditions, :]

        linreg.fit(fit_X, fit_Y)
        score = linreg.score(val_X, val_Y)
        scores[i] = score
    score = torch.mean(scores)
    return score



register("measure/linreg-angular-cv", partial(linreg_cv, arccos=True))


@register("measure/linreg-r2#5folds_cv")
def measure_linreg(X, Y,zero_pad=True, alpha=0, n_splits=5, agg_fun="r2"):
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

            if agg_fun == "r2":
                R2 = 1 - torch.linalg.norm(X - X_pred) ** 2 / torch.linalg.norm(X) ** 2
                return R2
            elif agg_fun == "pearsonr":
                r = torch.dot(X.ravel(), X_pred.ravel()) / (torch.linalg.norm(X) * torch.linalg.norm(X_pred))
                # from scipy.stats import pearsonr
                # r_gt = pearsonr(X.ravel().detach(), X_pred.ravel().detach())
                # breakpoint()
                # print(r_gt.statistic, r)
                return r
            else:
                raise NotImplementedError(f"agg_fun={agg_fun}")

    linreg = LinRegScore()

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



# register(
#     "measure.linreg-r2#no_cv",
#     partial(measure_linreg, n_splits=None)
# )
# register(
#     "measure.ridge-lambda1-r2#5folds_cv",
#     partial(measure_linreg, alpha=1)
# )
# register(
#     "measure.ridge-lambda10-r2#5folds_cv",
#     partial(measure_linreg, alpha=10)
# )
# register(
#     "measure.ridge-lambda100-r2#5folds_cv",
#     partial(measure_linreg, alpha=100)
# )
# register(
#     "measure.ridge-lambda1000-r2#5folds_cv",
#     partial(measure_linreg, alpha=1000)
# )
# register(
#     "measure.ridge-lambda1-r2#no_cv",
#     partial(measure_linreg, alpha=1, n_splits=None)
# )
# register(
#     "measure.ridge-lambda1-r2",
#     partial(measure_linreg, alpha=1, n_splits=None)
# )
# register(
#     "measure.ridge-lambda10-r2#no_cv",
#     partial(measure_linreg, alpha=10, n_splits=None)
# )
# register(
#     "measure.ridge-lambda100-r2#no_cv",
#     partial(measure_linreg, alpha=100, n_splits=None)
# )
# register(
#     "measure.ridge-lambda10-r2",
#     partial(measure_linreg, alpha=10, n_splits=None)
# )
# register(
#     "measure.ridge-lambda100-r2",
#     partial(measure_linreg, alpha=100, n_splits=None)
# )

# # pearsonr
# register(
#     "measure.linreg-pearsonr#5folds_cv",
#     partial(measure_linreg, agg_fun="pearsonr")
# )
# register(
#     "measure.linreg-pearsonr#5folds_cv",
#     partial(measure_linreg, agg_fun="pearsonr")
# )
# register(
#     "measure.ridge-lambda1-pearsonr#5folds_cv",
#     partial(measure_linreg, alpha=1, agg_fun="pearsonr")
# )
# register(
#     "measure.ridge-lambda10-pearsonr#5folds_cv",
#     partial(measure_linreg, alpha=10, agg_fun="pearsonr")
# )
# register(
#     "measure.ridge-lambda100-pearsonr#5folds_cv",
#     partial(measure_linreg, alpha=100, agg_fun="pearsonr")
# )
# register(
#     "measure.ridge-lambda1000-pearsonr#5folds_cv",
#     partial(measure_linreg, alpha=1000, agg_fun="pearsonr")
# )

# # pearsonr, no cv
# register(
#     "measure.linreg-pearsonr#no_cv",
#     partial(measure_linreg, n_splits=None, agg_fun="pearsonr")
# )
# register(
#     "measure.linreg-pearsonr#no_cv",
#     partial(measure_linreg, n_splits=None, agg_fun="pearsonr")
# )
# register(
#     "measure.ridge-lambda1-pearsonr#no_cv",
#     partial(measure_linreg, alpha=1, n_splits=None, agg_fun="pearsonr")
# )
# register(
#     "measure.ridge-lambda10-pearsonr#no_cv",
#     partial(measure_linreg, alpha=10, n_splits=None, agg_fun="pearsonr")
# )
# register(
#     "measure.ridge-lambda100-pearsonr#no_cv",
#     partial(measure_linreg, alpha=100, n_splits=None, agg_fun="pearsonr")
# )
# register(
#     "measure.ridge-lambda1000-pearsonr#no_cv",
#     partial(measure_linreg, alpha=1000, n_splits=None, agg_fun="pearsonr")
# )


for n_splits in [None, 5]:
    for alpha in [0, 1, 10, 100, 1000, 10000]:
        for agg_fun in ["r2", "pearsonr"]:
            name = "linreg" if alpha == 0 else f"ridge-lambda{alpha}"
            cv = "no_cv" if n_splits is None else f"{n_splits}folds_cv"
            # print("registering", f"measure.{name}-{agg_fun}#{cv}")
            register(
                f"measure/{name}-{agg_fun}#{cv}",
                partial(measure_linreg, alpha=alpha, n_splits=n_splits, agg_fun=agg_fun)
            )

# linear regression symmetric
def measure_linreg_sym(X, Y, zero_pad=True, alpha=0, n_splits=5, agg_fun="r2"):
    linreg = partial(measure_linreg, zero_pad=zero_pad, alpha=alpha, n_splits=n_splits, agg_fun=agg_fun)

    score1 = linreg(X, Y)
    score2 = linreg(Y, X)
    return (score1 + score2) / 2


for n_splits in [None, 5]:
    for alpha in [0, 1, 10, 100, 1000, 10000]:
        for agg_fun in ["r2", "pearsonr"]:
            name = "linreg" if alpha == 0 else f"ridge-lambda{alpha}"
            cv = "no_cv" if n_splits is None else f"{n_splits}folds_cv"
            # print("registering", f"measure.{name}-{agg_fun}-sym#{cv}")
            register(
                f"measure/{name}-{agg_fun}-sym#{cv}",
                partial(measure_linreg_sym, alpha=alpha, n_splits=n_splits, agg_fun=agg_fun)
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


@register("measure/procrustes-angular#5folds_cv")
def procrustes_angular_cv(X, Y, n_splits=5):
    measure = RegCCA(alpha=1)
    return kfold_crossval(measure=measure, n_splits=n_splits)(X, Y)


@register("measure/procrustes-angular-score#5folds_cv")
def procrustes_angular_score_cv(X, Y, n_splits=5):
    measure = RegCCA(alpha=1, scoring_method="euclidean")
    return kfold_crossval(measure=measure, n_splits=n_splits)(X, Y)


@register("measure/procrustes-euclidean-score#5folds_cv")
def procrustes_euclidean_score_cv(X, Y, n_splits=5):
    measure = RegCCA(alpha=1)
    return kfold_crossval(measure=measure, n_splits=n_splits)(X, Y)


def test_pytorch_metric(dataset, metric_id):
    data = dataset.get_activity()
    cond_avg = data.mean(axis=1, keepdims=True).repeat(data.shape[1], axis=1)

    metric = make(f"measure/{metric_id}")
    metric2 = similarity.make(f"measure/{metric_id}")

    score = metric(data, cond_avg)
    score2 = metric2.fit_score(data, cond_avg)
    assert np.allclose(score.numpy(), score2)
    print("Test passed for", metric_id)


@register("measure/nbs-squared")
def nbs_squared(X, Y):
    X, Y = reshape2d(X, Y)
    X = X - torch.mean(X, dim=0)
    Y = Y - torch.mean(Y, dim=0)

    sXY = torch.linalg.svdvals(X.T @ Y)
    sXX = torch.linalg.svdvals(X @ X.T)
    sYY = torch.linalg.svdvals(Y @ Y.T)

    nbs_squared = torch.sum(sXY)**2 / (torch.sum(sXX) * torch.sum(sYY))
    return nbs_squared


@register("measure/nbs")
def nbs(X, Y):
    return torch.sqrt(nbs_squared(X, Y))


@register("measure/nbs-angular")
def nbs_angular(X, Y):
    return torch.acos(nbs(X, Y))


@register("measure/nbs-angular-score")
def nbs_angular_score(X, Y):
    return 1 - nbs_angular(X, Y) / (np.pi/2)


@register("measure/cka")
def cka(X, Y):
    X, Y = reshape2d(X, Y)
    X = X - torch.mean(X, dim=0)
    Y = Y - torch.mean(Y, dim=0)

    sXY = torch.linalg.svdvals(X.T @ Y)
    sXX = torch.linalg.svdvals(X @ X.T)
    sYY = torch.linalg.svdvals(Y @ Y.T)

    cka = torch.sum(sXY**2) / (torch.sqrt(torch.sum(sXX**2)) * torch.sqrt(torch.sum(sYY**2)))
    return cka


@register("measure/ensd")
def ensd(X, Y):
    X, Y = reshape2d(X, Y)
    X = X - torch.mean(X, dim=0)
    Y = Y - torch.mean(Y, dim=0)

    YtX = Y.T @ X
    XtY = YtX.T
    XtX = X.T @ X
    YtY = Y.T @ Y
    score = torch.trace(YtX @ XtY) * torch.trace(XtX) * torch.trace(YtY) / (torch.trace(XtX @ XtX) * torch.trace(YtY @ YtY))

    return score


@register("measure/cka-hsic_song")
def cka_hsic_song(X, Y):
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

    X, Y = reshape2d(X, Y)
    gram_x = X @ X.T
    gram_y = Y @ Y.T

    gram_x = center_gram(gram_x)
    gram_y = center_gram(gram_y)

    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
    normalization_x = torch.linalg.norm(gram_x)
    normalization_y = torch.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


if __name__ == "__main__":
    import similarity
    X, Y = np.random.rand(10, 5), np.random.rand(10, 5)

    rsa1 = make("measure/rsa-euclidean-cosine")(torch.as_tensor(X), torch.as_tensor(Y))
    rsa2 = similarity.make("measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=cosine")(X, Y)
    print(rsa1, rsa2)
    assert np.allclose(rsa1, rsa2)


    rsa1 = make("measure/rsa-correlation-cosine")(torch.as_tensor(X), torch.as_tensor(Y))
    rsa2 = similarity.make("measure/rsatoolbox/rsa-rdm=correlation-compare=cosine")(X, Y)
    print(rsa1, rsa2)
    assert np.allclose(rsa1, rsa2)

    rsa1 = make("measure/rsa-euclidean-corr")(torch.as_tensor(X), torch.as_tensor(Y))
    rsa2 = similarity.make("measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=corr")(X, Y)
    print(rsa1, rsa2)
    assert np.allclose(rsa1, rsa2)
    breakpoint()


    cka = similarity.make("measure.kornblith19.cka-hsic_song")
    cka2 = similarity.make("measure.diffscore.cka-hsic_song")

    print(cka(X, Y))
    print(cka2(X, Y))
