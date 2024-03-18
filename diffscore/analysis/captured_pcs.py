from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from collections import defaultdict
import matplotlib.pyplot as plt

from netrep.validation import check_equal_shapes

from diffscore import Dataset
from diffscore import Measure
from diffscore.analysis.decoding import decoder_logistic
from diffscore.training.measure_optim import fit_measure


rcparams = {
    "figure.figsize": (3, 2),
    "figure.dpi": 130,
    "axes.spines.right": False,
    "axes.spines.top": False,
}


def projected_r2(X, Y):
    if X.ndim == 3:
        n_steps, n_trials = X.shape[0], X.shape[1]
        X = X.reshape(n_steps * n_trials, X.shape[-1])
        Y = Y.reshape(n_steps * n_trials, Y.shape[-1])

    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # zero padding
    X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=True)

    assert X.shape[0] > X.shape[1]

    # find linear alignment from Y to X (lin reg)
    Q, R = np.linalg.qr(Y)
    X_pred = Q @ (Q.T @ X)
    # compute X's PCs
    Ux, Sx, Vhx = np.linalg.svd(X, full_matrices=False)

    res = defaultdict(list)
    for i in range(X.shape[-1]):
        # project on PC i
        xi = X @ Vhx[i]
        xi_pred = X_pred @ Vhx[i]

        R2i = 1 - np.sum((xi - xi_pred) ** 2) / np.sum(xi ** 2)

        # reshape to temporal data
        if X.ndim == 3:
            xi = xi.reshape(n_steps, n_trials)
            xi_pred = xi_pred.reshape(n_steps, n_trials)

        assert np.allclose(np.sum(xi**2), Sx[i]**2)
        res["X_pc"].append(xi)
        res["X_pred_pc"].append(xi_pred)
        res["pc_var"].append(np.sum(xi ** 2))
        res["R2_pc"].append(R2i)
        res["R2_pc_weighted"].append(
            R2i * np.sum(xi ** 2)
        )

    return dict(res)


def pc_captured_variance(X, Ys, scores, n_components=None, plot_over_scores=True, threshold: float | str = "half", plot_threshold_lines=False, orth=False, cv=False, cv_kwargs={}, color_log_scale=True, save_dir=None):
    R2s = []
    for Y in Ys:
        res = projected_r2(X, Y)
        R2s.append(res["R2_pc"])

    R2s = np.array(R2s)
    if n_components is None:
        n_components = R2s.shape[1]

    # Interpolate and find the score where R2 crosses the threshold
    scores_at_threshold = []
    for i in range(n_components):
        f = interp1d(R2s[:, i], scores)

        if threshold == "half":
            # threshold is the score at which R2 is mid-way between its starting value and 1µ
            _threshold = (R2s[0, i] + 1) / 2
            # print("PC", i, "threshold:", _threshold)
        else:
            _threshold = threshold

        try:
            score_at_threshold = f(_threshold)
        except ValueError:
            # This occurs if the threshold is not in the range of R2 values
            score_at_threshold = np.nan
        scores_at_threshold.append(score_at_threshold)

    with plt.rc_context(rcparams):
        plt.figure(figsize=(3, 2), dpi=130)
        x = list(range(1, n_components+1))
        plt.plot(x, scores_at_threshold, color="tab:blue")
        plt.xlabel("PC")
        plt.ylabel(f"Score to reach $R^2_{{PC}} = {threshold}$")
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir / "scores_at_threshold.png")

        pca = PCA(n_components=None)
        if X.ndim == 3:
            pca.fit(X.reshape(X.shape[0] * X.shape[1], -1))
        else:
            pca.fit(X)
        expl_var = pca.explained_variance_[:n_components]
        plt.figure(figsize=(3, 2), dpi=130)
        plt.semilogx(expl_var, scores_at_threshold, color="tab:blue")
        plt.xlabel("PC explained variance")
        plt.ylabel(f"Score to reach $R^2_{{PC}} = {threshold}$")
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir / "scores_at_threshold_vs_expl_var.png")

        plt.figure(figsize=(3, 2), dpi=130)

        if color_log_scale:
            log_expl_var = np.log(expl_var)
            # normalize between 0 and 1
            log_expl_var = (log_expl_var - log_expl_var.min()) / (log_expl_var.max() - log_expl_var.min())
            colormap = plt.cm.get_cmap("plasma")
            colors = [colormap(log_expl_var[i]) for i in range(n_components)]
        else:
            colors = plt.cm.plasma(np.flip(np.linspace(0, 1, n_components)))

        for i in reversed(range(n_components)):
            color = colors[i]

            if plot_over_scores:
                plt.plot(scores, R2s[:, i], color=color)
            else:
                plt.plot(R2s[:, i], color=color)

            if plot_threshold_lines:
                plt.axvline(x=scores_at_threshold[i], color=color, linestyle='--')

        plt.xlabel("Score") if plot_over_scores else plt.xlabel("Iteration")
        plt.ylabel("PC R2")
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir / "R2_vs_score.png")

        plt.figure()
        plt.plot(scores)
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir / "score.png")

    return {
        "scores_at_threshold": scores_at_threshold,
        "R2_pc": R2s
    }


def pipeline_optim_score(dataset, measure, stop_score, decoder="logistic", labels=None, save_dir=None):
    """
    Args:
        dataset: tuple (X, condition) or dataset id (str). 
            X: np.ndarray of shape (n_steps, n_conditions, n_neurons)
            condition: list[dict] of length n_conditions
        labels: list of str, labels to decode. If None, all labels are decoded.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print("Saving to", save_dir)

    if isinstance(measure, (list, tuple)):
        results = []
        for m in measure:
            # TODO: measure id when measure is not str for save_dir
            res = pipeline_optim_score(dataset, m, stop_score, decoder, save_dir=save_dir / m)
            results.append(res)
            print(res)
        results_df = pd.concat(results)
        print(results_df)
        if save_dir:
            save_name = dataset if isinstance(dataset, str) else "score_vs_decoding_acc"
            results_df.to_csv(save_dir / f"{save_name}.csv")
        return

    if isinstance(dataset, str):
        dataset_id = dataset
        dataset = Dataset(dataset)
    else:
        dataset_id = None
    X, conditions = dataset

    if isinstance(measure, str):
        measure_id = measure
        measure = Measure(measure)
    else:
        # TODO
        measure_id = None

    if isinstance(decoder, str):
        if decoder == "logistic":
            decoder = decoder_logistic
        else:
            raise ValueError("Unknown decoder {}".format(decoder))

    res = fit_measure(data=X, measure=measure, stop_crit=stop_score)
    Ys, scores = res["fitted_datasets"], res["scores"]

    pc_captured_variance(X, Ys, scores, save_dir=save_dir)

    labels = conditions[0].keys() if labels is None else labels
    print(conditions)
    print(labels)

    decoding_acc_results = {}
    ref_acc_results = {}
    for label in labels:
        cond = [c[label] for c in conditions]

        ref_acc = decoder(X, cond)["score"]
        ref_acc = np.mean(ref_acc)  # mean over time dimension
        print("Reference accuracy for label", label, ":", ref_acc)

        decoding_acc = []
        for Y in Ys:
            acc = decoder(Y, cond)["score"]
            acc = np.mean(acc)  # mean over time dimension
            decoding_acc.append(acc)

        decoding_acc_results[label] = decoding_acc
        ref_acc_results[label] = ref_acc

    with plt.rc_context(rcparams):
        plt.figure()
        for label, acc in decoding_acc_results.items():
            plt.plot(scores, acc, label=label)
            color = plt.gca().lines[-1].get_color()
            plt.axhline(y=ref_acc_results[label], linestyle="--", color=color)

        plt.xlabel("Score")
        plt.ylabel("Decoding accuracy")
        plt.tight_layout()

        if save_dir is not None:
            plt.savefig(save_dir / "score_vs_decoding_acc.png")

    # normalize decoding accuracy by reference accuracy
    normalized_decoding_acc = {
        label: np.array(acc) / ref_acc
        for label, acc in decoding_acc_results.items()
    }
    # average across labels
    avg_decoding_acc = np.mean(list(normalized_decoding_acc.values()), axis=0)

    res_df = pd.DataFrame({
        "score": scores,
        "decoding_accuracy": avg_decoding_acc,
        "measure": [measure_id] * len(scores),
        "dataset": [dataset_id] * len(scores),
    })
    if save_dir:
        res_df.to_csv(save_dir / "score_vs_decoding_acc.csv")

    return res_df
