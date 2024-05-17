from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

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
from diffscore.training.measure_optim import fit_measure, OptimResult


rcparams = {
    "figure.figsize": (3, 2),
    "figure.dpi": 130,
    "axes.spines.right": False,
    "axes.spines.top": False,
}


def projected_r2(X, Y, orth=False):
    if X.ndim == 3:
        n_steps, n_trials = X.shape[0], X.shape[1]
        X = X.reshape(n_steps * n_trials, X.shape[-1])
        Y = Y.reshape(n_steps * n_trials, Y.shape[-1])

    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    # zero padding
    X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=True)

    assert X.shape[0] >= X.shape[1]

    if not orth:
        # find linear alignment from Y to X (lin reg)
        Q, R = np.linalg.qr(Y)
        X_pred = Q @ (Q.T @ X)
    else:
        # find orthogonal alignment from Y to X
        U, S, Vh = np.linalg.svd(X.T @ Y, full_matrices=False)
        Q = Vh.T @ U
        X_pred = Y @ Q

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


def pc_captured_variance(
    X,
    Ys,
    scores,
    n_components=None,
    threshold: float | str = "half",
    plot_threshold_lines=False,
    color_log_scale=True,
    expl_var_ratio=False,
    orth=False,
    save_dir=None
):
    R2s = []
    for Y in Ys:
        res = projected_r2(X, Y, orth=orth)
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
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / "scores_at_threshold.png")

        pca = PCA(n_components=None)
        if X.ndim == 3:
            pca.fit(X.reshape(X.shape[0] * X.shape[1], -1))
        else:
            pca.fit(X)

        if expl_var_ratio:
            expl_var = pca.explained_variance_ratio_[:n_components]
        else:
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

        if color_log_scale:
            log_expl_var = np.log(expl_var)
            # normalize between 0 and 1
            log_expl_var = (log_expl_var - log_expl_var.min()) / (log_expl_var.max() - log_expl_var.min())
            colormap = plt.colormaps["plasma"]
            colors = [colormap(log_expl_var[i]) for i in range(n_components)]
        else:
            colors = plt.cm.plasma(np.flip(np.linspace(0, 1, n_components)))

        plt.figure(figsize=(3, 2), dpi=130)
        for i in reversed(range(n_components)):
            color = colors[i]
            plt.plot(scores, R2s[:, i], color=color)
            if plot_threshold_lines:
                plt.axvline(x=scores_at_threshold[i], color=color, linestyle='--')
        plt.xlabel("Score")
        plt.ylabel("PC R2")
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir / "R2_vs_score.png")


        plt.figure(figsize=(3, 2), dpi=130)
        for i in reversed(range(n_components)):
            color = colors[i]
            plt.plot(R2s[:, i], color=color)
            if plot_threshold_lines:
                plt.axvline(x=scores_at_threshold[i], color=color, linestyle='--')
        plt.xlabel("Iteration")
        plt.ylabel("PC R2")
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir / "R2_vs_iteration.png")


        plt.figure()
        plt.plot(scores)
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir / "score.png")

    return {
        "scores_at_threshold": np.array(scores_at_threshold),
        "pc_captured_variance": R2s,
        "pc_explained_variance": expl_var
    }


def pipeline_optim_score(dataset, measure, stop_score, decoder="logistic", conditions=None, labels=None, decoding_analysis=True, save_dir=None, **kwargs):
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
        # store results for each measure
        pc_results = []
        decoding_results = []
        for m in measure:
            # TODO: measure id when measure is not str for save_dir
            fit_res, pc_res, decoding_res = pipeline_optim_score(dataset, m, stop_score, decoder, save_dir=save_dir / m, **kwargs)
            # results for scores to reach R2 threshold
            pc_results.append(pc_res)
            # results for decoding accuracy
            decoding_results.append(decoding_res)

        pc_results_df = pd.concat(pc_results)
        decoding_results_df = pd.concat(decoding_results)
        print(decoding_results_df)
        if save_dir:
            # save_name = dataset if isinstance(dataset, str) else "score_vs_decoding_acc"
            decoding_results_df.to_csv(save_dir / "scores_vs_decoding_acc.csv")
            pc_results_df.to_csv(save_dir / "scores_at_threshold.csv")

        return pc_results_df, decoding_results_df

    if isinstance(dataset, str):
        dataset_id = dataset
        dataset = Dataset(dataset)
    else:
        dataset_id = None

    # TODO: simplify API
    if isinstance(dataset, tuple):
        X, conditions = dataset
    else:
        X = dataset

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

    fit_res = fit_measure(dataset=X, measure=measure, stop_crit=stop_score, **kwargs)
    Ys, scores = fit_res["fitted_datasets"], fit_res["scores"]

    # TODO: give very low R2
    # pc_res = pc_captured_variance(
    #     X, Ys, scores, expl_var_ratio=True, 
    #     orth=True, save_dir=save_dir / "orth" if save_dir else None
    # )

    pc_res = pc_captured_variance(X, Ys, scores, expl_var_ratio=True, save_dir=save_dir)
    pc_res_df = pd.DataFrame({
        "scores_at_threshold": pc_res["scores_at_threshold"],
        "pc_explained_variance": pc_res["pc_explained_variance"],
        "measure": [measure_id] * len(pc_res["scores_at_threshold"]),
        "dataset": [dataset_id] * len(pc_res["scores_at_threshold"]),
    })
    if save_dir:
        pc_res_df.to_csv(save_dir / "scores_at_threshold.csv")

    if not decoding_analysis:
        return fit_res, pc_res_df

    labels = conditions[0].keys() if labels is None else labels

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

    decoding_res_df = pd.DataFrame({
        "score": scores,
        "decoding_accuracy": avg_decoding_acc,
        "measure": [measure_id] * len(scores),
        "dataset": [dataset_id] * len(scores),
        # add individual decoding accuracy for each label
        # use convention "decode.{label}" for columns corresponding to decoded task variables
        **{f"decode.{label}": acc for label, acc in decoding_acc_results.items()}
    })
    if save_dir:
        decoding_res_df.to_csv(save_dir / "score_vs_decoding_acc.csv")

    return fit_res, pc_res_df, decoding_res_df


@dataclass
class PCResult:
    threshold_scores: np.ndarray
    pc_captured_variance: np.ndarray
    pc_variance: np.ndarray


def analyze_pc(res: OptimResult, save_dir=None) -> PCResult:
    pc_res = pc_captured_variance(res.X, res.Ys, res.scores, expl_var_ratio=True, save_dir=save_dir)
    return PCResult(
        threshold_scores=pc_res["scores_at_threshold"],
        pc_captured_variance=pc_res["pc_captured_variance"],
        pc_variance=pc_res["pc_explained_variance"]
    )
