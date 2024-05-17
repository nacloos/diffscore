# TODO: better name than optim_square?
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from diffscore import Measure, Dataset, fit_measure, pipeline_optim_score
from diffscore_exp import make


fig_dir = Path(__file__).parent / "figures" / "adversarial_optim_square" / "test"


def measure_target_scores_abs(measure1, measure2, target_score1, target_score2):
    def score(X, Y):
        # absolute value of score differences
        dist = torch.abs(measure1(X, Y) - target_score1) + torch.abs(measure2(X, Y) - target_score2)
        return -dist  # score is negative distance
    return score

def measure_target_scores_square(measure1, measure2, target_score1, target_score2):
    def score(X, Y):
        # square of score differences
        dist = (measure1(X, Y) - target_score1) ** 2 + (measure2(X, Y) - target_score2) ** 2
        return -dist  # score is negative distance
    return score



def optim_target_scores(measure1, measure2, X, target_score1, target_score2, Y0=None, n_iter=100, save_dir=None, **kwargs):
    measure = measure_target_scores_abs(measure1, measure2, target_score1, target_score2)
    # measure = measure_target_scores_square(measure1, measure2, target_score1, target_score2)

    # fit_res = fit_measure(
    #     dataset=X,
    #     measure=measure,
    #     stop_crit=np.inf,
    #     n_iter=n_iter,
    #     init_data_fit=Y0,
    #     **kwargs
    # )
    fit_res, _ = pipeline_optim_score(
        dataset=X,
        measure=measure,
        stop_score=np.inf,
        n_iter=n_iter,
        init_data_fit=Y0,
        decoding_analysis=False,
        save_dir=save_dir,
        **kwargs
    )

    X = fit_res["reference_dataset"]
    Ys = fit_res["fitted_datasets"]
    if Y0 is not None:
        assert np.allclose(Y0, Ys[0]), f"Y0={Y0} Ys[0]={Ys[0]}"
    Y0 = Ys[0]
    measure1_scores = np.array([measure1(X, Y) for Y in Ys])
    measure2_scores = np.array([measure2(X, Y) for Y in Ys])

    return fit_res | {
        "measure1_scores": measure1_scores,
        "measure2_scores": measure2_scores,
        "Y0": Y0
    }


def optim_square(measure1_id, measure2_id, dataset_id, N, n_iter, plot_nbs_bounds=False, save_dir=None, verbose=True, **kwargs):
    Y0 = None

    measure1 = Measure(measure1_id)
    measure2 = Measure(measure2_id)

    if "siegel15" in dataset_id:
        import diffscore_exp
        X, cond = diffscore_exp.make(f"data.{dataset_id}")
    else:
        X, cond = Dataset(dataset_id)

    # target_scores1 = np.linspace(0, 1, N)
    # target_scores2 = np.linspace(0, 1, N)
    # target_scores = np.array([
    #     [target_score1, target_score2] for target_score1 in target_scores1 for target_score2 in target_scores2
    # ])
    # # keep only edges
    # target_scores = target_scores[
    #     (target_scores[:, 0] == 0) | (target_scores[:, 0] == 1) | (target_scores[:, 1] == 0) | (target_scores[:, 1] == 1)
    # ]
    targets = np.linspace(0, 1, N)
    target_scores = [(target, 0) for target in targets] + \
        [(1, target) for target in targets] + \
        [(target, 1) for target in reversed(targets)] + \
        [(0, target) for target in reversed(targets)]

    Y0 = np.random.randn(*X.shape)
    results = []
    for target_score1, target_score2 in target_scores:
        if verbose:
            print(f"Optimization target: ({target_score1}, {target_score2})")
        _save_dir = save_dir / f"{target_score1}_{target_score2}" if save_dir is not None else None
        res = optim_target_scores(measure1, measure2, X, target_score1, target_score2, Y0=Y0, n_iter=n_iter, save_dir=_save_dir, **kwargs)
        Y0 = res["Y0"] if Y0 is None else Y0
        results.append(res)

    # plt.figure(figsize=(3, 3), dpi=150)
    plt.figure(figsize=(1.5, 1.5), dpi=150)
    for res in results:
        # plt.plot(res["measure1_scores"], res["measure2_scores"], color="lightgreen")
        plt.plot(res["measure1_scores"], res["measure2_scores"], color="cornflowerblue")

    # plot boundaries
    boundary1 = np.array([
        [res["measure1_scores"][-1], res["measure2_scores"][-1]] for res in results
    ])
    # plt.plot(boundary1[:, 0], boundary1[:, 1], color="#4a4a4a", linestyle="-", marker=".")
    # plt.plot(boundary1[:, 0], boundary1[:, 1], color="gray", linestyle="-", marker=".")
    plt.plot(boundary1[:, 0], boundary1[:, 1], color="black", linestyle="-", marker=".")

    # plot target scores
    for target_score1, target_score2 in target_scores:
        # plt.plot(target_score1, target_score2, marker=".", color="lightcoral")
        plt.plot(target_score1, target_score2, marker=".", color="#FFBBBB")

    # add identity line
    # plt.plot([0, 1], [0, 1], color="lightgray", zorder=-1)

    if measure1_id == "cka" and measure2_id == "nbs":
        # theoretical bounds
        def nbs_bounds(X, Y):
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], -1)
                Y = Y.reshape(Y.shape[0], -1)
            X = X - np.mean(X, axis=0)
            Y = Y - np.mean(Y, axis=0)
            Kx = X @ X.T
            Ky = Y @ Y.T
            rX = np.linalg.matrix_rank(Kx)
            rY = np.linalg.matrix_rank(Ky)
            cka_score = np.linalg.norm(X.T @ Y, "fro") ** 2 / (np.linalg.norm(Kx, "fro") * np.linalg.norm(Ky, "fro"))
            nbs_score = np.linalg.norm(X.T @ Y, "nuc") / np.sqrt(np.linalg.norm(Kx, "nuc") * np.linalg.norm(Ky, "nuc"))
            lower_bound = cka_score / np.sqrt(rX * rY)
            upper_bound = min(rX, rY) * cka_score
            # take sqrt to get bounds on NBS instead of NBS^2
            return np.sqrt(lower_bound), np.sqrt(upper_bound), nbs_score, cka_score

        Ys = [r["fitted_datasets"][-1] for r in results]
        for Y in Ys:
            lower_bound, upper_bound, nbs_score, cka_score = nbs_bounds(X, Y)
            assert np.allclose(measure1(X, Y), cka_score), f"{measure1(X, Y)} != {cka_score}"
            assert np.allclose(measure2(X, Y), nbs_score), f"{measure2(X, Y)} != {nbs_score}"
            assert lower_bound <= nbs_score <= upper_bound, f"{lower_bound} <= {nbs_score} <= {upper_bound}"

            if plot_nbs_bounds:
                plt.scatter(cka_score, lower_bound, color="tab:green", marker=".")
                plt.scatter(cka_score, upper_bound, color="tab:green", marker=".")

    # plt.xlabel(measure1_id)
    # plt.ylabel(measure2_id)
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    plt.axis("equal")
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])

    # vertical and horizontal lines at 0.9
    # color = "lightgray"
    color = "#595959"
    plt.axvline(0.9, ymin=-0.12, color=color, linestyle="--", zorder=10)
    plt.axhline(0.9, xmin=-0.105, color=color, linestyle="--", zorder=10)

    ax = plt.gca()
    ax.spines['left'].set_position(('data', -0.105))
    ax.spines['bottom'].set_position(('data', -0.12))
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_dir is not None:
        plt.savefig(save_dir / f"{measure1_id}_{measure2_id}_{dataset_id}.png")
        plt.savefig(save_dir / f"{measure1_id}_{measure2_id}_{dataset_id}.pdf")

    # for each target score, plot measure1_scores and measure2_scores vs iterations
    for i, res in enumerate(results):
        _save_dir = save_dir / f"{target_scores[i][0]}_{target_scores[i][1]}" if save_dir is not None else None
        plt.figure(figsize=(1.5, 1.5), dpi=150)
        plt.plot(res["measure1_scores"], color="cornflowerblue", label=measure1_id)
        plt.plot(res["measure2_scores"], color="coral", label=measure2_id)
        # plot target score as horizontal line
        plt.axhline(target_scores[i][0], color="cornflowerblue", linestyle="--")
        plt.axhline(target_scores[i][1], color="coral", linestyle="--")

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(_save_dir / f"{measure1_id}_{measure2_id}_{dataset_id}.png")
            plt.savefig(_save_dir / f"{measure1_id}_{measure2_id}_{dataset_id}.pdf")


from dataclasses import dataclass

@dataclass
class SquareResult:
    target_scores: np.ndarray
    scores: np.ndarray
    Ys: np.ndarray
    X: np.ndarray


# TODO?
@dataclass
class OptimParams:
    lr: float
    optimizer: str
    max_iter: int
    stop_score: float


def square(dataset, measure1, measure2, N, n_iter):
    optim_square(measure1, measure2, dataset, N, n_iter)





