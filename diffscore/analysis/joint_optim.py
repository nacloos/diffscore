from collections import defaultdict
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

import diffscore
import diffscore.analysis.similarity_measures


save_dir = Path(__file__).parent / "results" / "joint_optim"
save_dir.mkdir(parents=True, exist_ok=True)


def joint_optim(measure1, measure2, X0, Y0, target_score1, target_score2, lr, n_iter):
    X = torch.tensor(X0, dtype=torch.float32, requires_grad=True)
    Y = torch.tensor(Y0, dtype=torch.float32, requires_grad=True)
    
    optimizer = torch.optim.Adam([X, Y], lr=lr)
    
    history = defaultdict(list)
    
    history['X'].append(X0.copy())
    history['Y'].append(Y0.copy())
    
    for i in range(n_iter):
        optimizer.zero_grad()

        score1 = measure1(X, Y)
        score2 = measure2(X, Y)
        
        # loss = (score1 - target_score1)**2 + (score2 - target_score2)**2
        # converges closer to target with absolute difference compared to squared difference
        loss = torch.abs(score1 - target_score1) + torch.abs(score2 - target_score2)
        
        history['score1'].append(score1.item())
        history['score2'].append(score2.item())
        history['loss'].append(loss.item())
        history['X'].append(X.detach().numpy().copy())
        history['Y'].append(Y.detach().numpy().copy())
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item():.6f}")
    
    history = {
        'score1': history['score1'],
        'score2': history['score2'],
        'loss': history['loss'],
        'X': history['X'],
        'Y': history['Y']
    }

    return X.detach().numpy(), Y.detach().numpy(), history


def joint_optim_square(measure1, measure2, X0, Y0, N, lr, n_iter, save_dir=None):
    points = np.linspace(0, 1, N)
    # don't include (1, 1) because of svd error
    targets = [(target, 1) for target in points[:-1]] + \
        [(1, target) for target in reversed(points[:-1])] + \
        [(target, 0) for target in reversed(points[1:])] + \
        [(0, target) for target in points[1:]]
    # add (0, 0)
    targets = [(0, 0)] + targets
    # add (0.99, 0.99)
    targets = targets + [(0.99, 0.99)]

    results = []
    for target in targets:
        print(f"Target: {target}")
        X, Y, history = joint_optim(measure1, measure2, X0, Y0, target[0], target[1], lr, n_iter)
        results.append(history)
    
    return targets, results


def plot_joint_optim_square(measure1, measure2, X0, Y0, N, lr, n_iter, save_dir=None):
    if isinstance(measure1, str):
        measure1_id = measure1
        measure1 = diffscore.make(f"measure/{measure1}")
    else:
        measure1_id = "Measure1"

    if isinstance(measure2, str):
        measure2_id = measure2
        measure2 = diffscore.make(f"measure/{measure2}")
    else:
        measure2_id = "Measure2"

    if save_dir is None or not (save_dir / "results.pkl").exists():
        targets, results = joint_optim_square(measure1, measure2, X0, Y0, N, lr, n_iter, save_dir)

        # save results
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / "results.pkl", "wb") as f:
                pickle.dump({
                    "targets": targets,
                    "results": results
                }, f)
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "results.pkl", "rb") as f:
            data = pickle.load(f)
            targets = data["targets"]
            results = data["results"]

    plt.figure(figsize=(3, 3), dpi=300)
    for result in results:
        plt.plot(result['score1'], result['score2'], color="cornflowerblue")

    for target in targets:
        plt.scatter(target[0], target[1], color="coral", marker=".")

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(f'{measure1_id}')
    plt.ylabel(f'{measure2_id}')
    plt.axis('equal')

    plt.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{measure1_id}-{measure2_id}.png", dpi=300)
    else:
        plt.show()


def compute_slopes(measure1, measure2, X0, Y0, lr, n_iter, save_dir=None):
    targets = [(0.95, 0.0), (0.0, 0.95)]
    measure1 = diffscore.make(f"measure/{measure1_id}")
    measure2 = diffscore.make(f"measure/{measure2_id}")

    if save_dir is None or not (save_dir / f"results.pkl").exists():
        results = []
        for target in targets:
            print(f"Target: {target}")
            X, Y, history = joint_optim(measure1, measure2, X0, Y0, target[0], target[1], lr, n_iter)
            results.append(history)

        # save results
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            with open(save_dir / f"results.pkl", "wb") as f:
                pickle.dump(results, f)
    else:
        with open(save_dir / f"results.pkl", "rb") as f:
            results = pickle.load(f)

    # Calculate and print slopes of the dashed lines
    last_scores = [(result['score1'][-1], result['score2'][-1]) for result in results]
    slope1 = (1 - last_scores[0][1]) / (1 - last_scores[0][0])
    slope2 = (1 - last_scores[1][1]) / (1 - last_scores[1][0])

    if save_dir is not None:
        plt.figure(figsize=(3, 3), dpi=300)
        for result in results:
            plt.plot(result['score1'], result['score2'], color="cornflowerblue")
        for target in targets:
            plt.scatter(target[0], target[1], color="coral", marker=".")

        # add marker for init scores
        plt.scatter(results[0]['score1'][0], results[0]['score2'][0], color="cornflowerblue", marker="o")

        # draw line connecting last scores for horizontal targets and vertical targets
        # plt.plot([last_scores[0][0], last_scores[1][0]], [last_scores[0][1], last_scores[1][1]], color="black", linestyle="--")
        # plt.plot([last_scores[2][0], last_scores[3][0]], [last_scores[2][1], last_scores[3][1]], color="black", linestyle="--")
        # draw line connecting each pair of last scores and (1, 1)
        plt.plot([last_scores[0][0], 1], [last_scores[0][1], 1], color="black", linestyle="--")
        plt.plot([last_scores[1][0], 1], [last_scores[1][1], 1], color="black", linestyle="--")

        # add text for slopes in bottom left corner
        plt.text(0.05, 0.05, f"Slope {measure1_id}: {slope1:.3f}", ha='left', va='bottom', fontsize=6)
        plt.text(0.05, 0.0, f"Slope {measure2_id}: {slope2:.3f}", ha='left', va='bottom', fontsize=6)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel(f'{measure1_id}')
        plt.ylabel(f'{measure2_id}')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(save_dir / f"{measure1_id}-{measure2_id}.png", dpi=300)


    return slope1, slope2


if __name__ == "__main__":
    lr = 0.01
    n_iter = 1000
    N = 5


    shape = (20, 10)
    # shape = (20, 2)
    X0, Y0 = np.random.randn(*shape), np.random.randn(*shape)

    save_dir = save_dir / f"lr={lr}-n_iter={n_iter}-N={N}-shape={shape}"

    measure_ids = [
        # ("cka", "procrustes-angular-score"),
        # # ("cka", "cka-kernel=linear-hsic=lange-score")
        # ("cka-kernel=linear-hsic=gretton-score", "cka-kernel=linear-hsic=lange-score"),
        # ("nbs", "procrustes-angular_score"),
        # ("nbs", "procrustes-score"),
        ("cca-score", "procrustes-angular_score"),
        # ("cca-squared_score", "procrustes-angular_score"),
        # (
        #     "kernel=(centered-whitened-linear)-similarity=cosine-score",
        #     "kernel=(centered-linear)-similarity=cosine-score"
        # ),
        # ("cka", "kernel=linear-similarity=(centered-cosine)-score"),

        # ("cka", "rdm=squared_euclidean-similarity=cosine-score"),
        # ("cka", "rdm=squared_euclidean-similarity=(centered-cosine)-score"),
        # ("rdm=squared_euclidean-similarity=(centered-cosine)-score", "rdm=squared_euclidean-similarity=cosine-score"),

        # (
        #     "kernel=linear-similarity=(centered-cosine)-angular_score",
        #     "kernel=linear-similarity=(centered-bures)-angular_score"
        # ),
        # (
        #     "kernel=linear-similarity=(centered-cosine)-score",
        #     "kernel=linear-similarity=(centered-bures)-angular_score"
        # ),
        # (
        #     "rdm=correlation-similarity=cosine-score",
        #     "rdm=squared_euclidean-similarity=cosine-score"
        # ),
        # {
        #     "rdm=squared_euclidean-similarity=cosine-score",
        #     "rdm=squared_euclidean-similarity=upper_triangular_correlation-score"
        # },
        # {
        #     "rdm=squared_euclidean-similarity=(centered-zero_diagonal-cosine)-score",
        #     "rdm=squared_euclidean-similarity=upper_triangular_correlation-score"
        # },


        # (
        #     "rdm=squared_euclidean-similarity=cosine-score",
        #     "rdm=correlation-similarity=cosine-score"
        # ),
    ]
    for measure1_id, measure2_id in measure_ids:
        _save_dir = save_dir / f"{measure1_id}-{measure2_id}"


        plot_joint_optim_square(measure1_id, measure2_id, X0, Y0, N, lr, n_iter, _save_dir / "square")
        
        slope1, slope2 = compute_slopes(measure1_id, measure2_id, X0, Y0, lr, n_iter, _save_dir / "slopes")
        print(f"Slope of line 1: {slope1:.3f}")
        print(f"Slope of line 2 (inverse): {1/slope2:.3f}")
