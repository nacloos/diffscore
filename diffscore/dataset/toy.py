from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def animate_data(X, Ys, scores, colors=None, interval=1, fps=100, frame_step=1, show_title=True, save_path=None):
    def plot_helper(X):
        lines = []
        for cond in range(X.shape[1]):
            color = colors[cond] if colors is not None else None
            line, = plt.plot(X[:, cond, 0], X[:, cond, 1], color=color)
            lines.append(line)
        plt.axis('off')
        return lines

    def _animate_data(Ys, save_path):
        fig = plt.figure(figsize=(1, 1), dpi=300)
        # ax = plt.subplot(1, 2, 1, adjustable='box', aspect='equal')
        # plot_helper(X)

        # plt.subplot(1, 2, 2, sharex=ax, sharey=ax, adjustable='box', aspect='equal')
        plt.subplot(1, 1, 1, adjustable='box', aspect='equal')
        Y_lines = plot_helper(Ys[-1])
        plt.tight_layout()

        def animate_fun(i):
            if show_title:
                # add score to title
                plt.suptitle("Similarity: {:.3f}".format(scores[i]))
            for cond, line in enumerate(Y_lines):
                Y = Ys[i]
                line.set_data(Y[:, cond, 0], Y[:, cond, 1])

        anim = animation.FuncAnimation(fig, animate_fun, frames=len(Ys), interval=interval, repeat=True)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            anim.save(save_path, writer='imagemagick', fps=fps)

    Ys = Ys[::frame_step]
    _animate_data(Ys, save_path)
    _animate_data([X for _ in range(len(Ys))], save_path.with_name("reference.gif"))


def toy2d(a=0.5):
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    datapoints = [[0.01, 0.01], [0.4, 0.3], [0.7, 0.5], [0.88, 0.613], [1.13, 0.68], [1.43, 0.73], [1.68, 0.6],
                  [1.86, 0.473], [1.97, 0.24], [2.02, 0.04]]
    X0 = np.array(datapoints)

    scaling1 = np.array([[1, 1+a]])
    scaling2 = np.array([[1, 1+2*a]])
    X_cond = [X0, X0*scaling1, X0*scaling2, X0@rotation_matrix(np.pi), X0*scaling1@rotation_matrix(np.pi),
              X0*scaling2@rotation_matrix(np.pi)]
    # X_cond = [Xi @ rotation_matrix(-2.5*np.pi/6) for Xi in X_cond]
    X = np.array(X_cond).transpose(1, 0, 2)  # time x condition x neuron

    conditions = [
        {"choice": -1, "stimulus": -3},
        {"choice": -1, "stimulus": -2},
        {"choice": -1, "stimulus": -1},
        {"choice": 1, "stimulus": 1},
        {"choice": 1, "stimulus": 2},
        {"choice": 1, "stimulus": 3},
    ]

    return X, conditions


def exp_gaussian(dim=100, n_steps=15, n_trials=40):
    # generate exp eigenspectrum
    # eigenvalues = np.random.exponential(scale=1.0, size=dim)
    eigenvalues = np.logspace(np.log10(0.0001), np.log10(1), dim)
    cov_matrix = np.diag(eigenvalues)
    # expl_var = np.sort(eigenvalues)[::-1]
    X = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov_matrix, size=(n_steps, n_trials))
    cond = np.arange(n_trials)
    return X, cond


def ultrametric(dim=50, n_branches=[2, 5, 10], gamma=0.8, n_levels=None, seed=None, center=False, normalize=False):
    rng = np.random.default_rng(seed=seed)
    if n_levels is None:
        n_levels = len(n_branches)

    def make_descendants(ancestor, n_descendants, gamma):
        # resample ancestor bits with probability (1-gamma)
        resampling_mask = rng.choice([0, 1], (n_descendants, len(ancestor)),
                                           p=[gamma, 1-gamma])
        descendants = np.repeat(ancestor[None, :], n_descendants, axis=0)
        descendants = (1-resampling_mask)*descendants + resampling_mask*rng.choice([0, 1], descendants.shape)
        return descendants

    def make_leaves(n_branches, n_levels, gamma):
        if not isinstance(n_branches, Iterable):
            n_branches = [n_branches]*n_levels
        else:
            assert len(n_branches) == n_levels, "n_branches must have length n_levels"

        ancestor = rng.choice([0, 1], size=dim)
        ancestors = [ancestor]

        for level in range(n_levels):
            all_descendants = []
            for ancestor in ancestors:
                descendants = make_descendants(ancestor, n_descendants=n_branches[level], gamma=gamma)
                all_descendants.extend(descendants)
            ancestors = all_descendants
        leaves = np.array(ancestors)
        return leaves

    X = make_leaves(n_branches, n_levels, gamma)
    X = X.astype(np.float32)

    # cond = np.eye(X.shape[0])
    # cond is the last but one branch (leaves within branch are different repetitions)
    # use np.repeat to repeat n_trials times identity of the last but one branch
    cond = np.repeat(np.arange(np.prod(n_branches[:-1])), X.shape[0]//np.prod(n_branches[:-1]))
    cond = [{"category": c} for c in cond]

    if center:
        X = X - X.mean(axis=0)[None]

    if normalize:
        X = X / np.linalg.norm(X, axis=-1)[:, None]

    return X, cond


def test_ultrametric():
    X, cond = ultrametric(dim=1000, n_branches=[2, 2, 3, 10], gamma=0.8)

    # mds
    from sklearn.manifold import MDS
    mds = MDS(n_components=2)
    mds_X = mds.fit_transform(X)

    plt.figure()
    # color by condition
    plt.scatter(mds_X[:, 0], mds_X[:, 1], c=cond)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    proj_X = pca.fit_transform(X)

    plt.figure()
    plt.scatter(proj_X[:, 0], proj_X[:, 1])

    plt.figure()
    plt.plot(cond)

    plt.figure()
    im = plt.imshow(np.corrcoef(X))
    plt.colorbar(im)
    plt.show()


if __name__ == "__main__":
    test_ultrametric()
