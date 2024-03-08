from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import animation


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
