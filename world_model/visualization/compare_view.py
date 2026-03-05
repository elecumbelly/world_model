"""Side-by-side comparison of real vs dream trajectories."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_comparison_grid(
    real_obs: list[np.ndarray],
    dream_obs: list[np.ndarray],
    num_frames: int = 10,
    save_path: str | None = None,
):
    """Plot a grid of real (top) vs dream (bottom) frames.

    Args:
        real_obs: list of (H, W, 3) uint8 real observation frames
        dream_obs: list of (H, W, 3) uint8 dream observation frames
        num_frames: how many frames to show
    """
    n = min(num_frames, len(real_obs), len(dream_obs))
    step = max(1, len(real_obs) // n)
    indices = list(range(0, len(real_obs), step))[:n]

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        axes[0, i].imshow(real_obs[idx])
        axes[0, i].axis("off")
        axes[0, i].set_title(f"t={idx}", fontsize=8)

        dream_idx = min(idx, len(dream_obs) - 1)
        axes[1, i].imshow(dream_obs[dream_idx])
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Real", fontsize=12)
    axes[1, 0].set_ylabel("Dream", fontsize=12)
    plt.suptitle("Real vs Dream Trajectory")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def save_comparison_gif(
    real_obs: list[np.ndarray],
    dream_obs: list[np.ndarray],
    save_path: str = "comparison.gif",
    fps: int = 5,
):
    """Save an animated GIF showing real (left) vs dream (right) side by side.

    Args:
        real_obs: list of (H, W, 3) uint8 frames
        dream_obs: list of (H, W, 3) uint8 frames
        save_path: output path for GIF
        fps: frames per second
    """
    n_frames = min(len(real_obs), len(dream_obs))

    fig, (ax_real, ax_dream) = plt.subplots(1, 2, figsize=(6, 3))
    ax_real.set_title("Real")
    ax_dream.set_title("Dream")
    ax_real.axis("off")
    ax_dream.axis("off")

    im_real = ax_real.imshow(real_obs[0])
    im_dream = ax_dream.imshow(dream_obs[0])
    title = fig.suptitle("Step 0")

    def update(frame):
        im_real.set_data(real_obs[frame])
        im_dream.set_data(dream_obs[frame])
        title.set_text(f"Step {frame}")
        return im_real, im_dream, title

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=True)
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
