import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


def simulate_diffusion(n_particles=1000, n_steps=200, dt=0.1):
    """Simulate both isotropic and anisotropic diffusion."""
    # Isotropic diffusion (equal in all directions)
    iso_positions = np.zeros((n_steps, n_particles, 2))
    iso_positions[0] = np.random.randn(n_particles, 2) * 0.5

    # Anisotropic diffusion (different rates in x and y)
    aniso_positions = np.zeros((n_steps, n_particles, 2))
    aniso_positions[0] = np.random.randn(n_particles, 2) * 0.5

    # Define barriers for anisotropic diffusion - creating a tube (x1, y1, x2, y2)
    barriers = [
        (-12, -4, 12, -4),  # Bottom wall of tube
        (-12, 4, 12, 4),  # Top wall of tube
    ]

    # Diffusion coefficients
    D_iso = 1.0  # Isotropic diffusion coefficient
    D_aniso_x = 2.0  # Anisotropic x-direction
    D_aniso_y = 0.5  # Anisotropic y-direction

    def reflect_from_barriers(positions, new_positions):
        """Reflect particles that cross horizontal barriers."""
        for _, y_wall, _, _ in barriers:
            # Check if particle crossed this horizontal barrier
            crossed = (positions[:, 1] - y_wall) * (new_positions[:, 1] - y_wall) < 0
            new_positions[crossed, 1] = 2 * y_wall - new_positions[crossed, 1]
        return new_positions

    for step in range(1, n_steps):
        # Isotropic diffusion: random walk with equal variance in all directions
        iso_positions[step] = iso_positions[step - 1] + np.sqrt(
            2 * D_iso * dt
        ) * np.random.randn(n_particles, 2)

        # Anisotropic diffusion: different variances in x and y
        dx = np.sqrt(2 * D_aniso_x * dt) * np.random.randn(n_particles)
        dy = np.sqrt(2 * D_aniso_y * dt) * np.random.randn(n_particles)
        aniso_positions[step] = aniso_positions[step - 1] + np.column_stack([dx, dy])

        # Apply barrier reflections to anisotropic diffusion
        aniso_positions[step] = reflect_from_barriers(
            aniso_positions[step - 1], aniso_positions[step]
        )

    return iso_positions, aniso_positions, barriers


def main(show_signal_plots=True, show_axes=True):
    print("Generating diffusion animation...")

    # Simulate diffusion
    iso_pos, aniso_pos, barriers = simulate_diffusion(
        n_particles=3000, n_steps=200, dt=0.1
    )

    # Set up the figure based on whether signal plots are shown
    if show_signal_plots:
        fig = plt.figure(figsize=(12, 9))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[1.2, 1])
        ax1 = fig.add_subplot(gs[0, 0])  # Isotropic particles
        ax2 = fig.add_subplot(gs[0, 1])  # Anisotropic particles
        ax3 = fig.add_subplot(gs[1, 0])  # Isotropic signal
        ax4 = fig.add_subplot(gs[1, 1])  # Anisotropic signal
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax3 = None  # type: ignore
        ax4 = None  # type: ignore

    # Set limits
    limit = 15
    for ax in [ax1, ax2]:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if not show_axes:
            ax.axis("off")

    ax1.set_title("Isotropic Diffusion", fontsize=14, fontweight="bold")
    ax2.set_title(
        "Anisotropic Diffusion (with Barriers)", fontsize=14, fontweight="bold"
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    # Draw barriers on the right plot
    for x1, y1, x2, y2 in barriers:
        ax2.plot([x1, x2], [y1, y2], "k-", linewidth=3, alpha=0.8)

    # Initialize scatter plots with initial data
    scatter1 = ax1.scatter(
        iso_pos[0][:, 0], iso_pos[0][:, 1], c="blue", alpha=0.6, s=10
    )
    scatter2 = ax2.scatter(
        aniso_pos[0][:, 0], aniso_pos[0][:, 1], c="red", alpha=0.6, s=10
    )

    # Add step counter on ax1
    time_text = ax1.text(
        0.5, 1.08, "", ha="center", transform=ax1.transAxes, fontsize=12
    )

    # Initialize signal-related variables (will be set if show_signal_plots is True)
    time_steps: np.ndarray | None = None
    signal_iso_x: np.ndarray | None = None
    signal_iso_y: np.ndarray | None = None
    signal_aniso_x: np.ndarray | None = None
    signal_aniso_y: np.ndarray | None = None
    line_iso_x = None
    line_iso_y = None
    line_aniso_x = None
    line_aniso_y = None

    if show_signal_plots:
        # Calculate signal attenuation S(t) due to diffusion
        # S(t) = |mean(exp(i * k * r(t)))| where k is a wave vector
        k_value = 0.5  # Gradient strength
        time_steps = np.arange(len(iso_pos)) * 0.1

        # Calculate phase accumulation and signal (vectorized)
        signal_iso_x = np.abs(np.mean(np.exp(1j * k_value * iso_pos[:, :, 0]), axis=1))
        signal_iso_y = np.abs(np.mean(np.exp(1j * k_value * iso_pos[:, :, 1]), axis=1))
        signal_aniso_x = np.abs(
            np.mean(np.exp(1j * k_value * aniso_pos[:, :, 0]), axis=1)
        )
        signal_aniso_y = np.abs(
            np.mean(np.exp(1j * k_value * aniso_pos[:, :, 1]), axis=1)
        )

        # Setup signal plots
        assert ax3 is not None and ax4 is not None
        for ax, title in [(ax3, "Isotropic"), (ax4, "Anisotropic")]:
            ax.set(
                xlim=(0, time_steps[-1]),
                ylim=(0, 1.1),
                xlabel="Time",
                ylabel="Signal S(t)",
            )
            ax.set_title(f"{title} Diffusion Signal", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

        # Plot background curves and initialize animated lines
        ax3.plot(time_steps, signal_iso_x, "b--", alpha=0.3, linewidth=1)
        ax3.plot(time_steps, signal_iso_y, "g--", alpha=0.3, linewidth=1)
        ax4.plot(time_steps, signal_aniso_x, "r--", alpha=0.3, linewidth=1)
        ax4.plot(time_steps, signal_aniso_y, "m--", alpha=0.3, linewidth=1)

        (line_iso_x,) = ax3.plot([], [], "b-", linewidth=2, label="X")
        (line_iso_y,) = ax3.plot([], [], "g-", linewidth=2, label="Y")
        (line_aniso_x,) = ax4.plot([], [], "r-", linewidth=2, label="X")
        (line_aniso_y,) = ax4.plot([], [], "m-", linewidth=2, label="Y")

        ax3.legend(loc="upper right", fontsize=9)
        ax4.legend(loc="upper right", fontsize=9)

    def animate(frame):
        scatter1.set_offsets(iso_pos[frame])
        scatter2.set_offsets(aniso_pos[frame])
        time_text.set_text(f"Step: {frame}")

        artists = [scatter1, scatter2, time_text]

        if show_signal_plots:
            assert all(
                x is not None
                for x in [
                    time_steps,
                    signal_iso_x,
                    signal_iso_y,
                    signal_aniso_x,
                    signal_aniso_y,
                    line_iso_x,
                    line_iso_y,
                    line_aniso_x,
                    line_aniso_y,
                ]
            )
            t = time_steps[: frame + 1]
            line_iso_x.set_data(t, signal_iso_x[: frame + 1])
            line_iso_y.set_data(t, signal_iso_y[: frame + 1])
            line_aniso_x.set_data(t, signal_aniso_x[: frame + 1])
            line_aniso_y.set_data(t, signal_aniso_y[: frame + 1])
            artists.extend([line_iso_x, line_iso_y, line_aniso_x, line_aniso_y])

        return artists

    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=len(iso_pos), interval=50, blit=True, repeat=True
    )
    print("Displaying animation... Close the window to exit.")
    plt.show()

    # Save animation as MP4
    print("Saving animation to diffusion_animation.mp4...")
    anim.save("diffusion_animation.mp4", writer="ffmpeg", fps=20, dpi=100)
    print("Animation saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion animation")
    parser.add_argument(
        "--no-signal-plots", action="store_true", help="Disable signal plots"
    )
    parser.add_argument("--no-axes", action="store_true", help="Hide axes")
    args = parser.parse_args()
    main(show_signal_plots=not args.no_signal_plots, show_axes=not args.no_axes)
