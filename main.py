import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
        """Reflect particles that cross barriers."""
        for i in range(len(new_positions)):
            for x1, y1, x2, y2 in barriers:
                # Check if particle crossed a barrier
                old_pos = positions[i]
                new_pos = new_positions[i]

                # Line segment intersection and reflection
                if intersects_segment(old_pos, new_pos, (x1, y1), (x2, y2)):
                    # Reflect the particle
                    if x1 == x2:  # Vertical barrier
                        new_positions[i, 0] = 2 * x1 - new_pos[0]
                    elif y1 == y2:  # Horizontal barrier
                        new_positions[i, 1] = 2 * y1 - new_pos[1]
        return new_positions

    def intersects_segment(p1, p2, q1, q2):
        """Check if line segment p1-p2 intersects q1-q2."""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

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


def main():
    print("Generating diffusion animation...")

    # Simulate diffusion
    iso_pos, aniso_pos, barriers = simulate_diffusion(
        n_particles=3000, n_steps=200, dt=0.1
    )

    # Set up the figure with 2 rows: particles on top, signal plots below
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, height_ratios=[1.2, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Isotropic particles
    ax2 = fig.add_subplot(gs[0, 1])  # Anisotropic particles
    ax3 = fig.add_subplot(gs[1, 0])  # Isotropic signal
    ax4 = fig.add_subplot(gs[1, 1])  # Anisotropic signal

    # Set limits
    limit = 15
    for ax in [ax1, ax2]:
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

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

    # Calculate signal attenuation S(t) due to diffusion
    # S(t) = |mean(exp(i * k * r(t)))| where k is a wave vector
    k_value = 0.5  # Gradient strength
    time_steps = np.arange(len(iso_pos)) * 0.1

    # Calculate phase accumulation and signal for different gradient directions
    signal_iso_x = np.zeros(len(iso_pos))
    signal_iso_y = np.zeros(len(iso_pos))
    signal_aniso_x = np.zeros(len(aniso_pos))
    signal_aniso_y = np.zeros(len(aniso_pos))

    for i in range(len(iso_pos)):
        # Isotropic: gradient in x and y directions
        signal_iso_x[i] = np.abs(np.mean(np.exp(1j * k_value * iso_pos[i, :, 0])))
        signal_iso_y[i] = np.abs(np.mean(np.exp(1j * k_value * iso_pos[i, :, 1])))

        # Anisotropic: gradient in x and y directions
        signal_aniso_x[i] = np.abs(np.mean(np.exp(1j * k_value * aniso_pos[i, :, 0])))
        signal_aniso_y[i] = np.abs(np.mean(np.exp(1j * k_value * aniso_pos[i, :, 1])))

    # Setup signal plots
    ax3.set_xlim(0, time_steps[-1])
    ax3.set_ylim(0, 1.1)
    ax3.set_xlabel("Time", fontsize=11)
    ax3.set_ylabel("Signal S(t)", fontsize=11)
    ax3.set_title("Isotropic Diffusion Signal", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    ax4.set_xlim(0, time_steps[-1])
    ax4.set_ylim(0, 1.1)
    ax4.set_xlabel("Time", fontsize=11)
    ax4.set_ylabel("Signal S(t)", fontsize=11)
    ax4.set_title("Anisotropic Diffusion Signal", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    # Plot complete signal curves as background
    ax3.plot(
        time_steps, signal_iso_x, "b--", alpha=0.3, linewidth=1, label="Gradient in X"
    )
    ax3.plot(
        time_steps, signal_iso_y, "g--", alpha=0.3, linewidth=1, label="Gradient in Y"
    )
    ax4.plot(
        time_steps, signal_aniso_x, "r--", alpha=0.3, linewidth=1, label="Gradient in X"
    )
    ax4.plot(
        time_steps, signal_aniso_y, "m--", alpha=0.3, linewidth=1, label="Gradient in Y"
    )

    ax3.legend(loc="upper right", fontsize=9)
    ax4.legend(loc="upper right", fontsize=9)

    # Initialize animated signal lines
    (line_iso_x,) = ax3.plot([], [], "b-", linewidth=2, label="X direction")
    (line_iso_y,) = ax3.plot([], [], "g-", linewidth=2, label="Y direction")
    (line_aniso_x,) = ax4.plot([], [], "r-", linewidth=2, label="X direction")
    (line_aniso_y,) = ax4.plot([], [], "m-", linewidth=2, label="Y direction")

    # Add step counter on ax1
    time_text = ax1.text(
        0.5, 1.08, "", ha="center", transform=ax1.transAxes, fontsize=12
    )

    def animate(frame):
        scatter1.set_offsets(iso_pos[frame])
        scatter2.set_offsets(aniso_pos[frame])
        time_text.set_text(f"Step: {frame}")

        # Update signal plots up to current frame
        current_time = time_steps[: frame + 1]
        line_iso_x.set_data(current_time, signal_iso_x[: frame + 1])
        line_iso_y.set_data(current_time, signal_iso_y[: frame + 1])
        line_aniso_x.set_data(current_time, signal_aniso_x[: frame + 1])
        line_aniso_y.set_data(current_time, signal_aniso_y[: frame + 1])

        return (
            scatter1,
            scatter2,
            time_text,
            line_iso_x,
            line_iso_y,
            line_aniso_x,
            line_aniso_y,
        )

    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=len(iso_pos), interval=50, blit=True, repeat=True
    )

    # Save animation as MP4
    print("Saving animation to diffusion_animation.mp4...")
    anim.save("diffusion_animation.mp4", writer="ffmpeg", fps=20, dpi=100)
    print("Animation saved successfully!")

    print("Displaying animation... Close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
