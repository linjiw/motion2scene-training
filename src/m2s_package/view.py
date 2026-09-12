"""CPU trajectory preview for native Motion2Scene reference NPZ files."""


def render(motion, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with np.load(motion, allow_pickle=False) as data:
        qpos = data["qpos"].copy()
        time = data["time_s"].copy()
    if qpos.ndim != 2 or len(time) != len(qpos) or not np.isfinite(qpos).all():
        raise ValueError("Expected finite T,D qpos and matching time_s")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(qpos[:, 0], qpos[:, 1])
    axes[0].scatter(qpos[[0, -1], 0], qpos[[0, -1], 1], c=["green", "red"])
    axes[0].set(
        xlabel="World x (m)", ylabel="World y (m)", title="Root trajectory: start / end"
    )
    axes[0].axis("equal")
    axes[1].plot(time, qpos[:, 2])
    axes[1].set(xlabel="Time (s)", ylabel="Root height (m)", title="Reference height")
    fig.suptitle(motion.name + " — reference data, not a policy rollout")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
