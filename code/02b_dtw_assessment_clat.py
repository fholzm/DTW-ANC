import numpy as np
from scipy import signal
import sofar as sf
import dtw
import matplotlib
import matplotlib.pyplot as plt
from utils import interpolate
import os
import toml

matplotlib.use("Agg")  # Use non-interactive backend for figure generation

# Configure figure defaults --> 8 pt Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 1 * 8
plt.rcParams["axes.titlesize"] = 1 * 8
plt.rcParams["axes.labelsize"] = 1 * 8
plt.rcParams["xtick.labelsize"] = 1 * 8
plt.rcParams["ytick.labelsize"] = 1 * 8
plt.rcParams["legend.fontsize"] = 1 * 8


def load_sofa_dataset(config: dict) -> tuple[np.ndarray, np.ndarray, float, range]:
    """Load dataset from SOFA file

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    irs : np.ndarray
        Impulse responses array
    positions : np.ndarray
        Positions array
    fs : float
        Sampling frequency
    ir_range : range
        Range of impulse response taps considered
    """
    sofa_data = sf.read_sofa(config["dataset_path"], "r")
    irs = sofa_data.Data_IR
    positions = 360 - sofa_data.SourcePosition[:, 0]  # Convert to intrinsic rotation
    fs = sofa_data.Data_SamplingRate

    # Sort by angle
    sort_indices = np.argsort(positions)
    irs = irs[sort_indices, ...]
    positions = np.round(positions[sort_indices]).astype(int)

    # Duplicate 360° to 0°
    if positions[0] != 0:
        irs = np.concatenate((irs[[-1]], irs), axis=0)
        positions = np.concatenate((np.array([0]), positions), axis=0)

    if config["ir_ds_factor"] > 1:
        irs_ds = signal.decimate(irs, config["ir_ds_factor"], axis=2, zero_phase=True)
        irs = irs_ds
        fs /= config["ir_ds_factor"]

    if config["ir_range"][1] > irs.shape[2]:
        ir_range = range(config["ir_range"][0], irs.shape[2])
    else:
        ir_range = range(config["ir_range"][0], config["ir_range"][1])

    irs = irs[:, :, ir_range]

    return irs, positions, fs, ir_range


if __name__ == "__main__":
    config = toml.load("configs/eval_spacing_rot.toml")
    os.makedirs(config["fn_figure_dir"], exist_ok=True)

    irs, position, fs, ir_range = load_sofa_dataset(config)

    # Extract IRs at evaluation positions
    angle_eval = [110, 90]

    ir0 = irs[position == angle_eval[0], 0, :][0]
    ir1 = irs[position == angle_eval[1], 0, :][0]

    # Calculate and plot DTW alignment
    step_pattern = interpolate.select_step_pattern(config["step_patterns"][0])
    ax = dtw.dtw(ir0, ir1, step_pattern=step_pattern, keep_internals=True).plot(
        type="twoway", offset=-0.05
    )

    # Update figure properties
    fig = ax.figure
    fig.set_size_inches(3.5, 1.4)

    # Axis labels and title
    ax.set_xlabel("Taps")
    ax.set_ylabel(f"Coefficients at {angle_eval[0]}°")

    right_ax = fig.axes[1]
    right_ax.set_ylabel(f"Coefficients at {angle_eval[1]}°", color="tab:blue")
    right_ax.tick_params(axis="y", colors="tab:blue")

    # Limits / grid / ticks
    ax.set_xlim(0, len(ir0))
    tick_locator = matplotlib.ticker.MultipleLocator(0.025)
    ax.yaxis.set_major_locator(tick_locator)
    right_ax.yaxis.set_major_locator(tick_locator)

    # Legend
    ax.legend(
        [f"{angle_eval[0]}°", f"{angle_eval[1]}°", "Alignment"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        borderaxespad=0,
        frameon=False,
    )

    # Layout and save/show
    fig.tight_layout(pad=0.1)

    if config["export_figures"]:
        plt.savefig("../results/figures/holzm3.pdf")
