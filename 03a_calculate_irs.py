import numpy as np
from scipy import signal
import sofar as sf
import matplotlib.pyplot as plt
from utils import metrics, interpolate
import pandas as pd
import os
import toml
import glob

config = {
    "fn_output": "faust_dsp/eval_anc_irs.h",
    "mode": "tr",  # 'tr' or 'rot'
    "precision": "float",  # 'float' or 'double'
    "dataset_path": "data/TASCAR_IRs/measured_irs_tr.npz",
    "spacing_fixpos": 15.0,  # spacing of the reference points, centimeters
    "step_patterns": "symmetricP2",  # Allowed step patterns for DTW
    "ir_range": [0, 64],  # taps of the IR to consider in analysis (was: range(0, 64))
    "export_results": True,
}


def load_npz_dataset(config: dict) -> tuple[np.ndarray, np.ndarray, float, range]:
    """Load dataset from npz file

    Parameters
    ----------
    config : dict

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

    ir_data = np.load(config["dataset_path"])
    irs = ir_data["irs"]
    positions = ir_data["positions"]
    fs = ir_data["fs"].item()

    if config["ir_range"][1] > irs.shape[2]:
        ir_range = range(config["ir_range"][0], irs.shape[2])
    else:
        ir_range = range(config["ir_range"][0], config["ir_range"][1])

    irs = irs[:, :, ir_range]

    return irs, positions, fs, ir_range


def main():

    irs, position, fs, ir_range = load_npz_dataset(config)

    label_pos = "Position (cm)" if config["mode"] == "tr" else "Angle (degrees)"

    # Plot IR dataset
    hrir_to_plot = np.abs(irs[:, 0, ir_range])
    hrir_to_plot /= np.max(hrir_to_plot)

    t_axis = np.arange(len(ir_range)) / fs * 1000  # in ms

    plt.figure()
    ax1 = plt.gca()
    plt.pcolormesh(
        position,
        t_axis,
        20 * np.log10(hrir_to_plot.T + 1e-12),
        shading="auto",
        cmap="Greys",
        vmin=-60,
        vmax=0,
    )
    plt.colorbar(label="Magnitude (dB)", pad=0.15)
    plt.title("IR dataset")
    plt.xlabel(label_pos)
    ax1.set_ylabel("Time (ms)")

    # Add second y-axis with samples
    ax2 = ax1.twinx()
    ax2.set_ylim(config["ir_range"][0] - 0.5, config["ir_range"][1] - 1.5)
    ax2.set_ylabel("Samples")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Select reference positions based on angle spacing
    ref_indices = np.where((position - position[0]) % config["spacing_fixpos"] < 1e-6)[
        0
    ]

    positions_fixed = position[ref_indices] / 100.0

    # We have a complete symmetric setup with G_0 = G_1, therefore we can use one side only
    irs_clean = np.squeeze(irs[ref_indices, 0])
    irs_clean = np.squeeze(irs_clean[..., ir_range])
    irs_lower_warped = np.zeros_like(irs_clean)
    irs_upper_warped = np.zeros_like(irs_clean)
    displacement_lower = np.zeros_like(irs_clean)
    displacement_upper = np.zeros_like(irs_clean)

    # Select desired step pattern for dtw
    step_pattern = interpolate.select_step_pattern(config["step_patterns"])

    # Outer loop - iterate over all fixed positions
    for ii in range(len(ref_indices) - 1):
        ir_pos0 = irs_clean[ii]
        ir_pos1 = irs_clean[ii + 1]

        irs_lower_warped[ii], displacement_lower[ii] = interpolate.calculate_dtw(
            ir_pos0, ir_pos1, step_pattern
        )
        irs_upper_warped[ii + 1], displacement_upper[ii + 1] = (
            interpolate.calculate_dtw(ir_pos1, ir_pos0, step_pattern)
        )

    # Export (pre-warped and clean) IRs as C++ header as std::vector<std::vector<precision>>
    if config["export_results"]:
        with open(config["fn_output"], "w") as f:
            f.write(
                f"// Auto-generated FAUST DSP header file containing pre-warped IRs\n"
            )
            f.write(f"// Precision: {config['precision']}\n\n")
            f.write("#include <vector>\n\n")

            # Export irs_clean
            f.write(f"std::vector<std::vector<{config['precision']}>> irs_clean = {{\n")
            for ir in irs_clean:
                ir_str = ", ".join([f"{sample:.8e}f" for sample in ir])
                f.write(f"    {{{ir_str}}},\n")
            f.write("};\n\n")

            # Export irs_lower_warped
            f.write(
                f"std::vector<std::vector<{config['precision']}>> irs_lower_warped = {{\n"
            )
            for ir in irs_lower_warped:
                ir_str = ", ".join([f"{sample:.8e}f" for sample in ir])
                f.write(f"    {{{ir_str}}},\n")
            f.write("};\n\n")

            # Export irs_upper_warped
            f.write(
                f"std::vector<std::vector<{config['precision']}>> irs_upper_warped = {{\n"
            )
            for ir in irs_upper_warped:
                ir_str = ", ".join([f"{sample:.8e}f" for sample in ir])
                f.write(f"    {{{ir_str}}},\n")
            f.write("};\n\n")

            # Export displacement_lower
            f.write(
                f"std::vector<std::vector<{config['precision']}>> displacement_lower = {{\n"
            )
            for disp in displacement_lower:
                disp_str = ", ".join([f"{sample:.8e}f" for sample in disp])
                f.write(f"    {{{disp_str}}},\n")
            f.write("};\n\n")

            # Export displacement_upper
            f.write(
                f"std::vector<std::vector<{config['precision']}>> displacement_upper = {{\n"
            )
            for disp in displacement_upper:
                disp_str = ", ".join([f"{sample:.8e}f" for sample in disp])
                f.write(f"    {{{disp_str}}},\n")
            f.write("};\n")

    # plt.show()


if __name__ == "__main__":
    main()
