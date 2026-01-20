import numpy as np
from scipy.interpolate import CubicSpline
from scipy import signal
import sofar as sf
import dtw
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils import metrics
import pandas as pd
import os
import toml
import argparse

# global variables
# config = {
#     "fn_output_prefix": "eval_spacing_",
#     "fn_result_dir": "results/",
#     "fn_figure_dir": "figures/",
#     "dataset_path": "data/THK_NF/HRIR_CIRC360_NF025.sofa",
#     "angle_spacings": [
#         5,
#         10,
#         15,
#         20,
#         30,
#         45,
#     ],  # spacing of the reference points, degrees
#     "interpolation_methods": ["direct", "nn", "dtw"],  # Methods to test
#     "step_patterns": ["symmetricP2"],  # Allowed step patterns for DTW
#     "ir_ds_factor": 6,  # downsampling factor for IRs
#     "ir_range": range(0, 32),  # taps of the IR to consider in analysis
#     "plot_sm_limits": [-40, 8],  # dB
#     "plot_mag_limits": [-30, 0],  # dB
#     "plot_nfft": 512,
#     "export_figures": True,
#     "export_results": True,
# }

# config = {
#     "fn_output_prefix": "eval_spacing_",
#     "fn_result_dir": "results/eval_tr/",
#     "fn_figure_dir": "figures/eval_tr/",
#     "dataset_path": "data/TASCAR_IRs/measured_irs_tr.npz",
#     "mode": "tr",
#     "spacing_fixpos": [
#         2.5,
#         5,
#         10,
#         15,
#         25,
#     ],  # spacing of the reference points, centimeters
#     "interpolation_methods": ["direct", "nn", "dtw"],  # Methods to test
#     "step_patterns": ["symmetricP2"],  # Allowed step patterns for DTW
#     "ir_ds_factor": 6,  # downsampling factor for IRs
#     "ir_range": range(0, 64),  # taps of the IR to consider in analysis
#     "plot_sm_limits": [-40, 8],  # dB
#     "plot_mag_limits": [-30, 0],  # dB
#     "plot_nfft": 512,
#     "export_figures": True,
#     "export_results": True,
# }

# config = {
#     "fn_output_prefix": "eval_steppattern_",
#     "fn_result_dir": "results/",
#     "fn_figure_dir": "figures/",
#     "dataset_path": "data/THK_NF/HRIR_CIRC360_NF025.sofa",
#     "angle_spacings": [
#         20,
#     ],  # spacing of the reference points, degrees
#     "interpolation_methods": ["direct", "nn", "dtw"],  # Methods to test
#     "step_patterns": [
#         "symmetric1",
#         "symmetric2",
#         "symmetricP0",
#         "symmetricP05",
#         "symmetricP1",
#         "symmetricP2",
#     ],  # Allowed step patterns for DTW
#     "ir_ds_factor": 6,  # downsampling factor for IRs
#     "ir_range": range(0, 32),  # taps of the IR to consider in analysis
#     "plot_sm_limits": [-40, 8],  # dB
#     "plot_mag_limits": [-30, 0],  # dB
#     "plot_nfft": 512,
#     "export_figures": True,
#     "export_results": True,
# }


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
    angles : np.ndarray
        Angles array
    fs : float
        Sampling frequency
    ir_range : range
        Range of impulse response taps considered
    """
    sofa_data = sf.read_sofa(config["dataset_path"], "r")
    irs = sofa_data.Data_IR
    angles = 360 - sofa_data.SourcePosition[:, 0]  # Convert to intrinsic rotation
    fs = sofa_data.Data_SamplingRate

    # Sort by angle
    sort_indices = np.argsort(angles)
    irs = irs[sort_indices, ...]
    angles = np.round(angles[sort_indices]).astype(int)

    # Duplicate 360° to 0°
    if angles[0] != 0:
        irs = np.concatenate((irs[[-1]], irs), axis=0)
        angles = np.concatenate((np.array([0]), angles), axis=0)

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


def select_step_pattern(step_pattern_name: str) -> dtw.StepPattern:
    """Select DTW step pattern based on name

    Parameters
    ----------
    step_pattern_name : str
        Name of the step pattern as defined in the dtw package
    Returns
    -------
    step_pattern : dtw.StepPattern
        Selected step pattern
    """

    if step_pattern_name == "symmetric1":
        step_pattern = dtw.symmetric1
    elif step_pattern_name == "symmetric2":
        step_pattern = dtw.symmetric2
    elif step_pattern_name == "symmetricP0":
        step_pattern = dtw.symmetricP0
    elif step_pattern_name == "symmetricP05":
        step_pattern = dtw.symmetricP05
    elif step_pattern_name == "symmetricP1":
        step_pattern = dtw.symmetricP1
    elif step_pattern_name == "symmetricP2":
        step_pattern = dtw.symmetricP2
    else:
        raise ValueError(f"Unknown step pattern: {step_pattern_name}")

    # TODO: Add Rabiner-Juang patterns

    return step_pattern


def calculate_dtw(
    ir_query: np.ndarray, ir_reference: np.ndarray, stepPattern: dtw.StepPattern
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate DTW alignment between two impulse responses

    Parameters
    ----------
    ir_query : np.ndarray
        Impulse response to be warped
    ir_reference : np.ndarray
        Reference impulse response to warp to
    stepPattern : dtw.StepPattern
        Allowed step pattern for the DTW

    Returns
    -------
    ir_query_warped: np.ndarray
        Warped impulse response
    displacement: np.ndarray
        Displacement vector used for warping
    """
    # Calculate DTW between two IRs
    alignment = dtw.dtw(
        ir_query, ir_reference, step_pattern=stepPattern, keep_internals=True
    )

    # Obtain updated indices to warp query IR
    wq = dtw.warp(alignment, index_reference=False)

    # Warp query IR
    ir_query_warped = ir_query[wq]

    # Extract dipslacement, used to reconstruct time axis
    displacement = np.arange(len(ir_query)) - wq

    return ir_query_warped, displacement


def interpolate_ir_dtw(
    ir_pos0: np.ndarray,
    ir_pos1: np.ndarray,
    ir_pos0_warped: np.ndarray,
    ir_pos1_warped: np.ndarray,
    displacement_pos0: np.ndarray,
    displacement_pos1: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate between two impulse responses using DTW-based warping, based on warping a single IR

    Parameters
    ----------
    ir_pos0 : np.ndarray
        Impulse response at position 0
    ir_pos1 : np.ndarray
        Impulse response at position 1
    ir_pos0_warped : np.ndarray
        Warped impulse response at position 0
    ir_pos1_warped : np.ndarray
        Warped impulse response at position 1
    displacement_pos0 : np.ndarray
        Displacement vector used for warping at position 0
    displacement_pos1 : np.ndarray
        Displacement vector used for warping at position 1
    alpha : float
        Interpolation factor (0 = position 1, 1 = position 0)
    Returns
    -------
    ir_interpolated : np.ndarray
        Interpolated impulse response
    """

    # Linear interpolation of warped IRs
    if alpha >= 0.5:
        ir_interpolated_warped = alpha * ir_pos0 + (1 - alpha) * ir_pos1_warped

        # Find updated indices for de-warping
        idx_dewarping = np.arange(len(ir_pos0)) - displacement_pos1 * (1 - alpha)
    else:
        ir_interpolated_warped = alpha * ir_pos0_warped + (1 - alpha) * ir_pos1

        # Find updated indices for de-warping
        idx_dewarping = np.arange(len(ir_pos0)) - displacement_pos0 * (alpha)

    # Apply spline interpolation to get samples at integer-indices
    cs = CubicSpline(idx_dewarping, ir_interpolated_warped)
    ir_interpolated = cs(np.arange(len(ir_interpolated_warped)))

    return ir_interpolated


def interpolate_ir_direct(
    ir_pos0: np.ndarray, ir_pos1: np.ndarray, alpha: float
) -> np.ndarray:
    """Interpolate between two impulse responses using direct linear interpolation
    Parameters

    ----------
    ir_pos0 : np.ndarray
        Impulse response at position 0
    ir_pos1 : np.ndarray
        Impulse response at position 1
    alpha : float
        Interpolation factor (0 = position 1, 1 = position 0)

    Returns
    -------
    ir_interpolated : np.ndarray
        Interpolated impulse response
    """

    ir_interpolated = alpha * ir_pos0 + (1 - alpha) * ir_pos1
    return ir_interpolated


def interpolate_ir_nn(
    ir_pos0: np.ndarray, ir_pos1: np.ndarray, alpha: float
) -> np.ndarray:
    """Interpolate between two impulse responses using nearest neighbor interpolation

    Parameters
    ----------
    ir_pos0 : np.ndarray
        Impulse response at position 0
    ir_pos1 : np.ndarray
        Impulse response at position 1
    alpha : float
        Interpolation factor (0 = position 1, 1 = position 0)

    Returns
    -------
    ir_interpolated : np.ndarray
        Interpolated impulse response
    """

    ir_interpolated = ir_pos0 if alpha >= 0.5 else ir_pos1
    return ir_interpolated


def extract_median_error_per_quadrant(
    sm_values: np.ndarray, angles: np.ndarray, angle_spacing: int, in_db: bool = False
) -> tuple[float, float, float, float, float]:
    """Calculate the median system mismatch per quadrant on all interpolated positions

    Parameters
    ----------
    sm_values : np.ndarray
        system mismatch values
    angles : np.ndarray
        angles corresponding to the system mismatch values
    angle_spacing : int
        angle spacing used for reference positions

    Returns
    -------
    tuple[float,float, float, float, float]
        median system mismatch values overall and for front, back, contralateral, and ipsilateral quadrants
    """

    indices_interpolated = angles % angle_spacing != 0

    index_mask_front = (
        ((angles >= 315) & (angles < 360)) | (angles <= 45)
    ) & indices_interpolated

    index_mask_back = (angles >= 135) & (angles <= 225) & indices_interpolated
    index_mask_clat = (angles >= 45) & (angles <= 135) & indices_interpolated
    index_mask_ilat = (angles >= 225) & (angles <= 315) & indices_interpolated

    sm_overall = np.median(sm_values[indices_interpolated])
    sm_front = np.median(sm_values[index_mask_front])
    sm_back = np.median(sm_values[index_mask_back])
    sm_clat = np.median(sm_values[index_mask_clat])
    sm_ilat = np.median(sm_values[index_mask_ilat])

    if in_db:
        sm_overall = 20 * np.log10(sm_overall)
        sm_front = 20 * np.log10(sm_front)
        sm_back = 20 * np.log10(sm_back)
        sm_clat = 20 * np.log10(sm_clat)
        sm_ilat = 20 * np.log10(sm_ilat)

    return sm_overall, sm_front, sm_back, sm_clat, sm_ilat


def plot_mag_phase_error(
    angles: np.ndarray,
    mag_error: np.ndarray,
    phase_error: np.ndarray,
    fs: float,
    config: dict,
    export_figure: bool = False,
    export_figure_fn: str = "",
) -> None:
    """Plot magnitude and phase error for interpolated IRs

    Parameters
    ----------
    angles : np.ndarray
        Angles array
    mag_error : np.ndarray
        Magnitude error matrix (angles x frequencies)
    phase_error : np.ndarray
        Phase error matrix (angles x frequencies)
    fs : float
        Sampling frequency
    config : dict
        Configuration dictionary
    export_figure : bool
        Whether to export the figure
    export_figure_fn : str
        Filename for exporting the figure
    """
    label_pos = "Position (cm)" if config["mode"] == "tr" else "Angle (degrees)"

    freq = np.linspace(0, fs / 2, config["plot_nfft"] // 2 + 1)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.pcolormesh(
        angles,
        freq,
        20 * np.log10(mag_error.T + 1e-12),
        vmax=config["plot_mag_limits"][1],
        vmin=config["plot_mag_limits"][0],
        shading="auto",
    )
    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.colorbar(label="Magnitude error (dB)", extend="both")
    plt.title("Magnitude Error")
    plt.xlabel(label_pos)
    plt.ylabel("Frequency (Hz)")
    plt.ylim(10, fs / 2)

    plt.subplot(2, 1, 2)
    plt.pcolormesh(
        angles,
        freq,
        np.rad2deg(phase_error.T),
        vmin=-90,
        vmax=90,
        cmap="seismic",
        shading="auto",
    )
    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    cbar = plt.colorbar(label="Phase error (degrees)", extend="both")
    cbar.set_ticks([-90, 0, 90])
    cbar.set_ticklabels(["-90°", "0°", "90°"])
    plt.title("Phase Error")
    plt.xlabel(label_pos)
    plt.ylabel("Frequency (Hz)")
    plt.ylim(10, fs / 2)
    plt.tight_layout()

    if export_figure and export_figure_fn != "":
        plt.savefig(
            export_figure_fn,
            dpi=300,
        )


def main():
    # Load toml config from argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", nargs="?", const=1, type=str, default="config.toml"
    )

    args = parser.parse_args()
    config = toml.load(args.config)

    # Create directories for figures and results
    os.makedirs(config["fn_figure_dir"], exist_ok=True)
    os.makedirs(config["fn_result_dir"], exist_ok=True)

    # Import dataset
    if config["dataset_path"].endswith(".npz"):
        irs, position, fs, ir_range = load_npz_dataset(config)
    elif config["dataset_path"].endswith(".sofa"):
        irs, position, fs, ir_range = load_sofa_dataset(config)

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

    if config["export_figures"]:
        ds_name = config["dataset_path"].split("/")[-1].split(".")[0]
        plt.savefig(f"{config['fn_figure_dir']}/IR_dataset.png", dpi=300)

    results = None

    sm_all = []

    for method in config["interpolation_methods"]:
        step_patterns_to_use = config["step_patterns"] if method == "dtw" else [None]
        for step_pattern_str in step_patterns_to_use:
            for spacing_fixpos in config["spacing_fixpos"]:
                # Select reference positions based on angle spacing
                ref_indices = np.where(
                    (position - position[0]) % spacing_fixpos < 1e-6
                )[0]
                positions_per_segment = ref_indices[1] - ref_indices[0]
                ref_irs = np.squeeze(
                    irs[ref_indices, 0]
                )  # Symmetric setup --> use one side only

                # Select desired step pattern for dtw
                if method == "dtw":
                    step_pattern = select_step_pattern(step_pattern_str)

                # Initialize temporary variables to store results
                sm_tmp = np.zeros_like(position, dtype=float)
                mag_error_tmp = np.zeros((len(position), config["plot_nfft"] // 2 + 1))
                phase_error_tmp = np.zeros(
                    (len(position), config["plot_nfft"] // 2 + 1)
                )

                # Outer loop - iterate over all fixed positions
                for ii in range(len(ref_indices) - 1):
                    ir_pos0 = ref_irs[ii]
                    ir_pos1 = ref_irs[ii + 1]

                    if method == "dtw":
                        ir_pos0_warped, displacement_pos0 = calculate_dtw(
                            ir_pos0, ir_pos1, step_pattern
                        )
                        ir_pos1_warped, displacement_pos1 = calculate_dtw(
                            ir_pos1, ir_pos0, step_pattern
                        )

                    # Inner loop - interpolate to all positions in between
                    for jj in range(1, int(positions_per_segment)):
                        alpha = 1 - jj / positions_per_segment

                        if method == "direct":
                            ir_interpolated = interpolate_ir_direct(
                                ir_pos0, ir_pos1, alpha
                            )
                        elif method == "nn":
                            ir_interpolated = interpolate_ir_nn(ir_pos0, ir_pos1, alpha)
                        elif method == "dtw":
                            ir_interpolated = interpolate_ir_dtw(
                                ir_pos0,
                                ir_pos1,
                                ir_pos0_warped,
                                ir_pos1_warped,
                                displacement_pos0,
                                displacement_pos1,
                                alpha,
                            )

                        # Find target IR
                        position_target = position[ref_indices[ii] + jj]
                        idx_target = np.where(
                            np.abs(position - position_target) < 1e-6
                        )[0][0]
                        ir_target = np.squeeze(irs[idx_target, 0, ir_range])

                        # Calculate system mismatch
                        sm_tmp[idx_target] = metrics.system_mismatch(
                            ir_target, ir_interpolated
                        )

                        # Calculate magnitude and phase error for plotting
                        (
                            f_axis,
                            mag_error_tmp[ii * positions_per_segment + jj],
                            phase_error_tmp[ii * positions_per_segment + jj],
                        ) = metrics.mag_phase_error(
                            ir_target,
                            ir_interpolated,
                            nFFT=512,
                            fs=fs,
                            dB=False,
                            discont=1.5 * np.pi,
                        )

                # Extract system mismatch per quadrant

                sm_all.append(sm_tmp)

                if config["mode"] == "rot":
                    (
                        sm_tmp_overall,
                        sm_tmp_front,
                        sm_tmp_back,
                        sm_tmp_clat,
                        sm_tmp_ilat,
                    ) = extract_median_error_per_quadrant(
                        sm_tmp, position, spacing_fixpos, True
                    )
                    results_tmp = pd.DataFrame(
                        {
                            "method": [method],
                            "step_pattern": [step_pattern_str],
                            "spacing": [
                                angle_spacing,
                            ],
                            "sm_overall": [
                                sm_tmp_overall,
                            ],
                            "sm_front": [
                                sm_tmp_front,
                            ],
                            "sm_back": [sm_tmp_back],
                            "sm_clat": [sm_tmp_clat],
                            "sm_ilat": [sm_tmp_ilat],
                        }
                    )

                else:
                    sm_tmp_overall = 20 * np.log10(
                        np.median(sm_tmp[position % spacing_fixpos != 0])
                    )
                    results_tmp = pd.DataFrame(
                        {
                            "method": [method],
                            "step_pattern": [step_pattern_str],
                            "spacing": [
                                spacing_fixpos,
                            ],
                            "sm_overall": [
                                sm_tmp_overall,
                            ],
                        }
                    )

                if results is None:
                    results = results_tmp
                else:
                    results = pd.concat(
                        [results, results_tmp],
                        ignore_index=True,
                    )

                # Plot results for magnitude/phase error
                if method == "dtw" and len(config["step_patterns"]) > 1:
                    fn = (
                        config["fn_figure_dir"]
                        + f"fd_error_{method}_steppattern_{step_pattern_str}_spacing_{spacing_fixpos}.png"
                    )
                else:
                    fn = (
                        config["fn_figure_dir"]
                        + f"fd_error_{method}_spacing_{spacing_fixpos}.png"
                    )
                plot_mag_phase_error(
                    position,
                    mag_error_tmp,
                    phase_error_tmp,
                    fs,
                    config,
                    config["export_figures"],
                    fn,
                )

    # Plot system mismatch
    reg_smplot = 10 ** (config["plot_sm_limits"][0] / 20)
    if len(config["spacing_fixpos"]) > 1 and results is not None:
        for spacing_fixpos in config["spacing_fixpos"]:
            # Find all entry indices with the given angle spacing
            idx_dir = np.where(
                (results["spacing"].to_numpy() == spacing_fixpos)
                & (results["method"].to_numpy() == "direct")
            )[0][0]

            idx_nn = np.where(
                (results["spacing"].to_numpy() == spacing_fixpos)
                & (results["method"].to_numpy() == "nn")
            )[0][0]

            idx_dtw = np.where(
                (results["spacing"].to_numpy() == spacing_fixpos)
                & (results["method"].to_numpy() == "dtw")
            )[0][0]

            # Extract corresponding system mismatch arrays
            sm_dir = sm_all[idx_dir]
            sm_nn = sm_all[idx_nn]
            sm_dtw = sm_all[idx_dtw]

            # Apply regularization for low values
            sm_dir[sm_dir < reg_smplot] = reg_smplot
            sm_nn[sm_nn < reg_smplot] = reg_smplot
            sm_dtw[sm_dtw < reg_smplot] = reg_smplot

            # Convert to dB
            sm_dir = 20 * np.log10(sm_dir)
            sm_nn = 20 * np.log10(sm_nn)
            sm_dtw = 20 * np.log10(sm_dtw)

            # Plot results for system mismatch
            plt.figure()

            plt.plot(position, sm_nn, label="NN")
            plt.plot(position, sm_dir, label="Direct")
            plt.plot(position, sm_dtw, label="DTW")
            plt.xlabel(label_pos)
            plt.ylabel("System mismatch (dB)")
            plt.ylim(config["plot_sm_limits"])
            plt.title(f"System mismatch of interpolated IRs, spacing {spacing_fixpos}")
            plt.legend()
            plt.grid()

            if config["export_figures"]:
                fn = (
                    config["fn_figure_dir"]
                    + f"system_mismatch_spacing_{spacing_fixpos}.png"
                )
                plt.savefig(
                    fn,
                    dpi=300,
                )

    if len(config["step_patterns"]) > 1 and results is not None:
        idx_dtw = np.where(results["method"].to_numpy() == "dtw")[0]

        plt.figure()
        for idx in idx_dtw:
            step_pattern_str = results["step_pattern"].to_numpy()[idx]
            spacing_fixpos = results["spacing"].to_numpy()[idx]

            # Extract corresponding system mismatch arrays
            sm_dtw = sm_all[idx]

            # Apply regularization for low values
            sm_dtw[sm_dtw < reg_smplot] = reg_smplot

            # Convert to dB
            sm_dtw = 20 * np.log10(sm_dtw)

            # Plot results for system mismatch
            plt.plot(
                position,
                sm_dtw,
                label=f"DTW ({step_pattern_str})",
            )
        plt.xlabel(label_pos)
        plt.ylabel("System mismatch (dB)")
        plt.ylim(config["plot_sm_limits"])
        plt.title(f"System mismatch of interpolated IRs - DTW step patterns")
        plt.legend()
        plt.grid()

        if config["export_figures"]:
            fn = (
                config["fn_figure_dir"]
                + f"system_mismatch_dtw_steppatterns_spacing_{config['spacing_fixpos'][0]}.png"
            )
            plt.savefig(
                fn,
                dpi=300,
            )

    if config["export_results"] and results is not None:
        fn = config["fn_result_dir"] + f"{config['fn_output_prefix']}results"
        results.to_csv(fn + ".csv", index=False, float_format="%.2f")
        results.to_pickle(fn + ".pkl")

    print(results)
    # plt.show()


if __name__ == "__main__":
    main()
