import numpy as np
from scipy.interpolate import CubicSpline
from scipy import signal
import sofar as sf
import dtw
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils import metrics
import pandas as pd

# global variables
config = {
    "fn_output_prefix": "eval_spacing_",
    "fn_result_dir": "results/",
    "fn_figure_dir": "figures/",
    "dataset_path": "data/THK_NF/HRIR_CIRC360_NF025.sofa",
    "angle_spacings": [
        5,
        10,
        15,
        20,
        30,
        45,
    ],  # spacing of the reference points, degrees
    "interpolation_methods": ["direct", "nn", "dtw"],  # Methods to test
    "step_patterns": ["symmetricP2"],  # Allowed step patterns for DTW
    "ir_ds_factor": 6,  # downsampling factor for IRs
    "ir_range": range(0, 32),  # taps of the IR to consider in analysis
    "plot_sm_limits": [-40, 8],  # dB
    "plot_mag_limits": [-30, 0],  # dB
    "plot_nfft": 512,
    "export_figures": True,
    "export_results": True,
}

# config = {
#     "fn_output_prefix": "eval_steppattern_",
#     "fn_output_dir": "results/",
#     "dataset_path": "data/THK_NF/HRIR_CIRC360_NF025.sofa",
#     "angle_spacings": [
#         30,
#     ],  # spacing of the reference points, degrees
#     "interpolation_methods": ["direct", "nn", "dtw"],  # Methods to test
#     "step_patterns": [],  # Allowed step patterns for DTW
#     "ir_ds_factor": 6,  # downsampling factor for IRs
#     "ir_range": range(0, 32),  # taps of the IR to consider in analysis
#     # "stepPattern": dtw.symmetricP2,  # DTW step pattern
#     "stepPattern": dtw.rabinerJuangStepPattern(6, "c"),
#     "plot_sm_limits": [-40, 8],  # dB
#     "plot_mag_limits": [-30, 0],  # dB
#     "plot_nfft": 512,
#     "export_figures": True,
#     "export_results": True,
# }


def load_npz_dataset(file_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load dataset from npz file

    Parameters
    ----------
    file_path : str
        Path to the npz file

    Returns
    -------
    irs : np.ndarray
        Impulse responses array
    angles : np.ndarray
        Angles array
    fs : float
        Sampling frequency
    """
    ir_data = np.load(file_path)
    irs = ir_data["irs"]
    angles = np.round(ir_data["angles"]).astype(int)
    fs = ir_data["fs"].item()

    return irs, angles, fs


def load_sofa_dataset(config: dict) -> tuple[np.ndarray, np.ndarray, float]:
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

    if config["ir_range"].stop > irs.shape[2]:
        config["ir_range"] = range(config["ir_range"].start, irs.shape[2])

    # Plot IR dataset
    hrir_to_plot = np.abs(irs[:, 0, config["ir_range"]])
    hrir_to_plot /= np.max(hrir_to_plot)

    t_axis = np.arange(len(config["ir_range"])) / fs * 1000  # in ms

    plt.figure()
    ax1 = plt.gca()
    plt.pcolormesh(
        angles,
        t_axis,
        20 * np.log10(hrir_to_plot.T + 1e-12),
        shading="auto",
        cmap="Greys",
        vmin=-60,
        vmax=0,
    )
    plt.colorbar(label="Magnitude (dB)", pad=0.15)
    plt.title("IR dataset")
    plt.xlabel("Angle (degrees)")
    ax1.set_ylabel("Time (ms)")

    # Add second y-axis with samples
    ax2 = ax1.twinx()
    ax2.set_ylim(config["ir_range"].start - 0.5, config["ir_range"].stop - 1.5)
    ax2.set_ylabel("Samples")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if config["export_figures"]:
        ds_name = config["dataset_path"].split("/")[-1].split(".")[0]
        plt.savefig(f"figures/ir_{ds_name}.png", dpi=300)

    return irs, angles, fs


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
    export_figure : bool
        Whether to export the figure
    export_figure_fn : str
        Filename for exporting the figure
    """
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
    plt.xlabel("Source angle (degrees)")
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
    plt.xlabel("Source angle (degrees)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(10, fs / 2)
    plt.tight_layout()

    if export_figure and export_figure_fn != "":
        plt.savefig(
            export_figure_fn,
            dpi=300,
        )


def main():
    # Import dataset
    if config["dataset_path"].endswith(".npz"):
        irs, angles, fs = load_npz_dataset(config["dataset_path"])
    elif config["dataset_path"].endswith(".sofa"):
        irs, angles, fs = load_sofa_dataset(config)

    results = None

    sm_all = []

    for method in config["interpolation_methods"]:
        step_patterns_to_use = config["step_patterns"] if method == "dtw" else [None]
        for step_pattern_str in step_patterns_to_use:
            for angle_spacing in config["angle_spacings"]:
                # Select reference positions based on angle spacing
                ref_indices = np.where(angles % angle_spacing == 0)[0]
                ref_irs = np.squeeze(
                    irs[ref_indices, 0]
                )  # Symmetric setup --> use one side only

                # Select desired step pattern for dtw
                if method == "dtw":
                    step_pattern = select_step_pattern(step_pattern_str)

                # Initialize temporary variables to store results
                sm_tmp = np.zeros_like(angles, dtype=float)
                mag_error_tmp = np.zeros((len(angles), config["plot_nfft"] // 2 + 1))
                phase_error_tmp = np.zeros((len(angles), config["plot_nfft"] // 2 + 1))

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
                    for jj in range(1, angle_spacing):
                        alpha = 1 - jj / angle_spacing

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
                        angle_target = angles[ref_indices[ii]] + jj
                        idx_target = np.where(angles == angle_target)[0][0]
                        ir_target = np.squeeze(irs[idx_target, 0, config["ir_range"]])

                        # Calculate system mismatch
                        sm_tmp[idx_target] = metrics.system_mismatch(
                            ir_target, ir_interpolated
                        )

                        # Calculate magnitude and phase error for plotting
                        (
                            f_axis,
                            mag_error_tmp[ii * angle_spacing + jj],
                            phase_error_tmp[ii * angle_spacing + jj],
                        ) = metrics.mag_phase_error(
                            ir_target,
                            ir_interpolated,
                            nFFT=512,
                            fs=fs,
                            dB=False,
                            discont=1.5 * np.pi,
                        )

                # Extract system mismatch per quadrant
                sm_tmp_overall, sm_tmp_front, sm_tmp_back, sm_tmp_clat, sm_tmp_ilat = (
                    extract_median_error_per_quadrant(
                        sm_tmp, angles, angle_spacing, True
                    )
                )
                sm_all.append(sm_tmp)

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

                if results is None:
                    results = results_tmp
                else:
                    results = pd.concat(
                        [results, results_tmp],
                        ignore_index=True,
                    )

                # Plot results for magnitude/phase error
                if method == "dtw":
                    fn = (
                        config["fn_figure_dir"]
                        + f"fd_error_{method}_steppattern_{step_pattern_str}_spacing_{angle_spacing}deg.png"
                    )
                else:
                    fn = (
                        config["fn_figure_dir"]
                        + f"fd_error_{method}_spacing_{angle_spacing}deg.png"
                    )
                plot_mag_phase_error(
                    angles,
                    mag_error_tmp,
                    phase_error_tmp,
                    fs,
                    config["export_figures"],
                    fn,
                )

    # # TODO: Fix plot
    # # Plot results with regularization for low values
    # reg_lin = 10 ** (config["plot_sm_limits"][0] / 20)
    # sm_tmp[sm_tmp < reg_lin] = reg_lin
    # sm_nn_tmp[sm_nn_tmp < reg_lin] = reg_lin
    # sm_linear_tmp[sm_linear_tmp < reg_lin] = reg_lin

    # sm_dtw_db = 20 * np.log10(sm_tmp)
    # sm_nn_db = 20 * np.log10(sm_nn_tmp)
    # sm_linear_db = 20 * np.log10(sm_linear_tmp)

    # # Calculate mean/median on all positions that are interpolated
    # interp_mask = np.ones_like(angles, dtype=bool)
    # interp_mask[ref_indices] = False

    # print(f"System mismatch for angle spacing {angle_spacing}°:")
    # print(f"DTW: Median: {sm_tmp_overall} dB")
    # print(f"Linear: Median: {sm_linear_overall} dB")
    # print(f"Nearest neighbor: Median: {sm_nn_overall} dB")
    # print("----------")

    # # Plot results for system mismatch
    # plt.figure()
    # plt.plot(angles, sm_dtw_db, label="DTW-based interpolation")
    # plt.plot(angles, sm_linear_db, label="Linear interpolation")
    # plt.plot(angles, sm_nn_db, label="Nearest neighbor")
    # plt.xlabel("Angle (degrees)")
    # plt.ylabel("System mismatch (dB)")
    # plt.ylim(config["plot_sm_limits"])
    # plt.title(f"System mismatch of interpolated IRs, spacing {angle_spacing}°")
    # plt.legend()
    # plt.grid()

    # if config["export_figures"]:
    #     plt.savefig(
    #         f"figures/system_mismatch_spacing_{angle_spacing}deg.png",
    #         dpi=300,
    #     )

    if config["export_results"] and results is not None:
        fn = config["fn_result_dir"] + f"{config['fn_output_prefix']}results"
        results.to_csv(fn + ".csv", index=False, float_format="%.2f")
        results.to_pickle(fn + ".pkl")

    print(results)
    # plt.show()


if __name__ == "__main__":
    main()
