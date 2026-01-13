import numpy as np
from scipy.interpolate import CubicSpline
import dtw
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils import metrics
import pandas as pd

# global variables
config = {
    "dataset_path": "data/TASCAR_IRs/measured_irs.npz",
    "angle_spacings": [
        5,
        10,
        15,
        20,
        30,
        45,
    ],  # spacing of the reference points, degrees
    "ir_range": range(0, 256),  # taps of the IR to consider in analysis
    "stepPattern": dtw.symmetricP2,  # DTW step pattern
    # "stepPattern": dtw.rabinerJuangStepPattern(6, "c"),
    "plot_sm_lower_limit": -80,  # dB
    "plot_mag_lower_limit": -40,  # dB
    "plot_nfft": 512,
    "export_figures": True,
    "export_results": True,
}


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


def inteprolate_ir(
    ir_pos0: np.ndarray,
    ir_pos1_warped: np.ndarray,
    displacement: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate between two impulse responses using DTW-based warping

    Parameters
    ----------
    ir_pos0 : np.ndarray
        Impulse response at position 0
    ir_pos1_warped : np.ndarray
        Warped impulse response at position 1
    displacement : np.ndarray
        Displacement vector used for warping
    alpha : float
        Interpolation factor (0 = position 1, 1 = position 0)

    Returns
    -------
    ir_interpolated : np.ndarray
        Interpolated impulse response
    """

    # Linear interpolation of warped IRs
    ir_interpolated_warped = alpha * ir_pos0 + (1 - alpha) * ir_pos1_warped

    # Find updated indices for de-warping
    idx_dewarping = np.arange(len(ir_pos0)) - displacement * (1 - alpha)

    # Apply spline interpolation to get samples at integer-indices
    cs = CubicSpline(idx_dewarping, ir_interpolated_warped)
    ir_interpolated = cs(np.arange(len(ir_interpolated_warped)))

    return ir_interpolated


def extract_median_error_per_quadrant(
    sm_values: np.ndarray, angles: np.ndarray, angle_spacing: int, in_db: bool = False
) -> tuple[float, float, float, float, float]:
    """Calculate the median system mismatch per quadrant on all inteprolated positions

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
    angle_spacing: int,
    method_name: str,
    export_figures: bool = False,
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
    angle_spacing : int
        Angle spacing used
    method_name : str
        Name of the interpolation method
    export_figures : bool
        Whether to export the figure
    """
    freq = np.linspace(0, fs / 2, config["plot_nfft"] // 2 + 1)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.pcolormesh(
        angles,
        freq,
        20 * np.log10(mag_error.T + 1e-12),
        vmax=0,
        vmin=config["plot_mag_lower_limit"],
        shading="auto",
    )
    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.colorbar(label="Magnitude error (dB)")
    plt.title(f"{method_name} magnitude error, spacing {angle_spacing}°")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(10, fs / 2)

    plt.subplot(2, 1, 2)
    plt.pcolormesh(
        angles,
        freq,
        phase_error.T,
        vmin=-np.pi,
        vmax=np.pi,
        cmap="seismic",
        shading="auto",
    )
    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.colorbar(label="Phase error (radians)")
    plt.title(f"{method_name} phase error, spacing {angle_spacing}°")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(10, fs / 2)
    plt.tight_layout()

    if export_figures:
        method_slug = method_name.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(
            f"figures/mag_phase_error_{method_slug}_spacing_{angle_spacing}deg.png",
            dpi=300,
        )


def main():
    # Import dataset
    ir_data = np.load(config["dataset_path"])
    irs = ir_data["irs"]
    angles = ir_data["angles"]
    fs = ir_data["fs"].item()

    del ir_data

    results = pd.DataFrame(
        columns=[
            "method",
            "spacing",
            "sm_overall",
            "sm_front",
            "sm_back",
            "sm_clat",
            "sm_ilat",
        ]
    )

    for angle_spacing in config["angle_spacings"]:
        # Select reference positions based on angle spacing
        ref_indices = np.where(angles % angle_spacing == 0)[0]
        ref_irs = np.squeeze(
            irs[ref_indices, 0]
        )  # Symmetric setup --> use one side only
        ir_refs = ref_irs[:, config["ir_range"]]

        sm_dtw_tmp = np.zeros_like(angles, dtype=float)
        sm_nn_tmp = np.zeros_like(angles, dtype=float)
        sm_linear_tmp = np.zeros_like(angles, dtype=float)

        mag_error_dtw = np.zeros((len(angles), config["plot_nfft"] // 2 + 1))
        phase_error_dtw = np.zeros((len(angles), config["plot_nfft"] // 2 + 1))
        mag_error_nn = np.zeros((len(angles), config["plot_nfft"] // 2 + 1))
        phase_error_nn = np.zeros((len(angles), config["plot_nfft"] // 2 + 1))
        mag_error_linear = np.zeros((len(angles), config["plot_nfft"] // 2 + 1))
        phase_error_linear = np.zeros((len(angles), config["plot_nfft"] // 2 + 1))

        # Outer loop - iterate over all fixed positions
        for ii in range(len(ref_indices) - 1):
            ir_pos0 = ir_refs[ii]
            ir_pos1 = ir_refs[ii + 1]

            ir_pos1_warped, displacement = calculate_dtw(
                ir_pos1, ir_pos0, config["stepPattern"]
            )

            # Inner loop - interpolate to all positions in between
            for jj in range(1, angle_spacing):
                alpha = 1 - jj / angle_spacing
                ir_interpolated_dtw = inteprolate_ir(
                    ir_pos0, ir_pos1_warped, displacement, alpha
                )

                ir_interpolated_nn = ir_pos0 if alpha >= 0.5 else ir_pos1
                ir_inteprolated_linear = alpha * ir_pos0 + (1 - alpha) * ir_pos1

                # Find target IR
                angle_target = angles[ref_indices[ii]] + jj
                idx_target = np.where(angles == angle_target)[0][0]
                ir_target = np.squeeze(irs[idx_target, 0, config["ir_range"]])

                # Calculate system mismatch
                sm_dtw_tmp[idx_target] = metrics.system_mismatch(
                    ir_target, ir_interpolated_dtw
                )
                sm_nn_tmp[idx_target] = metrics.system_mismatch(
                    ir_target, ir_interpolated_nn
                )
                sm_linear_tmp[idx_target] = metrics.system_mismatch(
                    ir_target, ir_inteprolated_linear
                )

                # Calculate magnitude and phase error for plotting
                (
                    f_axis,
                    mag_error_dtw[ii * angle_spacing + jj],
                    phase_error_dtw[ii * angle_spacing + jj],
                ) = metrics.mag_phase_error(
                    ir_target, ir_interpolated_dtw, nFFT=512, fs=fs, dB=False
                )
                (
                    _,
                    mag_error_nn[ii * angle_spacing + jj],
                    phase_error_nn[ii * angle_spacing + jj],
                ) = metrics.mag_phase_error(
                    ir_target, ir_interpolated_nn, nFFT=512, fs=fs, dB=False
                )
                (
                    _,
                    mag_error_linear[ii * angle_spacing + jj],
                    phase_error_linear[ii * angle_spacing + jj],
                ) = metrics.mag_phase_error(
                    ir_target, ir_inteprolated_linear, nFFT=512, fs=fs, dB=False
                )

        # Extract system mismatch per quadrant
        sm_dtw_overall, sm_dtw_front, sm_dtw_back, sm_dtw_clat, sm_dtw_ilat = (
            extract_median_error_per_quadrant(sm_dtw_tmp, angles, angle_spacing, True)
        )
        sm_nn_overall, sm_nn_front, sm_nn_back, sm_nn_clat, sm_nn_ilat = (
            extract_median_error_per_quadrant(sm_nn_tmp, angles, angle_spacing, True)
        )
        (
            sm_linear_overall,
            sm_linear_front,
            sm_linear_back,
            sm_linear_clat,
            sm_linear_ilat,
        ) = extract_median_error_per_quadrant(
            sm_linear_tmp, angles, angle_spacing, True
        )

        results = pd.concat(
            [
                results,
                pd.DataFrame(
                    {
                        "method": ["DTW", "Nearest Neighbor", "Linear"],
                        "spacing": [angle_spacing, angle_spacing, angle_spacing],
                        "sm_overall": [
                            sm_dtw_overall,
                            sm_nn_overall,
                            sm_linear_overall,
                        ],
                        "sm_front": [sm_dtw_front, sm_nn_front, sm_linear_front],
                        "sm_back": [sm_dtw_back, sm_nn_back, sm_linear_back],
                        "sm_clat": [sm_dtw_clat, sm_nn_clat, sm_linear_clat],
                        "sm_ilat": [sm_dtw_ilat, sm_nn_ilat, sm_linear_ilat],
                    }
                ),
            ],
            ignore_index=True,
        )

        # Plot results with regularization for low values
        reg_lin = 10 ** (config["plot_sm_lower_limit"] / 20)
        sm_dtw_tmp[sm_dtw_tmp < reg_lin] = reg_lin
        sm_nn_tmp[sm_nn_tmp < reg_lin] = reg_lin
        sm_linear_tmp[sm_linear_tmp < reg_lin] = reg_lin

        sm_dtw_db = 20 * np.log10(sm_dtw_tmp)
        sm_nn_db = 20 * np.log10(sm_nn_tmp)
        sm_linear_db = 20 * np.log10(sm_linear_tmp)

        # Calculate mean/median on all positions that are interpolated
        interp_mask = np.ones_like(angles, dtype=bool)
        interp_mask[ref_indices] = False

        print(f"System mismatch for angle spacing {angle_spacing}°:")
        print(f"DTW: Median: {sm_dtw_overall} dB")
        print(f"Linear: Median: {sm_linear_overall} dB")
        print(f"Nearest neighbor: Median: {sm_nn_overall} dB")
        print("----------")

        # Plot results for system mismatch
        plt.figure()
        plt.plot(angles, sm_dtw_db, label="DTW-based interpolation")
        plt.plot(angles, sm_linear_db, label="Linear interpolation")
        plt.plot(angles, sm_nn_db, label="Nearest neighbor")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("System mismatch")
        plt.title(f"System mismatch of interpolated IRs, spacing {angle_spacing}°")
        plt.legend()
        plt.grid()

        if config["export_figures"]:
            plt.savefig(
                f"figures/system_mismatch_spacing_{angle_spacing}deg.png", dpi=300
            )

        # Plot results for magnitude/phase error
        plot_mag_phase_error(
            angles,
            mag_error_dtw,
            phase_error_dtw,
            fs,
            angle_spacing,
            "DTW-based interpolation",
            config["export_figures"],
        )
        plot_mag_phase_error(
            angles,
            mag_error_linear,
            phase_error_linear,
            fs,
            angle_spacing,
            "Linear interpolation",
            config["export_figures"],
        )
        plot_mag_phase_error(
            angles,
            mag_error_nn,
            phase_error_nn,
            fs,
            angle_spacing,
            "Nearest neighbor",
            config["export_figures"],
        )

    if config["export_results"]:
        results.to_csv("results/system_mismatch_results.csv", index=False)
        results.to_pickle("results/system_mismatch_results.pkl")

    print(results)
    plt.show()


if __name__ == "__main__":
    main()
