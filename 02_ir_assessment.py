import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import lagrange
import dtw
import matplotlib.pyplot as plt
from utils import metrics

# global variables
config = {
    "dataset_path": "data/TASCAR_IRs/measured_irs.npz",
    "angle_spacing": 20,  # spacing of the reference points, degrees
    "ir_range": range(0, 256),  # taps of the IR to consider in analysis
    "stepPattern": dtw.symmetricP2,  # DTW step pattern
    # "stepPattern": dtw.rabinerJuangStepPattern(6, "c"),
    "plot_lower_limit": -100,  # dB
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


def main():
    # Import dataset
    ir_data = np.load(config["dataset_path"])
    irs = ir_data["irs"]
    angles = ir_data["angles"]
    fs = ir_data["fs"].item()

    del ir_data

    # Select reference positions based on angle spacing
    ref_indices = np.where(angles % config["angle_spacing"] == 0)[0]
    ref_irs = np.squeeze(irs[ref_indices, 0])  # Symmetric setup --> use one side only
    ir_refs = ref_irs[:, config["ir_range"]]

    sm_dtw = np.zeros_like(angles, dtype=float)
    sm_nn = np.zeros_like(angles, dtype=float)
    sm_linear = np.zeros_like(angles, dtype=float)

    # Outer loop - iterate over all fixed positions
    for ii in range(len(ref_indices) - 1):
        ir_pos0 = ir_refs[ii]
        ir_pos1 = ir_refs[ii + 1]

        ir_pos1_warped, displacement = calculate_dtw(
            ir_pos1, ir_pos0, config["stepPattern"]
        )

        # Inner loop - interpolate to all positions in between
        for jj in range(1, config["angle_spacing"]):
            alpha = 1 - jj / config["angle_spacing"]
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
            sm_dtw[idx_target] = metrics.system_mismatch(ir_target, ir_interpolated_dtw)
            sm_nn[idx_target] = metrics.system_mismatch(ir_target, ir_interpolated_nn)
            sm_linear[idx_target] = metrics.system_mismatch(
                ir_target, ir_inteprolated_linear
            )

    # Set system mismatch at reference positions to lower plot limit
    sm_dtw[ref_indices] = 10 ** (config["plot_lower_limit"] / 20)
    sm_nn[ref_indices] = 10 ** (config["plot_lower_limit"] / 20)
    sm_linear[ref_indices] = 10 ** (config["plot_lower_limit"] / 20)

    sm_dtw_db = 20 * np.log10(sm_dtw)
    sm_nn_db = 20 * np.log10(sm_nn)
    sm_linear_db = 20 * np.log10(sm_linear)

    # Calculate mean/median on all positions that are interpolated
    interp_mask = np.ones_like(angles, dtype=bool)
    interp_mask[ref_indices] = False
    print(f"Median SM DTW: {np.median(sm_dtw_db[interp_mask]):.2f} dB")
    print(f"Median SM Linear: {np.median(sm_linear_db[interp_mask]):.2f} dB")
    print(f"Median SM NN: {np.median(sm_nn_db[interp_mask]):.2f} dB \n")

    print(f"Mean SM DTW: {20*np.log10(np.mean(sm_dtw[interp_mask])):.2f} dB")
    print(f"Mean SM Linear: {20*np.log10(np.mean(sm_linear[interp_mask])):.2f} dB")
    print(f"Mean SM NN: {20*np.log10(np.mean(sm_nn[interp_mask])):.2f} dB")

    # Plot results
    plt.figure()
    plt.plot(angles, sm_dtw_db, label="DTW-based interpolation")
    plt.plot(angles, sm_linear_db, label="Linear interpolation")
    plt.plot(angles, sm_nn_db, label="Nearest neighbor")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("System mismatch")
    plt.title("System mismatch of interpolated HRIRs")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
