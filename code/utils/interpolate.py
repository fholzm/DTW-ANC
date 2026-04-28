import numpy as np
import dtw
from scipy.interpolate import CubicSpline
from romanutils import rtoi


def get_global_alignment(ir_query: np.ndarray, ir_reference: np.ndarray):
    """Calculate time offset through cross correlation for global alignment

    Parameters
    ----------
    ir_query : np.ndarray
        Impulse response to be aligned
    ir_reference : np.ndarray
        Reference impulse response

    Returns
    -------
    time_offset: int
        Time offset to apply to ir_query for global alignment with ir_reference
    """

    # Calculate cross-correlation between the two IRs
    cross_correlation = np.correlate(ir_query, ir_reference, mode="full")
    time_offset = np.argmax(cross_correlation) - (len(ir_reference) - 1)

    return -time_offset


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
    elif step_pattern_name.startswith("rabinerJuang"):
        step_pattern_tmp = step_pattern_name[len("rabinerJuang") : -1]
        slope_weighting = step_pattern_name[-1]
        patterntype = rtoi(step_pattern_tmp)
        step_pattern = dtw.rabinerJuangStepPattern(
            patterntype, slope_weighting=slope_weighting
        )
    elif step_pattern_name.startswith("mvm"):
        elasticity = int(step_pattern_name[len("mvm") :])
        step_pattern = dtw.mvmStepPattern(elasticity)
    else:
        raise ValueError(f"Unknown step pattern: {step_pattern_name}")

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
    dewarping_interpolator: str = "cs",
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
        Interpolation factor (0 = position 0, 1 = position 1)
    dewarping_interpolator : str
        Interpolator to extract values at integer positions after dewarping

    Returns
    -------
    ir_interpolated : np.ndarray
        Interpolated impulse response
    """

    # Linear interpolation of warped IRs
    if alpha >= 0.5:
        ir_interpolated_warped = (1 - alpha) * ir_pos0_warped + alpha * ir_pos1

        # Find updated indices for de-warping
        idx_dewarping = np.arange(len(ir_pos0)) - displacement_pos0 * (1 - alpha)
    else:
        ir_interpolated_warped = (1 - alpha) * ir_pos0 + alpha * ir_pos1_warped

        # Find updated indices for de-warping
        idx_dewarping = np.arange(len(ir_pos0)) - displacement_pos1 * (alpha)

    # Apply spline interpolation to get samples at integer-indices
    if dewarping_interpolator == "cs":
        interpolator = CubicSpline(
            idx_dewarping, ir_interpolated_warped, bc_type="natural"
        )
        ir_interpolated = interpolator(np.arange(len(ir_interpolated_warped)))
    elif dewarping_interpolator == "lin":
        ir_interpolated = np.interp(
            np.arange(len(ir_interpolated_warped)),
            idx_dewarping,
            ir_interpolated_warped,
        )
    else:
        raise ValueError(
            f"Unknown dewarping_interpolator: {dewarping_interpolator}. Supported values are 'cs' and 'lin'."
        )

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
        Interpolation factor (0 = position 0, 1 = position 1)

    Returns
    -------
    ir_interpolated : np.ndarray
        Interpolated impulse response
    """

    ir_interpolated = (1 - alpha) * ir_pos0 + alpha * ir_pos1
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
        Interpolation factor (0 = position 0, 1 = position 1)

    Returns
    -------
    ir_interpolated : np.ndarray
        Interpolated impulse response
    """

    ir_interpolated = ir_pos0 if alpha <= 0.5 else ir_pos1
    return ir_interpolated
