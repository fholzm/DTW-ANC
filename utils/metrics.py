import numpy as np


def system_mismatch(ir_target: np.ndarray, ir_interpolated: np.ndarray) -> float:
    """System mismatch

    System mismatch metric [1] to asses the quality of an interpolated impulse response. It is computed as the l2-norm of the difference
    between the target and interpolated impulse responses, normalized by the l2-norm of the target impulse response.

    Parameters
    ----------
    ir_target : np.ndarray
        Reference impulse response
    ir_interpolated : np.ndarray
        Interpolated impulse response under test

    Returns
    -------
    float
        System mismatch (linear scale). Lower values indicate better quality, 0 indicates perfect reconstruction.

    References
    [1] M. Buerger, S. Meier, C. Hofmann, W. Kellermann, E. Fischer, and H. Puder, “Retrieval of individualized head-related transfer functions for hearing aid applications,” in 2017 25th European Signal Processing Conference (EUSIPCO), Aug. 2017, pp. 6–10. doi: 10.23919/EUSIPCO.2017.8081158.


    ----------

    """
    return float(
        np.linalg.norm(ir_target - ir_interpolated) / np.linalg.norm(ir_target)
    )
