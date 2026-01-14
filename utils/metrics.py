import numpy as np
from typing import Optional


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


def mag_phase_error(
    ir_target: np.ndarray,
    ir_interpolated: np.ndarray,
    nFFT: Optional[int] = None,
    fs: Optional[float | int] = None,
    dB: bool = False,
    discont: float = np.pi,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # If nFFT is not given, set it to next power of two of the longest IR
    if nFFT is None:
        nFFT = int(2 ** (np.ceil(np.log2(max(len(ir_target), len(ir_interpolated))))))

    if fs is None:
        fs = 2 * np.pi

    f_axis = np.linspace(0, fs / 2, nFFT // 2 + 1)

    # Transform to frequency domain
    IR_TARGET_FFT = np.fft.rfft(ir_target, n=nFFT)
    IR_INTERPOLATED_FFT = np.fft.rfft(ir_interpolated, n=nFFT)

    # Calculate relative magnitude error
    mag_error = np.abs(IR_TARGET_FFT - IR_INTERPOLATED_FFT) / np.abs(IR_TARGET_FFT)

    if dB:
        mag_error = 20 * np.log10(mag_error + 1e-12)  # add small value to avoid log(0)

    # Calculate phase error
    phase_error = np.unwrap(
        np.angle(IR_TARGET_FFT, deg=False), discont=discont
    ) - np.unwrap(np.angle(IR_INTERPOLATED_FFT, deg=False), discont=discont)

    return f_axis, mag_error, phase_error
