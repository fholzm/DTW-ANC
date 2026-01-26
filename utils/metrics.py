import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


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
