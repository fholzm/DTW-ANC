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
    # mag_error = np.abs(IR_INTERPOLATED_FFT / (IR_TARGET_FFT + 1e-12))

    if dB:
        mag_error = 20 * np.log10(mag_error + 1e-12)  # add small value to avoid log(0)

    # Calculate phase error
    phase_error = np.angle(IR_TARGET_FFT * np.conj(IR_INTERPOLATED_FFT), deg=False)

    return f_axis, mag_error, phase_error


def extract_metric_per_quadrant(
    metric_raw: np.ndarray,
    angles: np.ndarray,
    angle_spacing: int,
    mode: str = "median",
    in_db: bool = False,
) -> tuple[float, float, float, float, float]:
    """Calculate the median system mismatch per quadrant on all interpolated positions

    Parameters
    ----------
    metric_raw : np.ndarray
        raw metric values (e.g., system mismatch) for all angles
    angles : np.ndarray
        angles corresponding to the system mismatch values
    angle_spacing : int
        angle spacing used for reference positions
    mode : str
        mode for calculating the metric per quadrant, possible values: "median" (default), "mean", "max", "min"
    in_db : bool
        whether to return the metric values in dB (20*log10) or linear scale (default: False)

    Returns
    -------
    tuple[float,float, float, float, float]
        metrics overall and for front, back, contralateral, and ipsilateral quadrants
    """

    if mode == "median":
        metric_func = np.median
    elif mode == "mean":
        metric_func = np.mean
    elif mode == "max":
        metric_func = np.max
    elif mode == "min":
        metric_func = np.min
    else:
        raise ValueError(
            f"Invalid mode: {mode}. Supported modes are: 'median', 'mean', 'max', 'min'."
        )

    indices_interpolated = angles % angle_spacing != 0

    index_mask_front = (
        ((angles >= 315) & (angles < 360)) | (angles <= 45)
    ) & indices_interpolated

    index_mask_back = (angles >= 135) & (angles <= 225) & indices_interpolated
    index_mask_clat = (angles >= 45) & (angles <= 135) & indices_interpolated
    index_mask_ilat = (angles >= 225) & (angles <= 315) & indices_interpolated

    sm_overall = metric_func(metric_raw[indices_interpolated])
    sm_front = metric_func(metric_raw[index_mask_front])
    sm_back = metric_func(metric_raw[index_mask_back])
    sm_clat = metric_func(metric_raw[index_mask_clat])
    sm_ilat = metric_func(metric_raw[index_mask_ilat])

    if in_db:
        sm_overall = 20 * np.log10(sm_overall)
        sm_front = 20 * np.log10(sm_front)
        sm_back = 20 * np.log10(sm_back)
        sm_clat = 20 * np.log10(sm_clat)
        sm_ilat = 20 * np.log10(sm_ilat)

    return sm_overall, sm_front, sm_back, sm_clat, sm_ilat


def plot_mag_phase_error(
    positions: np.ndarray,
    mag_error: np.ndarray,
    phase_error: np.ndarray,
    fs: float,
    stable_freq: np.ndarray,
    config: dict,
    export_figure: bool = False,
    export_figure_fn: str = "",
) -> None:
    """Plot magnitude and phase error for interpolated IRs

    Parameters
    ----------
    positions : np.ndarray
        Position array
    mag_error : np.ndarray
        Magnitude error matrix (positions x frequencies)
    phase_error : np.ndarray
        Phase error matrix (positions x frequencies)
    fs : float
        Sampling frequency
    stable_freq : np.ndarray
        Stable frequency for each position (frequencies above this value are considered unstable)
    config : dict
        Configuration dictionary
    export_figure : bool
        Whether to export the figure
    export_figure_fn : str
        Filename for exporting the figure
    """
    label_pos = "Position (cm)" if config["mode"] == "tr" else "Angle (degrees)"

    freq = np.linspace(0, fs / 2, config["plot_nfft"] // 2 + 1)

    plt.figure(figsize=(1 * 7.1 / 3, 1 * 7.1 / (3 * 1.2)))
    plt.subplot(2, 1, 1)
    plt.pcolormesh(
        positions,
        freq,
        20 * np.log10(mag_error.T + 1e-12),
        vmax=config["plot_mag_limits"][1],
        vmin=config["plot_mag_limits"][0],
        cmap="Greys",
        shading="auto",
    )
    plt.plot(
        positions, stable_freq, color="red", linestyle="-", label="Stable frequency"
    )
    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.colorbar(label="Error (dB)", extend="both")
    # plt.title("Normalized Magnitude Error")
    # plt.xlabel(label_pos)
    plt.ylabel("Frequency (Hz)")
    plt.ylim(10, fs / 2)

    plt.subplot(2, 1, 2)
    plt.pcolormesh(
        positions,
        freq,
        np.rad2deg(np.abs(phase_error.T)),
        vmin=0,
        vmax=90,
        cmap="Greys",
        shading="auto",
    )
    plt.plot(
        positions, stable_freq, color="red", linestyle="-", label="Stable frequency"
    )
    plt.yscale("log")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    cbar = plt.colorbar(label="Error (degrees)", extend="max")
    cbar.set_ticks([0, 45, 90])
    cbar.set_ticklabels(["0°", "45°", "90°"])
    # plt.title("Absolute Phase Error")
    plt.xlabel(label_pos)
    plt.ylabel("Frequency (Hz)")
    plt.ylim(10, fs / 2)
    plt.tight_layout()

    if export_figure and export_figure_fn != "":
        plt.savefig(
            export_figure_fn,
            dpi=300,
        )


def plot_mag_phase_error_multiplot(
    positions: np.ndarray,
    spacings: list,
    spacings_to_plot: list,
    methods: list,
    mag_error: list[np.ndarray],
    phase_error: list[np.ndarray],
    fs: float,
    stable_freq: np.ndarray,
    config: dict,
    export_figure: bool = False,
    export_figure_fn: str = "",
) -> None:
    """Plot magnitude and phase error for interpolated IRs

    Parameters
    ----------
    positions : np.ndarray
        Position array
    spacings : list
        List of spacings used for interpolation
    spacings_to_plot : list
        List of spacings to plot
    mag_error : list[np.ndarray]
        List of magnitude error matrices for each spacing (positions x frequencies)
    phase_error : list[np.ndarray]
        List of phase error matrices for each spacing (positions x frequencies)
    fs : float
        Sampling frequency
    stable_freq : np.ndarray
        Stable frequency for each position (frequencies above this value are considered unstable)
    config : dict
        Configuration dictionary
    export_figure : bool
        Whether to export the figure
    export_figure_fn : str
        Filename for exporting the figure
    """
    label_pos = "Position (cm)" if config["mode"] == "tr" else "Angle (degrees)"
    x_ticks = [0, 90, 180, 270, 360] if config["mode"] == "rot" else [25, 50, 75, 100]

    freq = np.linspace(0, fs / 2, config["plot_nfft"] // 2 + 1)

    n_spacings = len(spacings_to_plot)
    nc = 3

    figsize = (7.16, 7.16 * n_spacings / 4.5)

    fig = plt.figure(figsize=figsize)
    subfigs = fig.subfigures(nrows=n_spacings, ncols=1, hspace=0.02)
    if n_spacings == 1:
        subfigs = [subfigs]

    for i, spacing in enumerate(spacings_to_plot):
        sf = subfigs[i]
        sf_axes = sf.subplots(
            nrows=2,
            ncols=nc,
            sharex=True,
            sharey=False,
            gridspec_kw={"wspace": 0.15, "hspace": 0.15},
        )

        for j in range(nc):
            idx = np.where(
                (np.array(spacings) == spacing)
                & (np.array(methods) == config["interpolation_methods"][j])
            )[0][0]

            mag_error_spacing = mag_error[idx]
            phase_error_spacing = phase_error[idx]

            ax_mag = sf_axes[0, j]
            ax_phase = sf_axes[1, j]

            # Magnitude error
            im = ax_mag.pcolormesh(
                positions,
                freq,
                20 * np.log10(mag_error_spacing.T + 1e-12),
                vmax=config["plot_mag_limits"][1],
                vmin=config["plot_mag_limits"][0],
                cmap="Greys",
                shading="auto",
            )
            ax_mag.plot(
                positions,
                stable_freq[idx],
                color="red",
                linestyle="-",
                label="Stable frequency",
            )
            if j == nc - 1:
                sf.colorbar(
                    im,
                    ax=list(sf_axes[0, :]),
                    label="Mag. Err. (dB)",
                    extend="both",
                    fraction=0.05,
                    pad=0.02,
                )
            ax_mag.set_yscale("log")
            ax_mag.yaxis.set_major_formatter(ScalarFormatter())
            ax_mag.yaxis.get_major_formatter().set_scientific(False)
            ax_mag.set_ylim(100, fs / 2)
            if j > 0:
                ax_mag.set_yticklabels([])
            ax_mag.tick_params(labelbottom=False)

            # Phase error
            im = ax_phase.pcolormesh(
                positions,
                freq,
                np.rad2deg(np.abs(phase_error_spacing.T)),
                vmin=0,
                vmax=90,
                cmap="Greys",
                shading="auto",
            )
            ax_phase.plot(
                positions,
                stable_freq[idx],
                color="red",
                linestyle="-",
                label="Stable frequency",
            )
            if j == nc - 1:
                cbar = sf.colorbar(
                    im,
                    ax=list(sf_axes[1, :]),
                    label="Ph. Err. (deg)",
                    extend="max",
                    fraction=0.05,
                    pad=0.02,
                )
                cbar.set_ticks([0, 45, 90])
                cbar.set_ticklabels(["0°", "45°", "90°"])
            ax_phase.set_xticks(x_ticks)
            ax_phase.set_yscale("log")
            ax_phase.yaxis.set_major_formatter(ScalarFormatter())
            ax_phase.yaxis.get_major_formatter().set_scientific(False)
            ax_phase.set_ylim(100, fs / 2)
            if j > 0:
                ax_phase.set_yticklabels([])
            if i < n_spacings - 1:
                ax_phase.tick_params(labelbottom=False)
            else:
                ax_phase.set_xlabel(label_pos, labelpad=-0.5)

            # Subfigure caption below each mag/phase column pair
            caption_idx = i * nc + j
            caption_offset = -0.6 if i == n_spacings - 1 else -0.2

            if methods[idx] == "nn":
                method_name = "NN"
            elif methods[idx] == "direct":
                method_name = "Lin. interp."
            elif methods[idx] == "dtw":
                method_name = "DTW"
            else:
                method_name = "n/a"

            if config["mode"] == "tr":
                caption_text = f"{spacing:.0f} cm spacing, {method_name}"
            else:
                caption_text = f"{spacing:.1f}° spacing, {method_name}"

            ax_phase.text(
                0.5,
                caption_offset,
                f"({chr(97 + caption_idx)}) {caption_text}",
                ha="center",
                va="top",
                fontsize=8,
                fontfamily="serif",
                transform=ax_phase.transAxes,
            )

    fig.supylabel("Frequency (Hz)", x=0.05)

    if export_figure and export_figure_fn != "":
        plt.savefig(
            export_figure_fn,
            dpi=300,
            bbox_inches="tight",
        )
