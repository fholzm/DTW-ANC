import numpy as np
from scipy.signal import medfilt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import os

dir_audiofiles = "../results/anc_simulation/audiodata"
dir_figures = "../results/figures/"
rms_length = 100  # in ms
export_figures = True

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 1 * 8
plt.rcParams["axes.titlesize"] = 1 * 8
plt.rcParams["axes.labelsize"] = 1 * 8
plt.rcParams["xtick.labelsize"] = 1 * 8
plt.rcParams["ytick.labelsize"] = 1 * 8
plt.rcParams["legend.fontsize"] = 1 * 8

movements = [[10, 12], [22, 24], [34, 36]]


def parse_audiofiles(directory: str) -> tuple[int, list[str]]:
    """Extract filename patterns, methods, and number of realizations

    Parameters
    ----------
    directory : str
        Directory containing audio files to parse

    Returns
    -------
    n_realizations : int
        Number of realizations found in the filenames
    methods : list[str]
        List of unique methods found in the filenames
    """

    # List all files in the directory
    all_files = [f for f in os.listdir(directory) if f.endswith(".wav")]

    fn_pattern = [file.split(".wav")[0].split("_") for file in all_files]

    # Number of realization is second element in fn_pattern
    n_realizations = len(set([fn[1] for fn in fn_pattern]))

    # Method is last part if filename pattern
    methods = list(set([fn[-1] for fn in fn_pattern]))

    return n_realizations, methods


def sliding_window_rms(signal, window_size):
    """Calculates the root mean square of a signal using a sliding window.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    window_size : int
        Size of the sliding window.

    Returns
    -------
    np.ndarray
        RMS signal.
    """
    return np.sqrt(
        np.convolve(signal**2, np.ones(window_size) / window_size, mode="same")
    )


def main():
    os.makedirs(dir_figures, exist_ok=True)

    n_realizations_overall, methods = parse_audiofiles(dir_audiofiles)

    signal_rms = []
    rms_signal_mean = []

    min_filelength = float("inf")

    n_realizations = 0

    # Read audio files, compute RMS, medfilt, and average across realizations
    for realization in range(n_realizations_overall):
        signal_rms_tmp = []
        clipping_detected = False
        for method_idx, method in enumerate(methods):
            filename = f"realization_{realization}_{method}.wav"
            filepath = os.path.join(dir_audiofiles, filename)

            signal, fs = sf.read(filepath)

            if np.any(np.abs(signal) >= 0.99):
                print(f"Clipping detected in file: {filename}")
                clipping_detected = True

            if len(signal) < min_filelength:
                min_filelength = len(signal)

            signal_rms_tmp.append(
                sliding_window_rms(signal[..., 0], int(rms_length * fs / 1000))
            )

        if not clipping_detected:
            n_realizations += 1
            signal_rms.append(signal_rms_tmp)

    # Compute ensemble average of RMS values across realizations for each method
    for method_idx in range(len(methods)):
        rms_signal_mean_tmp = np.zeros((min_filelength,))
        for realization in range(n_realizations):
            rms_signal_mean_tmp += signal_rms[realization][method_idx][:min_filelength]
        rms_signal_mean.append(rms_signal_mean_tmp)

    methods_sorted = ["reference", "nn", "linear", "dtw"]
    method_labels = ["Reference", "NN", "LI", "DTW"]
    linecolors = ["black", "tab:blue", "tab:orange", "tab:green"]
    linestyles = ["-", "-", "-", "-"]

    plt.figure(figsize=(3.5, 1.75))
    time_axis = np.arange(min_filelength) / fs

    for idx, method in enumerate(methods_sorted):
        method_idx = methods.index(method)
        plt.plot(
            time_axis[rms_length * fs // 1000 : -rms_length * fs // 1000],
            20
            * np.log10(
                rms_signal_mean[method_idx][
                    rms_length * fs // 1000 : -rms_length * fs // 1000
                ]
            ),
            label=method_labels[idx],
            color=linecolors[idx],
            linestyle=linestyles[idx],
        )

    for movement in movements:
        plt.axvspan(
            movement[0],
            movement[1],
            facecolor="gray",
            edgecolor=None,
            alpha=0.5,
            label="Movement" if movement == movements[0] else None,
        )

    plt.xlabel("Time (s)")
    plt.ylabel("RMS Amplitude (dB)")
    plt.xlim(0, 46)
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        borderaxespad=0,
        frameon=False,
    )
    plt.grid()
    plt.tight_layout(pad=0.1)
    if export_figures:
        plt.savefig(os.path.join(dir_figures, "holzm5.png"), dpi=600)


if __name__ == "__main__":
    main()
