from pyfar.signals import noise
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import os

dir_audiofiles = "./results/anc_simulation"
dir_figures = "./figures/anc_simulation"
export_figures = True


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
    all_files = os.listdir(directory)

    fn_pattern = [file.split(".wav")[0].split("_") for file in all_files]

    # Number of realization is second element in fn_pattern
    n_realizations = len(set([fn[1] for fn in fn_pattern]))

    # Method is last part if filename pattern
    methods = list(set([fn[-1] for fn in fn_pattern]))

    return n_realizations, methods


def main():
    os.makedirs(dir_figures, exist_ok=True)

    n_realizations, methods = parse_audiofiles(dir_audiofiles)

    squared_signals = [[] for _ in range(len(methods))]
    mean_squared_signals = []

    min_filelength = float("inf")

    for realization in range(n_realizations):
        for method_idx, method in enumerate(methods):
            filename = f"realization_{realization}_{method}.wav"
            filepath = os.path.join(dir_audiofiles, filename)

            signal, fs = sf.read(filepath)

            if len(signal) < min_filelength:
                min_filelength = len(signal)

            squared_signal = signal**2
            squared_signals[method_idx].append(squared_signal)

    for method_idx in range(len(methods)):
        mean_squared_signal_tmp = np.zeros((min_filelength,))
        for realization in range(n_realizations):
            mean_squared_signal_tmp += squared_signals[method_idx][realization][
                :min_filelength, 0
            ]
        mean_squared_signals.append(mean_squared_signal_tmp / n_realizations)

    reference_idx = methods.index("reference")

    noise_reduction_db = np.zeros((len(methods) - 1, min_filelength))
    for method_idx, method in enumerate(methods):
        if method == "reference":
            continue
        nr_db = 10 * np.log10(
            mean_squared_signals[reference_idx] / mean_squared_signals[method_idx]
        )
        noise_reduction_db[method_idx - 1, :] = nr_db

    plt.figure(figsize=(10, 6))
    time_axis = np.arange(min_filelength) / fs

    for method_idx, method in enumerate(methods):
        if method == "reference":
            continue
        plt.plot(
            time_axis,
            noise_reduction_db[method_idx - 1, :],
            label=f"{method}",
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Noise Reduction (dB)")
    plt.title("ANC Noise Reduction over Time")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if export_figures:
        plt.savefig(os.path.join(dir_figures, "anc_noise_reduction.png"))


if __name__ == "__main__":
    main()
