import numpy as np
import soundfile as sf
from scipy import signal
import os

n_realizations = 100
fs = 16000
sig_length = 50 * fs  # 50 seconds
lp_cutoff = 1000.0  # Hz
lp_order = 8
seed = 42


def main():
    os.makedirs("results/anc_simulation/audiodata/innov", exist_ok=True)

    sos = signal.butter(lp_order, lp_cutoff, btype="low", fs=fs, output="sos")

    for realization in range(n_realizations):
        np.random.seed(seed + realization)
        noise = np.random.uniform(-1, 1, sig_length)
        noise = signal.sosfilt(sos, noise)

        filename = f"results/anc_simulation/audiodata/innov/noise_{realization}.wav"
        sf.write(filename, noise, fs)


if __name__ == "__main__":
    main()
