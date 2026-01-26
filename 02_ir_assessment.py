import numpy as np
from scipy import signal
import sofar as sf
import matplotlib.pyplot as plt
from utils import metrics, interpolate
import pandas as pd
import os
import toml
import glob


def load_npz_dataset(config: dict) -> tuple[np.ndarray, np.ndarray, float, range]:
    """Load dataset from npz file

    Parameters
    ----------
    config : dict

    Returns
    -------
    irs : np.ndarray
        Impulse responses array
    positions : np.ndarray
        Positions array
    fs : float
        Sampling frequency
    ir_range : range
        Range of impulse response taps considered
    """

    ir_data = np.load(config["dataset_path"])
    irs = ir_data["irs"]
    positions = ir_data["positions"]
    fs = ir_data["fs"].item()

    if config["ir_ds_factor"] > 1:
        irs_ds = signal.decimate(irs, config["ir_ds_factor"], axis=2, zero_phase=True)
        irs = irs_ds
        fs /= config["ir_ds_factor"]

    if config["ir_range"][1] > irs.shape[2]:
        ir_range = range(config["ir_range"][0], irs.shape[2])
    else:
        ir_range = range(config["ir_range"][0], config["ir_range"][1])

    irs = irs[:, :, ir_range]

    return irs, positions, fs, ir_range


def load_sofa_dataset(config: dict) -> tuple[np.ndarray, np.ndarray, float, range]:
    """Load dataset from SOFA file

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    irs : np.ndarray
        Impulse responses array
    positions : np.ndarray
        Positions array
    fs : float
        Sampling frequency
    ir_range : range
        Range of impulse response taps considered
    """
    sofa_data = sf.read_sofa(config["dataset_path"], "r")
    irs = sofa_data.Data_IR
    positions = 360 - sofa_data.SourcePosition[:, 0]  # Convert to intrinsic rotation
    fs = sofa_data.Data_SamplingRate

    # Sort by angle
    sort_indices = np.argsort(positions)
    irs = irs[sort_indices, ...]
    positions = np.round(positions[sort_indices]).astype(int)

    # Duplicate 360° to 0°
    if positions[0] != 0:
        irs = np.concatenate((irs[[-1]], irs), axis=0)
        positions = np.concatenate((np.array([0]), positions), axis=0)

    if config["ir_ds_factor"] > 1:
        irs_ds = signal.decimate(irs, config["ir_ds_factor"], axis=2, zero_phase=True)
        irs = irs_ds
        fs /= config["ir_ds_factor"]

    if config["ir_range"][1] > irs.shape[2]:
        ir_range = range(config["ir_range"][0], irs.shape[2])
    else:
        ir_range = range(config["ir_range"][0], config["ir_range"][1])

    irs = irs[:, :, ir_range]

    return irs, positions, fs, ir_range


def main():
    config_files = sorted(glob.glob("configs/*.toml"))
    if not config_files:
        print("No config files found in ./configs directory")
        return

    # Process each config file
    for config_file in config_files:
        print(f"\n{'='*80}")
        print(f"Processing config: {config_file}")
        print(f"{'='*80}\n")

        process_config(config_file)


def process_config(config_path: str):
    """Process a single config file"""
    config = toml.load(config_path)

    # Create directories for figures and results
    os.makedirs(config["fn_figure_dir"], exist_ok=True)
    os.makedirs(config["fn_result_dir"], exist_ok=True)

    # Import dataset
    if config["dataset_path"].endswith(".npz"):
        irs, position, fs, ir_range = load_npz_dataset(config)
    elif config["dataset_path"].endswith(".sofa"):
        irs, position, fs, ir_range = load_sofa_dataset(config)

    label_pos = "Position (cm)" if config["mode"] == "tr" else "Angle (degrees)"

    # Plot IR dataset
    hrir_to_plot = np.abs(irs[:, 0, ir_range])
    hrir_to_plot /= np.max(hrir_to_plot)

    t_axis = np.arange(len(ir_range)) / fs * 1000  # in ms

    plt.figure()
    ax1 = plt.gca()
    plt.pcolormesh(
        position,
        t_axis,
        20 * np.log10(hrir_to_plot.T + 1e-12),
        shading="auto",
        cmap="Greys",
        vmin=-60,
        vmax=0,
    )
    plt.colorbar(label="Magnitude (dB)", pad=0.15)
    plt.title("IR dataset")
    plt.xlabel(label_pos)
    ax1.set_ylabel("Time (ms)")

    # Add second y-axis with samples
    ax2 = ax1.twinx()
    ax2.set_ylim(config["ir_range"][0] - 0.5, config["ir_range"][1] - 1.5)
    ax2.set_ylabel("Samples")
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if config["export_figures"]:
        ds_name = config["dataset_path"].split("/")[-1].split(".")[0]
        plt.savefig(f"{config['fn_figure_dir']}/IR_dataset.png", dpi=300)

    results = None

    sm_all = []

    for method in config["interpolation_methods"]:
        step_patterns_to_use = config["step_patterns"] if method == "dtw" else [None]
        for step_pattern_str in step_patterns_to_use:
            for spacing_fixpos in config["spacing_fixpos"]:
                # Select reference positions based on angle spacing
                ref_indices = np.where(
                    (position - position[0]) % spacing_fixpos < 1e-6
                )[0]
                positions_per_segment = ref_indices[1] - ref_indices[0]
                ref_irs = np.squeeze(
                    irs[ref_indices, 0]
                )  # Symmetric setup --> use one side only

                # Select desired step pattern for dtw
                if method == "dtw":
                    step_pattern = interpolate.select_step_pattern(step_pattern_str)

                # Initialize temporary variables to store results
                sm_tmp = np.zeros_like(position, dtype=float)
                mag_error_tmp = np.zeros((len(position), config["plot_nfft"] // 2 + 1))
                phase_error_tmp = np.zeros(
                    (len(position), config["plot_nfft"] // 2 + 1)
                )

                # Outer loop - iterate over all fixed positions
                for ii in range(len(ref_indices) - 1):
                    ir_pos0 = ref_irs[ii]
                    ir_pos1 = ref_irs[ii + 1]

                    if method == "dtw":
                        ir_pos0_warped, displacement_pos0 = interpolate.calculate_dtw(
                            ir_pos0, ir_pos1, step_pattern
                        )
                        ir_pos1_warped, displacement_pos1 = interpolate.calculate_dtw(
                            ir_pos1, ir_pos0, step_pattern
                        )

                    # Inner loop - interpolate to all positions in between
                    for jj in range(1, int(positions_per_segment)):
                        alpha = 1 - jj / positions_per_segment

                        if method == "direct":
                            ir_interpolated = interpolate.interpolate_ir_direct(
                                ir_pos0, ir_pos1, alpha
                            )
                        elif method == "nn":
                            ir_interpolated = interpolate.interpolate_ir_nn(
                                ir_pos0, ir_pos1, alpha
                            )
                        elif method == "dtw":
                            ir_interpolated = interpolate.interpolate_ir_dtw(
                                ir_pos0,
                                ir_pos1,
                                ir_pos0_warped,
                                ir_pos1_warped,
                                displacement_pos0,
                                displacement_pos1,
                                alpha,
                                config["dewarping_interpolator"],
                            )

                        # Find target IR
                        position_target = position[ref_indices[ii] + jj]
                        idx_target = np.where(
                            np.abs(position - position_target) < 1e-6
                        )[0][0]
                        ir_target = np.squeeze(irs[idx_target, 0, ir_range])

                        # Calculate system mismatch
                        sm_tmp[idx_target] = metrics.system_mismatch(
                            ir_target, ir_interpolated
                        )

                        # Calculate magnitude and phase error for plotting
                        (
                            f_axis,
                            mag_error_tmp[ii * positions_per_segment + jj],
                            phase_error_tmp[ii * positions_per_segment + jj],
                        ) = metrics.mag_phase_error(
                            ir_target,
                            ir_interpolated,
                            nFFT=512,
                            fs=fs,
                            dB=False,
                            discont=1.5 * np.pi,
                        )

                # Extract system mismatch per quadrant

                sm_all.append(sm_tmp)

                if config["mode"] == "rot":
                    (
                        sm_tmp_overall,
                        sm_tmp_front,
                        sm_tmp_back,
                        sm_tmp_clat,
                        sm_tmp_ilat,
                    ) = metrics.extract_median_error_per_quadrant(
                        sm_tmp, position, spacing_fixpos, True
                    )
                    results_tmp = pd.DataFrame(
                        {
                            "method": [method],
                            "step_pattern": [step_pattern_str],
                            "spacing": [
                                spacing_fixpos,
                            ],
                            "sm_overall": [
                                sm_tmp_overall,
                            ],
                            "sm_front": [
                                sm_tmp_front,
                            ],
                            "sm_back": [sm_tmp_back],
                            "sm_clat": [sm_tmp_clat],
                            "sm_ilat": [sm_tmp_ilat],
                        }
                    )

                else:
                    sm_tmp_overall = 20 * np.log10(
                        np.median(sm_tmp[position % spacing_fixpos != 0])
                    )
                    results_tmp = pd.DataFrame(
                        {
                            "method": [method],
                            "step_pattern": [step_pattern_str],
                            "spacing": [
                                spacing_fixpos,
                            ],
                            "sm_overall": [
                                sm_tmp_overall,
                            ],
                        }
                    )

                if results is None:
                    results = results_tmp
                else:
                    results = pd.concat(
                        [results, results_tmp],
                        ignore_index=True,
                    )

                # Plot results for magnitude/phase error
                if method == "dtw" and len(config["step_patterns"]) > 1:
                    fn = (
                        config["fn_figure_dir"]
                        + f"fd_error_{method}_steppattern_{step_pattern_str}_spacing_{spacing_fixpos}.png"
                    )
                else:
                    fn = (
                        config["fn_figure_dir"]
                        + f"fd_error_{method}_spacing_{spacing_fixpos}.png"
                    )
                metrics.plot_mag_phase_error(
                    position,
                    mag_error_tmp,
                    phase_error_tmp,
                    fs,
                    config,
                    config["export_figures"],
                    fn,
                )

    # Plot system mismatch
    reg_smplot = 10 ** (config["plot_sm_limits"][0] / 20)
    if len(config["spacing_fixpos"]) > 1 and results is not None:
        for spacing_fixpos in config["spacing_fixpos"]:
            # Find all entry indices with the given angle spacing
            idx_dir = np.where(
                (results["spacing"].to_numpy() == spacing_fixpos)
                & (results["method"].to_numpy() == "direct")
            )[0][0]

            idx_nn = np.where(
                (results["spacing"].to_numpy() == spacing_fixpos)
                & (results["method"].to_numpy() == "nn")
            )[0][0]

            idx_dtw = np.where(
                (results["spacing"].to_numpy() == spacing_fixpos)
                & (results["method"].to_numpy() == "dtw")
            )[0][0]

            # Extract corresponding system mismatch arrays
            sm_dir = sm_all[idx_dir]
            sm_nn = sm_all[idx_nn]
            sm_dtw = sm_all[idx_dtw]

            # Apply regularization for low values
            sm_dir[sm_dir < reg_smplot] = reg_smplot
            sm_nn[sm_nn < reg_smplot] = reg_smplot
            sm_dtw[sm_dtw < reg_smplot] = reg_smplot

            # Convert to dB
            sm_dir = 20 * np.log10(sm_dir)
            sm_nn = 20 * np.log10(sm_nn)
            sm_dtw = 20 * np.log10(sm_dtw)

            # Plot results for system mismatch
            plt.figure()

            plt.plot(position, sm_nn, label="NN")
            plt.plot(position, sm_dir, label="Direct")
            plt.plot(position, sm_dtw, label="DTW")
            plt.xlabel(label_pos)
            plt.ylabel("System mismatch (dB)")
            plt.ylim(config["plot_sm_limits"])
            plt.title(f"System mismatch of interpolated IRs, spacing {spacing_fixpos}")
            plt.legend()
            plt.grid()

            if config["export_figures"]:
                fn = (
                    config["fn_figure_dir"]
                    + f"system_mismatch_spacing_{spacing_fixpos}.png"
                )
                plt.savefig(
                    fn,
                    dpi=300,
                )

    if len(config["step_patterns"]) > 1 and results is not None:
        idx_dtw = np.where(results["method"].to_numpy() == "dtw")[0]

        plt.figure()
        for idx in idx_dtw:
            step_pattern_str = results["step_pattern"].to_numpy()[idx]
            spacing_fixpos = results["spacing"].to_numpy()[idx]

            # Extract corresponding system mismatch arrays
            sm_dtw = sm_all[idx]

            # Apply regularization for low values
            sm_dtw[sm_dtw < reg_smplot] = reg_smplot

            # Convert to dB
            sm_dtw = 20 * np.log10(sm_dtw)

            # Plot results for system mismatch
            plt.plot(
                position,
                sm_dtw,
                label=f"DTW ({step_pattern_str})",
            )
        plt.xlabel(label_pos)
        plt.ylabel("System mismatch (dB)")
        plt.ylim(config["plot_sm_limits"])
        plt.title(f"System mismatch of interpolated IRs - DTW step patterns")
        plt.legend()
        plt.grid()

        if config["export_figures"]:
            fn = (
                config["fn_figure_dir"]
                + f"system_mismatch_dtw_steppatterns_spacing_{config['spacing_fixpos'][0]}.png"
            )
            plt.savefig(
                fn,
                dpi=300,
            )

    if config["export_results"] and results is not None:
        fn = config["fn_result_dir"] + f"{config['fn_output_prefix']}results"
        results.to_csv(fn + ".csv", index=False, float_format="%.2f")
        results.to_pickle(fn + ".pkl")

    print(results)
    # plt.show()


if __name__ == "__main__":
    main()
