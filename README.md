# Dynamic Time Warping for Secondary Path Interpolation in Local Active Noise Control

[![Github Repository](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/fholzm/DTW-ANC) [![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/7746185/tree) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

For stable and performant local active noise control (ANC), accurate estimates of the secondary path between sources and the desired point of cancellation are required. This poses a problem in time-variant scenes with variable point of cancellation, as in practice these paths are recorded prior to operation for a limited number of positions. In this article, we propose an interpolation approach for impulse responses of secondary paths. By applying dynamic time warping (DTW), impulse responses are properly aligned before interpolation and de-warping. The system is benchmarked against nearest-neighbor and linear interpolation for translation and rotation of the listener's head. Our DTW-based technique exhibits lower overall system mismatch and extends the frequency range for stable operation, especially for lateral translation and coarse measurement grids. It outperforms the other systems especially in terms of phase error towards high frequencies. In simulation, an up to 15 dB higher noise reduction could be achieved for broadband disturbances. The proposed technique can potentially reduce the required number of measurement positions of secondary paths substantially while increasing the controlled frequency bandwidth for time-variant real-world systems.

## Usage

Navigate to the `code/` directory. All experiments are written in Python, you can run them using the `code/run.sh` file on UN*X systems. An environment with all required packages can be easily installed via [uv](https://docs.astral.sh/uv/). To generate data and perform the ANC experiments, [TASCAR](https://tascar.org/), Jack, and [FAUST](https://faust.grame.fr/) are required and accesisble in your systems path. For ease of use, you can run the code in the provided Ubuntu-based [devcontainer](https://containers.dev/), which comes with all necessary dependencies.

Although the ANC simulations don't require much processing power, they take a lot of time to run (400 simulations with 46 seconds length in real time) and tend to stall sometimes. For this reason, they can be manually started by running the `code/run_ancsim.sh` files. If simulation stops and you want to continue it without starting from the beginning, you can set the `start_realization` variable in `code/03d_run_anc_simulation.py`.

## Structure

The code is structured this way:

- `code/` contains all code for the experiments, including the FAUST DSP code for the ANC system.
  - `code/run.sh` is the main script to generate data and run IR-based assessmets.
  - `code/run_ancsim.sh` is the main script to run the ANC simulations for the lateral transition experiments.
  - `code/01_generate_data_tr.py` generates the data for lateral transition experiments.
  - `code/02_ir_assessment.py` performs evaluations of the interpolated impulse responses.
  - `code/03a_calculate_irs.py` calculates and warpes the filter coefficients for the ANC simulation.
  - `code/03b_build_faust_apps.py` compiles the FAUST DSP code for the ANC simulaiton.
  - `code/03c_create_innovationsignal.py` renders the test signals for the ANC simulaitons.
  - `code/03d_run_anc_simulation.py` runs the ANC simulations for the lateral transition experiments.
  - `code/03e_analysis_anc_simulation.py` analyzes the results of the ANC simulations.
  - `code/anc_simulation/` contains the code for the ANC simulation, including the FAUST DSP code for the ANC system.
  - `code/configs/` contains the configuration files for the experiments.
  - `code/utils/` contains utility functions in Python for the experiments.
- `data/THK_NF/` contains a near-field HRIR dataset by TH Köln, which is used for the experiments.
- `results/` contains the results of the experiments, including the metrics as `*.csv` and the figures.
- `third-party-licenses/` contains the licenses for the third-party code and data used in this repository.
- `.devcontainer/` contains the configuration for the development container, which can be used to run the code in a consistent environment.

## Citation

The article corresponding to this repository has been submitted to the IEEE Open Journal of Signal Processing. If you use the code or data in this repository, please cite the article after publication.

## Licenses

Code for this publication is licensed under GNU GPL v3.0.

### Third-Party Licenses

Some third-party code and data is included in this repository. The licenses themselves are included in the `./third-party-licenses/`directory.

- `code/anc_simulation/faust_dsp/lib/spline.h` by Tino Kluge is licensed under GNU GPL v2.0.
- The HRIR-dataset by TH Köln in `data/THK_NF/` is licensed under CC-BY-4.0.
- The FAUST-code of `code/anc_simulation/faust_dsp/fxlms.dsp` has been [published](https://zenodo.org/records/13859827) previously and licensed under MIT.
- The [jcmess](https://github.com/synthnassizer/jcmess) application is written by Athanasios Silis and licensed under Apache 2.0.
