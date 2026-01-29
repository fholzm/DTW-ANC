import subprocess


def main():
    subprocess.run(
        [
            "faust2jackconsole",
            "-double",
            "-osc",
            "anc_simulation/faust_dsp/sec_path_interp.dsp",
        ]
    )
    subprocess.run(
        ["faust2jackconsole", "-double", "-osc", "anc_simulation/faust_dsp/fxlms.dsp"]
    )
    subprocess.run(
        ["faust2jackconsole", "-double", "-osc", "anc_simulation/faust_dsp/noise.dsp"]
    )


if __name__ == "__main__":
    main()
