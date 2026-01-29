import subprocess


def main():
    subprocess.run(["faust2jack", "-double", "faust_dsp/sec_path_interp.dsp"])
    subprocess.run(["faust2jaqt", "-double", "faust_dsp/fxlms.dsp"])
    subprocess.run(["faust2jaqt", "-double", "faust_dsp/noise.dsp"])


if __name__ == "__main__":
    main()
