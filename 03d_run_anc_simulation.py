import os
import time
import argparse
from pythonosc import udp_client
import subprocess
from tqdm import tqdm

fs = 16000
blocksize = 16
noise_cutoff_freq = 1000.0  # Hz
start_position = 0.25  # meters
n_realizations = 1
stepsize = 0.01
interpolation_types = ["reference", "nn", "linear", "dtw"]

port_src = 9001
port_secpath = 9002
port_fxlms = 9003
port_pd = 9004


def main():
    # Create directory for results
    os.makedirs("results/anc_simulation/audiodata", exist_ok=True)

    # %% OSC configuration
    # Secondary path filter
    parser_secpath = argparse.ArgumentParser()
    parser_secpath.add_argument(
        "--ip", default="127.0.0.1", help="The ip of the OSC server"
    )
    parser_secpath.add_argument(
        "--port",
        type=int,
        default=port_secpath,
        help="The port the OSC server is listening on",
    )
    args_secpath = parser_secpath.parse_args()
    client_secpath = udp_client.SimpleUDPClient(args_secpath.ip, args_secpath.port)

    # FxLMS control filter
    parser_fxlms = argparse.ArgumentParser()
    parser_fxlms.add_argument(
        "--ip", default="127.0.0.1", help="The ip of the OSC server"
    )
    parser_fxlms.add_argument(
        "--port",
        type=int,
        default=port_fxlms,
        help="The port the OSC server is listening on",
    )
    args_fxlms = parser_fxlms.parse_args()
    client_fxlms = udp_client.SimpleUDPClient(args_fxlms.ip, args_fxlms.port)

    # pd
    parser_pd = argparse.ArgumentParser()
    parser_pd.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
    parser_pd.add_argument(
        "--port",
        type=int,
        default=port_pd,
        help="The port the OSC server is listening on",
    )
    args_pd = parser_pd.parse_args()
    client_pd = udp_client.SimpleUDPClient(args_pd.ip, args_pd.port)

    # %% Run simulations with multiple noise realizations
    for realization in tqdm(range(n_realizations), desc="Noise Realizations"):
        for interp_type in tqdm(
            interpolation_types, desc="Interpolation Types", leave=False
        ):
            with open("anc_simulation/fn_read.txt", "w") as f:
                f.write(
                    f"open "
                    + os.path.abspath(
                        f"results/anc_simulation/audiodata/innov/noise_{realization}.wav"
                    )
                    + ";\n"
                )

            # Set target directory for innovation signal and recorded data
            with open("anc_simulation/fn_write.txt", "w") as f:
                f.write(
                    f"open -bytes 3 -rate {fs} "
                    + os.path.abspath(
                        f"results/anc_simulation/audiodata/realization_{realization}_{interp_type}.wav"
                    )
                    + ";\n"
                )

            # Start programms
            p_pd = subprocess.Popen(
                [
                    "pd",
                    "-rt",
                    "-r",
                    str(fs),
                    "-nogui",
                    "-jack",
                    "-inchannels",
                    "2",
                    "-outchannels",
                    "1",
                    "./anc_simulation/anc_record.pd",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            p_controlfilter = subprocess.Popen(
                [
                    "./anc_simulation/faust_dsp/fxlms",
                    "-xmit",
                    "1",
                    "-port",
                    str(port_fxlms),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            p_secpath = subprocess.Popen(
                [
                    "./anc_simulation/faust_dsp/sec_path_interp",
                    "-xmit",
                    "1",
                    "-port",
                    str(port_secpath),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            time.sleep(0.1)

            p_tascar = subprocess.Popen(
                ["tascar_cli", "anc_simulation/tascar_sim.tsc"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            time.sleep(3)

            # Restore JACK connections
            os.system("./jcmess -D")
            time.sleep(0.1)
            os.system("./jcmess -l anc_simulation/jack_connections.txt")
            time.sleep(0.1)

            # %% Update all parameters via OSC, reset control filter coeffs
            # Disable any adaptation
            client_fxlms.send_message(
                "/ANC_control_filter/Adaptation/Disable_adaptation", 1
            )
            time.sleep(0.1)

            # Set roundtrip delay for error feedback in jack
            client_fxlms.send_message("/ANC_control_filter/Adaptation/Delay", blocksize)
            time.sleep(0.1)

            # Set stepsize
            client_fxlms.send_message(
                "/ANC_control_filter/Adaptation/Stepsize", stepsize
            )
            time.sleep(0.1)

            # Reset filter coefficients
            client_fxlms.send_message("/ANC_control_filter/Adaptation/Reset", 1)
            time.sleep(0.1)
            client_fxlms.send_message("/ANC_control_filter/Adaptation/Reset", 0)
            time.sleep(0.1)

            # Set interpolation type
            if interp_type == "nn":
                interp_code = 0
            elif interp_type == "linear":
                interp_code = 1
            elif interp_type == "dtw":
                interp_code = 2
            else:  # reference
                interp_code = 0

            client_secpath.send_message(
                "/Secondary_path_interpolator/Method", interp_code
            )
            client_secpath.send_message(
                "/Secondary_path_interpolator/Position", start_position
            )
            time.sleep(1)

            client_pd.send_message("/position", [start_position, 0])
            time.sleep(1)

            client_pd.send_message("/play", 1)
            client_pd.send_message("/record", 1)
            time.sleep(1)

            if interp_type != "reference":
                # Enable adaptation
                client_fxlms.send_message(
                    "/ANC_control_filter/Adaptation/Disable_adaptation", 0
                )

            time.sleep(9)

            client_pd.send_message("/position", [0.475, 2000])
            time.sleep(12)

            client_pd.send_message("/position", [0.55, 2000])
            time.sleep(12)

            client_pd.send_message("/position", [0.625, 2000])
            time.sleep(12)

            client_pd.send_message("/record", 0)
            client_pd.send_message("/play", 0)
            time.sleep(1)

            os.system("./jcmess -D")
            time.sleep(0.1)

            p_pd.kill()
            p_controlfilter.kill()
            p_secpath.kill()
            p_tascar.kill()
            time.sleep(3)


if __name__ == "__main__":
    main()
