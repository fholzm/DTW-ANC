import os
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
import subprocess
from tqdm import tqdm
from lxml import objectify, etree

# Define global parameters
sec_src_coords = [0.25, 0.0, 0.0]
sec_src_gain = -20.0
angles_to_test = np.arange(0, 361, 1, dtype=float)
fs = 48000
ir_length = 512
n_jobs = 8


def generate_tascar_project(rotation: float = 0.0) -> str:
    fn = f"ir_rot{rotation:.1f}"

    root = objectify.Element(
        "session",
        attribution="Felix Holzmüller (autogen)",
    )
    scene = objectify.SubElement(
        root,
        "scene",
        name=fn,
        guiscale="6",
        ismorder="0",
    )

    # Add secondary source at specified coordinates and gain
    source = objectify.SubElement(scene, "source", name="sec_source")
    snd = objectify.SubElement(source, "sound", type="omni", gain=str(sec_src_gain))
    pos = objectify.SubElement(source, "position")._setText(
        f"0 {sec_src_coords[0]} {sec_src_coords[1]} {sec_src_coords[2]}"
    )

    # Add hrtf receiver with specified orientation
    rec = objectify.SubElement(
        scene,
        "receiver",
        name="hrtf_receiver",
        type="hrtf",
        nf_filter="true",
        # sincorder="64",
        # prewarpingmode="2",
    )
    orient = objectify.SubElement(rec, "orientation")._setText(f"0 {rotation} 0 0")

    # Remove annotations from root
    objectify.deannotate(root, xsi_nil=True)
    etree.cleanup_namespaces(root)

    # Create string from xml
    obj_xml = etree.tostring(
        root, pretty_print=True, xml_declaration=True, encoding="utf-8"
    )

    try:
        with open(f"{fn}.tsc", "wb") as f:
            f.write(obj_xml)
    except IOError:
        pass

    return fn


def read_irs(fn: str) -> np.ndarray:
    irs = sf.read(f"{fn}.wav")[0]
    return irs[:ir_length]


def render_tascar_project(rotation: float = 0.0):
    # Generate project file
    fn = generate_tascar_project(rotation)

    # Render project
    cmd = (
        "tascar_renderfile --verbose --srate "
        + str(fs)
        + " --inputfile impulse.wav --outputfile "
        + fn
        + ".wav "
        + fn
        + ".tsc"
    )
    result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)

    with open("tascar.log", "a") as log:
        log.write(f"Scene at angle {rotation} rendered successfully.\n")
        log.write(result.stdout)
        log.write(result.stderr)
        log.write("\n")

    ir = read_irs(fn)

    # Delete tascar project
    os.remove(f"{fn}.tsc")
    os.remove(f"{fn}.wav")
    return ir


def main():
    os.remove("tascar.log") if os.path.exists("tascar.log") else None
    # Create impulse for measurement
    impulse = np.zeros(fs + ir_length)
    impulse[0] = 1.0
    sf.write("impulse.wav", impulse, fs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(render_tascar_project)(angle) for angle in tqdm(angles_to_test)
    )

    # Delete impulse after use
    os.remove("impulse.wav")

    # Convert results in a format, where irs for specific angles can easiöly be extracted
    irs = np.array(results).transpose(0, 2, 1)

    # Save results to file
    if not os.path.exists("data/TASCAR_IRs"):
        os.mkdir("data/TASCAR_IRs")
    np.savez("data/TASCAR_IRs/measured_irs.npz", irs=irs, angles=angles_to_test, fs=fs)


if __name__ == "__main__":
    main()
