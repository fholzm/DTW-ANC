import os
import numpy as np
import soundfile as sf
from joblib import Parallel, delayed
import subprocess
from tqdm import tqdm
from lxml import objectify, etree

# Define global parameters
sec_src_coords = [[0.177, 0.177, 0.0], [0.177, -0.177, 0.0]]
sec_src_gain = -20.0
positions_to_test = np.arange(0.5, 1.105, 0.005, dtype=float)
fs = 16000
ir_length = 16000
n_jobs = 8
sinc_order = 4


def generate_tascar_project(position: float = 0.0) -> str:
    fn = f"ir_pos{position:.3f}"

    root = objectify.Element(
        "session",
        attribution="Felix Holzmüller (autogen)",
    )
    scene = objectify.SubElement(
        root,
        "scene",
        name=fn,
        guiscale="6",
        ismorder="2",
    )

    # Add secondary source at specified coordinates and gain
    for i, coords in enumerate(sec_src_coords):
        source = objectify.SubElement(scene, "source", name=f"sec_source_{i}")
        snd = objectify.SubElement(
            source,
            "sound",
            type="omni",
            gain=str(sec_src_gain),
            sincorder=str(sinc_order),
        )
        pos = objectify.SubElement(source, "position")._setText(
            f"0 {coords[0]} {coords[1]} {coords[2]}"
        )

    # Add hrtf receiver with specified orientation
    rec = objectify.SubElement(
        scene,
        "receiver",
        name="hrtf_receiver",
        type="hrtf",
        nf_filter="true",
        sincorder=str(sinc_order),
        # prewarpingmode="2",
    )
    pos = objectify.SubElement(rec, "position")._setText(f"0 {position} 0 0")

    room_walls = objectify.SubElement(
        scene,
        "facegroup",
        name="room_walls",
        shoeboxwalls = "2.5 1.5 1",
        material = "window"
    )
    room_ceiling = objectify.SubElement(
        scene,
        "facegroup",
        name="room_ceiling",
        shoeboxceiling = "2.5 1.5 1",
        material = "carpet_on_concrete"
    )
    room_floor = objectify.SubElement(
        scene,
        "facegroup",
        name="room_floor",
        shoeboxfloor = "2.5 1.5 1",
        material = "carpet_on_concrete"
    )

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
    irs_raw = sf.read(f"{fn}.wav")[0]

    n_sec = len(sec_src_coords)

    irs = np.zeros((2, n_sec, ir_length))
    for i in range(n_sec):
        irs[:, i, :] = irs_raw[i * ir_length : (i + 1) * ir_length].T

    return irs


def render_tascar_project(position: float = 0.0):
    # Generate project file
    fn = generate_tascar_project(position)

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
        log.write(f"Scene at position {position} rendered successfully.\n")
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
    impulse = np.zeros((2 * ir_length, 2))
    impulse[0, 0] = 1.0
    impulse[ir_length, 1] = 1.0
    sf.write("impulse.wav", impulse, fs)

    results = Parallel(n_jobs=n_jobs)(
        delayed(render_tascar_project)(position) for position in tqdm(positions_to_test)
    )

    # Delete impulse after use
    os.remove("impulse.wav")

    # Convert results in a format, where irs for specific positions can easily be extracted
    irs = np.array(results)

    # Save results to file
    if not os.path.exists("../results/data/TASCAR_IRs"):
        os.makedirs("../results/data/TASCAR_IRs")
    np.savez(
        "../results/data/TASCAR_IRs/measured_irs_sim.npz",
        irs=irs,
        positions=positions_to_test * 100,  # save in cm
        fs=fs,
    )


if __name__ == "__main__":
    main()
