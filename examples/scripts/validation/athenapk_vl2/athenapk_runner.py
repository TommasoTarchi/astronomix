"""
Running AthenaPK and reading its output, for the VL2 validation.

Small helpers to write an AthenaPK input file for a validation case, run the
``athenaPK`` binary (located through the ``ATHENAPK_BIN`` / ``ATHENAPK_GPU_BIN``
environment variables), read Parthenon's HDF5 outputs into astronomix's array
layout, and start AthenaPK from an arbitrary initial condition by writing it into
a Parthenon restart file.
"""

# general
import glob
import os
import shutil
import subprocess

# numerics
import numpy as np

# file io
import h5py


def athenapk_binary(gpu: bool = False) -> str:
    """
    The AthenaPK executable to use (``ATHENAPK_GPU_BIN`` for the GPU build,
    ``ATHENAPK_BIN`` otherwise).

    Args:
        gpu: Return the GPU build.

    Returns:
        The path of the executable.
    """
    variable = "ATHENAPK_GPU_BIN" if gpu else "ATHENAPK_BIN"
    path = os.environ.get(variable)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(
            f"Set {variable} to an athenaPK executable (a double-precision build; "
            "see this directory's README.md)."
        )
    return path


def write_input_file(case, path: str):
    """
    Write the AthenaPK input file of a validation case.

    Every case runs the VL2 integrator on a single meshblock with two ghost
    cells and writes the initial and final state (conserved and primitive) as
    well as a restart file of the initial state.

    Args:
        case: The validation case (see ``cases.py``).
        path: Where to write the input file.
    """
    mesh_lines = []
    meshblock_lines = []
    for axis in range(3):
        active = axis < case.dimensionality
        boundary = case.boundary if active else "periodic"
        mesh_lines += [
            f"nx{axis + 1} = {case.num_cells[axis] if active else 1}",
            f"x{axis + 1}min = {case.lower_corner[axis] if active else -0.5}",
            f"x{axis + 1}max = {case.upper_corner[axis] if active else 0.5}",
            f"ix{axis + 1}_bc = {boundary}",
            f"ox{axis + 1}_bc = {boundary}",
        ]
        meshblock_lines.append(f"nx{axis + 1} = {case.num_cells[axis] if active else 1}")

    problem_lines = [f"{key} = {value}" for key, value in case.problem_parameters.items()]
    floor_lines = []
    if case.floors is not None:
        floor_lines = [f"dfloor = {case.floors[0]}", f"pfloor = {case.floors[1]}"]

    text = "\n".join(
        [
            "<job>",
            f"problem_id = {case.problem_generator}",
            f"<problem/{case.problem_generator}>",
            *problem_lines,
            "<parthenon/mesh>",
            "refinement = none",
            "nghost = 2",
            *mesh_lines,
            "<parthenon/meshblock>",
            *meshblock_lines,
            "<parthenon/time>",
            "integrator = vl2",
            f"cfl = {case.cfl}",
            f"tlim = {case.end_time}",
            "nlim = -1",
            "<hydro>",
            f"fluid = {'glmmhd' if case.mhd else 'euler'}",
            "eos = adiabatic",
            f"riemann = {case.riemann_solver}",
            f"reconstruction = {case.reconstruction}",
            f"gamma = {case.gamma!r}",
            f"glmmhd_alpha = {case.glm_alpha}",
            f"glmmhd_source = {'dedner_extended' if case.glm_extended_source else 'dedner_plain'}",
            f"first_order_flux_correct = {'true' if case.first_order_flux_correction else 'false'}",
            # scratch pads in global memory: single-meshblock pencils are too
            # long for shared memory (this does not change any result)
            "scratch_level = 1",
            *floor_lines,
            "<parthenon/output0>",
            "file_type = hdf5",
            f"dt = {case.end_time}",
            "variables = cons, prim",
            "<parthenon/output1>",
            "file_type = rst",
            f"dt = {case.end_time}",
            "",
        ]
    )
    with open(path, "w") as file:
        file.write(text)


def run_athenapk(arguments, run_directory: str, gpu: bool = False, threads: int = 16) -> str:
    """
    Run AthenaPK in ``run_directory`` (cleared of old outputs first).

    Args:
        arguments: The command-line arguments after the executable.
        run_directory: The working directory.
        gpu: Use the GPU build.
        threads: OpenMP threads of the CPU build.

    Returns:
        AthenaPK's standard output.
    """
    os.makedirs(run_directory, exist_ok=True)
    for old_output in glob.glob(os.path.join(run_directory, "parthenon.*")):
        os.remove(old_output)
    environment = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="false")
    result = subprocess.run(
        [athenapk_binary(gpu), *arguments],
        cwd=run_directory,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(f"athenaPK failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result.stdout


def read_output(path: str, variable: str = "cons"):
    """
    Read a single-meshblock Parthenon HDF5 output into astronomix's layout.

    Args:
        path: The ``.phdf`` / ``.rhdf`` file.
        variable: ``"cons"`` or ``"prim"``.

    Returns:
        ``(state, time, cycle)`` with the state shaped ``(num_vars, nx[, ny[, nz]])``.
    """
    with h5py.File(path) as file:
        data = file[variable][...]
        info = dict(file["Info"].attrs)
    if data.shape[0] != 1:
        raise ValueError("The validation expects a single meshblock.")
    dimensionality = int(info["NumDims"])
    # Parthenon stores (block, variable, z, y, x)
    state = np.transpose(data[0], (0, 3, 2, 1))
    state = state[(slice(None),) + (slice(None),) * dimensionality + (0,) * (3 - dimensionality)]
    return state, float(info["Time"]), int(info["NCycle"])


def final_output(run_directory: str, variable: str = "cons"):
    """The final state of a run (see :func:`read_output`)."""
    return read_output(os.path.join(run_directory, "parthenon.out0.final.phdf"), variable)


def initial_output(run_directory: str, variable: str = "cons"):
    """The initial state of a run (see :func:`read_output`)."""
    return read_output(os.path.join(run_directory, "parthenon.out0.00000.phdf"), variable)


def write_restart_with_state(template_restart: str, conserved_state, path: str) -> str:
    """
    Write a Parthenon restart file that holds ``conserved_state`` as the state
    at ``t = 0``, based on a restart file of the same mesh and configuration.

    The stored time step and hyperbolic time step are reset so that AthenaPK
    re-derives both (and the GLM cleaning speed) from the injected state.

    Args:
        template_restart: A restart file of a run with the same mesh.
        conserved_state: The conserved state, astronomix layout.
        path: Where to write the new restart file.

    Returns:
        ``path``.
    """
    shutil.copyfile(template_restart, path)
    state = np.asarray(conserved_state)
    state = state.reshape(state.shape + (1,) * (4 - state.ndim))
    parthenon_layout = np.transpose(state, (0, 3, 2, 1))[None]
    largest_float = np.finfo(np.float64).max
    with h5py.File(path, "r+") as file:
        if file["cons"].shape != parthenon_layout.shape:
            raise ValueError(f"State shape {parthenon_layout.shape} does not match {file['cons'].shape}.")
        file["cons"][...] = parthenon_layout
        file["Info"].attrs["Time"] = 0.0
        file["Info"].attrs["NCycle"] = np.int32(0)
        file["Info"].attrs["dt"] = largest_float
        if "Hydro/dt_hyp" in file["Params"].attrs:
            file["Params"].attrs["Hydro/dt_hyp"] = largest_float
    return path


def run_case(case, run_directory: str, initial_conserved_state=None, gpu: bool = False) -> str:
    """
    Run a validation case with AthenaPK, from its problem generator or — when
    ``initial_conserved_state`` is given — from that state (via a restart file).

    Args:
        case: The validation case.
        run_directory: The working directory.
        initial_conserved_state: Optional initial conserved state (astronomix layout).
        gpu: Use the GPU build.

    Returns:
        ``run_directory``.
    """
    os.makedirs(run_directory, exist_ok=True)
    input_file = os.path.join(run_directory, "case.in")
    write_input_file(case, input_file)

    if initial_conserved_state is None:
        run_athenapk(["-i", input_file], run_directory, gpu=gpu)
        return run_directory

    template_directory = os.path.join(run_directory, "template")
    run_athenapk(["-i", input_file, "parthenon/time/nlim=0"], template_directory, gpu=gpu, threads=4)
    restart = write_restart_with_state(
        os.path.join(template_directory, "parthenon.out1.00000.rhdf"),
        initial_conserved_state,
        os.path.join(run_directory, "initial_state.rhdf"),
    )
    # The restart file carries the template's input (including its cycle
    # limit of zero), so the limit is lifted on the command line.
    run_athenapk(["-r", restart, "parthenon/time/nlim=-1"], run_directory, gpu=gpu)
    return run_directory
