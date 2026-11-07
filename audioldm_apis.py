"""
RANDOM_API: `audioldm` with --mode generation, empty text prompt

VARIATION_API: `audioldm` with --mode transfer, a real input file, empty text prompt, and --transfer_strength = this iteration's variation degree (v)
"""

__version__ = "v5"

import glob
import os
import random
import shutil
import subprocess
import sys
import uuid

from config import PEConfig


def _find_audioldm_executable() -> str:
    """Locate the `audioldm` console script"""
    on_path = shutil.which("audioldm")
    if on_path:
        return on_path  # if script is on path

    raise FileNotFoundError(
        "Could not find the `audioldm` executable on PATH or next to sys.executable ({sys.executable}). Confirm `pip install audioldm` succeeded in this exact environment:"
        "run `pip show audioldm` and `python -c \"import audioldm\"` using the same interpreter that runs main.py."
    )


_AUDIOLDM_BIN = _find_audioldm_executable()


def _run_audioldm_cli(args: list) -> None:
    """Runs `audioldm <args>` and raises if it fails. args is a list of CLI tokens."""
    cmd = [_AUDIOLDM_BIN] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"audioldm CLI failed:\n{result.stderr}")


def _find_wavs(directory: str) -> list:
    """
    Recursively find every .wav file under `directory`. AudioLDM's CLI may nest generated output one or more levels below the -s path given (e.g. inside an auto-named per-run subfolder) rather than writing
    directly into it
    """
    return sorted(glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True))


def random_api(n: int, cfg: PEConfig, save_dir: str) -> list:
    """
    Generates n synthetic audios (starting from pure Gaussian noise inside AudioLDM's diffusion loop) and returns their file paths

    generates one sample per audioldm call, each into its own uniquely-named directory
    """
    os.makedirs(save_dir, exist_ok=True)
    all_paths = []
    for i in range(n):
        call_dir = os.path.join(save_dir, f"_call_{uuid.uuid4().hex}")
        os.makedirs(call_dir, exist_ok=False)
        _run_audioldm_cli([
            "--mode", "generation",
            "-t", "",  # empty prompt = unconditional
            "-n", "1",  # one sample per call, no batching
            "-s", call_dir,
            "--model_name", cfg.audioldm_model_name,
            "-dur", str(cfg.audioldm_duration_s),
            "--ddim_steps", str(cfg.audioldm_ddim_steps),
            "--seed", str(random.randint(0, 2**31 - 1)),  # distinct seed per sample
        ])
        produced = _find_wavs(call_dir)
        if not produced:
            listing = "\n".join(f"  {p}" for p in glob.glob(os.path.join(call_dir, "**", "*"), recursive=True)) or "  (directory is completely empty)"
            raise RuntimeError(
                f"audioldm produced 0 .wav files in {call_dir} despite exiting successfully (sample {i+1}/{n}). Actual contents of that directory:\n{listing}"
            )
        if len(produced) > 1:
            print(f"  WARNING: audioldm produced {len(produced)} files for a single -n 1 call in {call_dir} -- using the first one, ignoring the rest.")

        final_path = os.path.join(save_dir, f"{uuid.uuid4().hex}.wav")  # move file from temp subdirectory to save_dir
        os.rename(produced[0], final_path)
        shutil.rmtree(call_dir, ignore_errors=True)
        all_paths.append(final_path)

    assert len(all_paths) == n, f"expected {n} generated samples, got {len(all_paths)}"     # verify n paths were collected
    assert len(set(all_paths)) == n, "random_api produced duplicate file paths"    # verify no duplicate paths

    return all_paths


def variation_api(parent_path: str, transfer_strength: float, cfg: PEConfig, save_dir: str) -> str:
    """
    Takes a single existing audio file (a parent selected by the DP Nearest Neighbors Histogram), partially noises it (based on transfer_strength), and denoises it back into one offspring sample
    """
    os.makedirs(save_dir, exist_ok=True)
    call_dir = os.path.join(save_dir, f"_call_{uuid.uuid4().hex}")
    os.makedirs(call_dir, exist_ok=False)   # exist_ok=False: new empty directory

    _run_audioldm_cli([
        "--mode", "transfer",
        "-f", parent_path,                           # the parent sample being varied
        "-t", "",                                    # empty prompt: audio-to-audio, no text
        "--transfer_strength", str(transfer_strength),
        "-s", call_dir,
        "--model_name", cfg.audioldm_model_name,
        "-dur", str(cfg.audioldm_duration_s),
        "--ddim_steps", str(cfg.audioldm_ddim_steps),
        "--seed", str(random.randint(0, 2**31 - 1)),  # distinct noise per call
    ])

    produced_list = _find_wavs(call_dir)
    if not produced_list:
        listing = "\n".join(f"  {p}" for p in glob.glob(os.path.join(call_dir, "**", "*"), recursive=True)) or "  (directory is completely empty)"
        raise RuntimeError(
            f"audioldm --mode transfer produced 0 .wav files in {call_dir} despite exiting successfully (input was {parent_path}). Actual contents:\n{listing}"
        )
    if len(produced_list) > 1:
        print(f"  WARNING: audioldm produced {len(produced_list)} files for a single transfer call in {call_dir} -- using the first one, ignoring the rest.")

    # move output out of its throwaway per-call subfolder into save_dir
    final_path = os.path.join(save_dir, f"{uuid.uuid4().hex}.wav")
    os.rename(produced_list[0], final_path)
    shutil.rmtree(call_dir, ignore_errors=True)
    return final_path
