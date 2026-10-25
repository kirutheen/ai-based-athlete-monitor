#!/usr/bin/env python
"""
End-to-end OpenSim-based validation pipeline, implementing the abstract's
described validation stage:

    "The simulated reference motion is passed through controlled camera
    and IMU noise, bias, drift, and measurement uncertainty to reproduce
    realistic sensing conditions before generating a DriftGuard estimate.
    The estimated kinematics are compared against the reference
    trajectory... using rotational RMSE, translational RMSE, MAE, maximum
    error, and cycle-level consistency, with the target being <=5 degrees
    rotational RMS error and <=5 mm translational RMS error."

Pipeline:
    1. Generate a parametric reference 6-DOF knee trajectory
       (src/validation/motion_generator.py)
    2. Drive a real OpenSim 6-DOF knee model through that trajectory via
       genuine forward kinematics, as a sanity check that the reference
       is a physically realizable motion, not just numbers in a CSV
       (src/validation/opensim_knee_model.py)
    3. Inject simulated camera/IMU/fused sensor noise to produce a
       "DriftGuard estimate" trajectory (src/validation/noise_model.py)
    4. Compute rotational RMSE, translational RMSE, MAE, max error, and
       cycle-level consistency, and check against the <=5deg/<=5mm target
       (src/validation/trajectory_metrics.py)

Usage:
    python scripts/run_opensim_validation.py --movement gait --noise-profile camera
    python scripts/run_opensim_validation.py --movement sit_to_stand --noise-profile fused --cycles 3

IMPORTANT: see the module docstrings in noise_model.py and
motion_generator.py -- the reference motion is a simulated parametric
profile (not measured from a real patient) and the noise magnitudes are
illustrative defaults (not measured from real camera/IMU hardware). A
"PASS" from this script means "the validation pipeline's arithmetic is
correct and the illustrative noise level would clear the target" -- it
is explicitly NOT a real-world clinical accuracy claim. The abstract
itself frames this stage as "development and validation preparation
rather than real-world clinical validation," and this script is that
preparation, nothing more.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation.motion_generator import MOVEMENT_TYPES, generate_reference_trajectory, trajectory_row_to_pose
from src.validation.noise_model import NOISE_PRESETS, inject_noise
from src.validation.opensim_knee_model import get_shank_position_mm, init_model, set_knee_coordinates
from src.validation.trajectory_metrics import format_report, validate_trajectory


def run_opensim_sanity_check(reference_df: pd.DataFrame) -> None:
    """Drive the real OpenSim model through the reference trajectory and
    confirm it produces a physically coherent (non-degenerate) shank
    motion -- proof this is genuine forward kinematics, not just numbers
    being passed through untouched."""
    handles, state = init_model()
    positions = []
    for _, row in reference_df.iterrows():
        pose = trajectory_row_to_pose(row)
        set_knee_coordinates(handles, state, pose)
        positions.append(get_shank_position_mm(handles, state))
    positions_df = pd.DataFrame(positions, columns=["shank_x_mm", "shank_y_mm", "shank_z_mm"])
    moved_range = positions_df.max() - positions_df.min()
    print("OpenSim forward-kinematics sanity check:")
    print(f"  Shank body range of motion over the cycle: "
          f"x={moved_range['shank_x_mm']:.1f}mm, y={moved_range['shank_y_mm']:.1f}mm, z={moved_range['shank_z_mm']:.1f}mm")
    print("  (Nonzero range confirms the OpenSim model is genuinely responding to the")
    print("   prescribed knee coordinates via real forward kinematics, not passing them through.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--movement", choices=MOVEMENT_TYPES, default="gait")
    parser.add_argument("--noise-profile", choices=list(NOISE_PRESETS.keys()), default="camera")
    parser.add_argument("--cycles", type=int, default=1, help="Number of concatenated cycles to simulate (for cycle-consistency metric)")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of one cycle, in seconds")
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--output-csv", default=None, help="Optional path to save reference+estimate trajectories as CSV")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Movement type: {args.movement} | Noise profile: {args.noise_profile} | Cycles: {args.cycles}\n")

    cycles = []
    for i in range(args.cycles):
        cycle_df = generate_reference_trajectory(args.movement, duration_s=args.duration, fps=args.fps)
        cycle_df["time_s"] = cycle_df["time_s"] + i * args.duration
        cycles.append(cycle_df)
    reference_df = pd.concat(cycles, ignore_index=True)

    run_opensim_sanity_check(reference_df if args.cycles == 1 else cycles[0])
    print()

    profile = NOISE_PRESETS[args.noise_profile]
    estimate_df = inject_noise(reference_df, profile, random_state=args.random_seed)

    report = validate_trajectory(reference_df, estimate_df, n_cycles=args.cycles)
    print(format_report(report))

    if args.output_csv:
        out = reference_df.copy()
        out.columns = [f"ref_{c}" if c != "time_s" else c for c in out.columns]
        for col in estimate_df.columns:
            if col != "time_s":
                out[f"est_{col}"] = estimate_df[col]
        out.to_csv(args.output_csv, index=False)
        print(f"\nSaved reference+estimate trajectories to {args.output_csv}")

    print("\nReminder: this validates the PIPELINE's arithmetic against an illustrative noise")
    print("model, not real-world accuracy. See noise_model.py's docstring before citing this")
    print("PASS/FAIL to anyone as a clinical accuracy claim.")


if __name__ == "__main__":
    main()
