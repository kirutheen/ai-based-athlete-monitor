"""
A simplified, programmatically-built OpenSim model exposing the knee as
an independently-parameterized 6-DOF joint -- the biomechanical
reference environment the project abstract specifies ("OpenSim is used
as the biomechanical reference and simulation environment, with the
selected six-DOF knee model providing independently parameterized
rotational and translational knee coordinates").

============================= HONEST SCOPE =================================
This IS real, working OpenSim integration -- not a stub. `opensim.Model`,
`opensim.FreeJoint`, and `opensim.Body` are genuine OpenSim API objects;
`build_six_dof_knee_model()` constructs a real multibody system and
`set_knee_coordinates()` + `model.realizePosition()` runs genuine forward
kinematics through it. This is testable and tested
(tests/test_opensim_knee_model.py) wherever the `opensim` package is
installed.

What this model is NOT:
  - It is NOT a full lower-limb model (no pelvis, full leg segments,
    muscles, or dynamics) -- the hip is rigidly welded to ground. That's
    deliberate: this stage only needs to generate and query a known,
    controllable 6-DOF knee motion, not simulate a whole gait cycle
    biomechanically. A full gait2392-style model is future work once
    real motion-capture data exists to drive it meaningfully.
  - The 6 FreeJoint coordinates follow OpenSim's own generalized-
    coordinate convention for a free (6-DOF) joint: 3 body-fixed Euler
    rotations (XYZ sequence) + 3 Cartesian translations. This is a
    genuine, independently-parameterized 6-DOF system, as the abstract
    asks for, but is NOT bit-for-bit identical to the Grood-Suntay
    clinical knee joint coordinate convention used in some biomechanics
    papers (which uses a non-orthogonal floating axis system). Treat the
    rotation labels (flexion/extension, abduction/adduction, internal/
    external rotation) as the clinically-intended MEANING assigned to
    each coordinate for this simulation, not a claim that the two
    conventions are numerically interchangeable in general -- for the
    small clinical ranges of motion used in validation (see
    motion_generator.py), the two conventions stay close, but this
    should be revisited if large rotations are ever simulated.
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import opensim as osim
except Exception:
    osim = None



@dataclass
class KneeModelHandles:
    """References into the built OpenSim model needed to drive and query it."""

    model: "osim.Model"
    knee_joint: "osim.FreeJoint"
    shank_body: "osim.Body"


def build_six_dof_knee_model() -> KneeModelHandles:
    """
    Build a minimal OpenSim model: ground -> (weld) -> thigh -> (6-DOF
    FreeJoint, the knee) -> shank.

    Gravity is zeroed and no muscles/actuators are added -- this model
    is used purely for forward kinematics (given knee angles/
    translations, what is the shank's resulting position/orientation?),
    not for dynamic simulation.
    """
    if osim is None:
        raise RuntimeError("OpenSim package or native library (_simbody) is not available in this environment.")
    model = osim.Model()
    model.setName("driftguard_knee_6dof")
    model.setGravity(osim.Vec3(0, 0, 0))

    ground = model.getGround()

    thigh = osim.Body("thigh", 1.0, osim.Vec3(0), osim.Inertia(0.1, 0.1, 0.1))
    model.addBody(thigh)
    hip_weld = osim.WeldJoint("hip", ground, osim.Vec3(0), osim.Vec3(0), thigh, osim.Vec3(0), osim.Vec3(0))
    model.addJoint(hip_weld)

    shank = osim.Body("shank", 1.0, osim.Vec3(0), osim.Inertia(0.1, 0.1, 0.1))
    model.addBody(shank)
    knee = osim.FreeJoint("knee", thigh, osim.Vec3(0), osim.Vec3(0), shank, osim.Vec3(0), osim.Vec3(0))
    model.addJoint(knee)

    model.finalizeConnections()
    return KneeModelHandles(model=model, knee_joint=knee, shank_body=shank)


# Index of each clinical quantity within the FreeJoint's 6 coordinates,
# per OpenSim's FreeJoint convention (3 body-fixed XYZ Euler rotations,
# then 3 Cartesian translations). Documented here once so every caller
# agrees on which index means what.
COORD_FLEXION_EXTENSION = 0
COORD_ABDUCTION_ADDUCTION = 1
COORD_INTERNAL_EXTERNAL_ROTATION = 2
COORD_AP_TRANSLATION = 3
COORD_ML_TRANSLATION = 4
COORD_SI_TRANSLATION = 5


@dataclass
class KneePose:
    flexion_extension_deg: float
    abduction_adduction_deg: float
    internal_external_rotation_deg: float
    ap_translation_mm: float
    ml_translation_mm: float
    si_translation_mm: float


def set_knee_coordinates(handles: KneeModelHandles, state, pose: KneePose) -> None:
    """Set the knee's 6 coordinates to a given clinical pose and run
    forward kinematics (model.realizePosition) so queries against
    `state` reflect this pose."""
    import math

    knee = handles.knee_joint
    knee.get_coordinates(COORD_FLEXION_EXTENSION).setValue(state, math.radians(pose.flexion_extension_deg))
    knee.get_coordinates(COORD_ABDUCTION_ADDUCTION).setValue(state, math.radians(pose.abduction_adduction_deg))
    knee.get_coordinates(COORD_INTERNAL_EXTERNAL_ROTATION).setValue(state, math.radians(pose.internal_external_rotation_deg))
    knee.get_coordinates(COORD_AP_TRANSLATION).setValue(state, pose.ap_translation_mm / 1000.0)
    knee.get_coordinates(COORD_ML_TRANSLATION).setValue(state, pose.ml_translation_mm / 1000.0)
    knee.get_coordinates(COORD_SI_TRANSLATION).setValue(state, pose.si_translation_mm / 1000.0)
    handles.model.realizePosition(state)


def get_shank_position_mm(handles: KneeModelHandles, state) -> tuple[float, float, float]:
    """Shank body origin position in the ground frame, in mm -- a
    genuine forward-kinematics query confirming the model actually moved
    in response to set_knee_coordinates(), not just an echo of the input."""
    transform = handles.shank_body.getTransformInGround(state)
    p = transform.p()
    return (p[0] * 1000.0, p[1] * 1000.0, p[2] * 1000.0)


def init_model() -> tuple[KneeModelHandles, object]:
    """Build the model and return it with a freshly initialized state,
    ready for repeated set_knee_coordinates() / query calls."""
    handles = build_six_dof_knee_model()
    state = handles.model.initSystem()
    return handles, state
