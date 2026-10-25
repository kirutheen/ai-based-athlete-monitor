"""
Sport-Specific Movement Architecture and Profiles.

Defines sport-specific biomechanical parameters, movement event priorities, threshold overrides,
and clinical focus areas for: Basketball, Football/Soccer, Cricket, Volleyball, and Tennis.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SportProfile:
    sport_id: str
    display_name: str
    icon_name: str
    description: str
    primary_movements: list[str]
    primary_joint_focus: list[str]
    valgus_warning_threshold_deg: float
    flexion_warning_threshold_deg: float
    lsi_asymmetry_threshold_percent: float
    clinical_notes: str


SPORT_PROFILES: dict[str, SportProfile] = {
    "basketball": SportProfile(
        sport_id="basketball",
        display_name="Basketball",
        icon_name="basketball",
        description="Emphasizes high-impact rebound/layup landing mechanics, deceleration, and dynamic knee valgus.",
        primary_movements=["Drop Jump Landing", "Deceleration", "Side-step Cut"],
        primary_joint_focus=["Knee Valgus / FPPA", "Landing Flexion", "Trunk Lateral Control"],
        valgus_warning_threshold_deg=10.0,
        flexion_warning_threshold_deg=45.0,
        lsi_asymmetry_threshold_percent=15.0,
        clinical_notes="Monitor shallow landing knee flexion (<45°) combined with dynamic knee valgus during bilateral landings.",
    ),
    "football": SportProfile(
        sport_id="football",
        display_name="Football / Soccer",
        icon_name="football",
        description="Emphasizes single-leg cutting, high-speed directional changes, acceleration/deceleration, and pelvic stability.",
        primary_movements=["Single-leg Cut", "Deceleration", "Unanticipated Pivot"],
        primary_joint_focus=["Pelvic Obliquity / Hip Drop", "Single-leg Knee Flexion", "Trunk Lean"],
        valgus_warning_threshold_deg=8.0,
        flexion_warning_threshold_deg=40.0,
        lsi_asymmetry_threshold_percent=12.0,
        clinical_notes="Focus on pelvic drop and lateral trunk displacement during single-leg plant and cut maneuvers.",
    ),
    "cricket": SportProfile(
        sport_id="cricket",
        display_name="Cricket",
        icon_name="cricket",
        description="Emphasizes pace bowling front-leg delivery stride landing, trunk lateral flexion, and batting stride stability.",
        primary_movements=["Delivery Stride Landing", "Running between Wickets", "Batting Stride"],
        primary_joint_focus=["Front-knee Extension at Contact", "Trunk Lateral Flexion", "Pelvic Tilt"],
        valgus_warning_threshold_deg=9.0,
        flexion_warning_threshold_deg=35.0,
        lsi_asymmetry_threshold_percent=15.0,
        clinical_notes="Evaluate front knee stiffness and trunk hyperextension/lateral flexion during the fast bowling delivery stride.",
    ),
    "volleyball": SportProfile(
        sport_id="volleyball",
        display_name="Volleyball",
        icon_name="volleyball",
        description="Emphasizes repetitive block/spike jump landings, bilateral landing symmetry, and deep squat flexion.",
        primary_movements=["Block Landing", "Spike Approach Landing", "Lateral Shuffling"],
        primary_joint_focus=["Bilateral Knee Flexion LSI", "Peak Valgus after Contact", "Ankle Dorsiflexion"],
        valgus_warning_threshold_deg=10.0,
        flexion_warning_threshold_deg=50.0,
        lsi_asymmetry_threshold_percent=10.0,
        clinical_notes="High frequency of landing events requires monitoring bilateral limb symmetry index (LSI < 90%).",
    ),
    "tennis": SportProfile(
        sport_id="tennis",
        display_name="Tennis",
        icon_name="tennis",
        description="Emphasizes multi-directional court movement, split-step landing, lateral push-off, and trunk rotation.",
        primary_movements=["Split-step Landing", "Lateral Push-off", "Serve Recovery"],
        primary_joint_focus=["Hip Abduction Angle", "Trunk Rotation", "Ankle Inversion/Eversion"],
        valgus_warning_threshold_deg=8.0,
        flexion_warning_threshold_deg=42.0,
        lsi_asymmetry_threshold_percent=12.0,
        clinical_notes="Assess single-leg stabilization and ankle dorsiflexion range of motion during sudden lateral braking.",
    ),
}


@dataclass
class MovementConfig:
    movement_id: str
    sport_id: str
    display_name: str
    category: str  # Landing, Cutting, Squatting, Single-Leg, Acceleration
    description: str
    primary_joints: list[str]
    expected_duration_s: float


MOVEMENT_REGISTRY: dict[str, list[MovementConfig]] = {
    "basketball": [
        MovementConfig("drop_jump", "basketball", "Drop Jump Landing", "Landing", "High-impact vertical rebound landing evaluating dynamic valgus.", ["Knee FPPA", "Knee Flexion", "Trunk Lean"], 3.0),
        MovementConfig("side_cut", "basketball", "Side-Step Cut", "Cutting", "Unanticipated lateral plant-and-cut maneuver evaluating hip drop.", ["Hip Abduction", "Pelvic Obliquity", "Knee Valgus"], 2.5),
        MovementConfig("squat", "basketball", "Double-Leg Squat", "Squatting", "Deep bilateral flexion evaluating depth symmetry.", ["Knee Flexion", "Ankle Dorsiflexion"], 4.0),
    ],
    "football": [
        MovementConfig("single_leg_cut", "football", "Single-Leg Cut", "Cutting", "High-speed plant and change of direction on grass/turf.", ["Pelvic Obliquity", "Trunk Lean", "Hip Abduction"], 2.5),
        MovementConfig("deceleration", "football", "Single-Leg Deceleration", "Single-Leg", "Sudden braking maneuver evaluating eccentric knee control.", ["Knee Flexion", "Ankle Dorsiflexion"], 3.0),
        MovementConfig("header_landing", "football", "Header Landing", "Landing", "Unilateral jump landing following aerial duel.", ["Knee FPPA", "Valgus Duration"], 2.0),
    ],
    "cricket": [
        MovementConfig("delivery_stride", "cricket", "Bowling Delivery Stride", "Single-Leg", "Front leg plant at delivery stride in fast bowling.", ["Front Knee Extension", "Trunk Lateral Flexion"], 3.5),
        MovementConfig("wicket_sprint", "cricket", "Turning Between Wickets", "Cutting", "180-degree turn and push-off at crease.", ["Ankle Dorsiflexion", "Knee Valgus"], 3.0),
    ],
    "volleyball": [
        MovementConfig("block_landing", "volleyball", "Block Jump Landing", "Landing", "Bilateral landing following net block.", ["Knee Flexion LSI", "Valgus LSI"], 2.5),
        MovementConfig("spike_landing", "volleyball", "Spike Approach Landing", "Landing", "Asymmetric landing following attack jump.", ["Knee FPPA", "Ankle Dorsiflexion"], 2.5),
    ],
    "tennis": [
        MovementConfig("split_step", "tennis", "Split-Step Landing", "Single-Leg", "Braking split-step before court reaction.", ["Ankle Dorsiflexion", "Hip Abduction"], 2.0),
        MovementConfig("lateral_pushoff", "tennis", "Lateral Push-off", "Cutting", "Open-stance forehand plant and push-off.", ["Pelvic Rotation", "Knee FPPA"], 2.5),
    ],
}


def get_sport_profile(sport_id: str) -> SportProfile:
    """Retrieve sport profile by ID; defaults to Basketball if unrecognized."""
    clean_id = (sport_id or "basketball").lower()
    return SPORT_PROFILES.get(clean_id, SPORT_PROFILES["basketball"])


def list_sports() -> list[dict]:
    """Return all available sport profiles as serializable dicts."""
    return [vars(profile) for profile in SPORT_PROFILES.values()]


def list_movements(sport_id: str | None = None) -> list[dict]:
    """Return available movement taxonomy for a sport or all sports."""
    if sport_id:
        movs = MOVEMENT_REGISTRY.get(sport_id.lower(), MOVEMENT_REGISTRY["basketball"])
        return [vars(m) for m in movs]
    all_movs = []
    for mov_list in MOVEMENT_REGISTRY.values():
        all_movs.extend([vars(m) for m in mov_list])
    return all_movs

