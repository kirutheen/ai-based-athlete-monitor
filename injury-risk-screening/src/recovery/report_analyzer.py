"""
Medical Report Analyzer for Athlete Recovery Plans.
Parses medical report text and extracts structured recovery parameters.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List


def analyze_medical_report(text: str) -> Dict[str, Any]:
    """
    Analyzes medical report text to extract diagnosis, affected body part,
    phase, recovery duration, hydration, medications, nutrition items, and exercises.
    Guarantees structured dict with all required fields.
    """
    raw_lower = text.lower()

    # 1. Body Part Detection
    body_part = "Knee"
    if "ankle" in raw_lower:
        body_part = "Ankle"
    elif "shoulder" in raw_lower:
        body_part = "Shoulder"
    elif "hamstring" in raw_lower:
        body_part = "Hamstring"
    elif "hip" in raw_lower or "groin" in raw_lower:
        body_part = "Hip"
    elif "back" in raw_lower or "lumbar" in raw_lower:
        body_part = "Lower Back"
    elif "knee" in raw_lower or "acl" in raw_lower or "pcl" in raw_lower or "mcl" in raw_lower or "patellar" in raw_lower:
        body_part = "Knee"

    # 2. Diagnosis Detection
    diagnosis = f"Post-injury {body_part} Rehabilitation"
    diag_patterns = [
        (r"(acl|mcl|pcl|lcl)\s*(sprain|tear|strain|reconstruction|injury)?", lambda m: f"{m.group(1).upper()} {m.group(2).capitalize() if m.group(2) else 'Injury'}"),
        (r"(patellar|quadriceps|achilles|rotator cuff)\s*(tendinopathy|tendinitis|tear|strain)", lambda m: f"{m.group(1).capitalize()} {m.group(2).capitalize()}"),
        (r"(grade\s*[iI|1-3]+)\s*([a-zA-Z\s]+(sprain|strain|tear))", lambda m: f"{m.group(2).strip().title()} ({m.group(1).upper()})"),
        (r"diagnosis[:\s\-\–]+([^\.\n\,]+)", lambda m: m.group(1).strip().title()),
    ]
    for pattern, formatter in diag_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted = formatter(match)
            if len(extracted) > 3:
                diagnosis = extracted
                break

    # 3. Phase Detection
    phase = "Acute Recovery"
    if "sub-acute" in raw_lower or "rehab phase 2" in raw_lower or "strengthening" in raw_lower:
        phase = "Sub-acute Rehabilitation"
    elif "return to play" in raw_lower or "sport specific" in raw_lower:
        phase = "Return to Play"
    elif "acute" in raw_lower or "post-op" in raw_lower or "immobilization" in raw_lower:
        phase = "Acute Protection & Recovery"

    # 4. Recovery Duration Detection
    duration_days = 30
    duration_source = "estimated"
    dur_match = re.search(r"(\d+)\s*(days|weeks|months)", text, re.IGNORECASE)
    if dur_match:
        val = int(dur_match.group(1))
        unit = dur_match.group(2).lower()
        if "week" in unit:
            duration_days = val * 7
        elif "month" in unit:
            duration_days = val * 30
        else:
            duration_days = val
        duration_source = "explicit"

    # 5. Hydration Target
    hydration_target = 8
    hyd_match = re.search(r"(\d+)\s*(glasses|liters|l)\s*(of water|hydration|daily)", text, re.IGNORECASE)
    if hyd_match:
        val = int(hyd_match.group(1))
        if "liter" in hyd_match.group(2).lower() or "l" == hyd_match.group(2).lower():
            hydration_target = val * 4  # approx 250ml glasses
        else:
            hydration_target = val

    # 6. Medications Extraction
    medications: List[Dict[str, Any]] = []
    if "ibuprofen" in raw_lower or "advil" in raw_lower or "nsaid" in raw_lower:
        medications.append({
            "name": "Ibuprofen",
            "dosage": "400mg",
            "frequency": "Twice daily after meals",
            "duration_days": min(7, duration_days)
        })
    if "acetaminophen" in raw_lower or "paracetamol" in raw_lower or "tylenol" in raw_lower:
        medications.append({
            "name": "Acetaminophen",
            "dosage": "500mg",
            "frequency": "Every 6 hours as needed for pain",
            "duration_days": min(10, duration_days)
        })
    if "curcumin" in raw_lower or "turmeric" in raw_lower or "anti-inflammatory supplement" in raw_lower:
        medications.append({
            "name": "Curcumin / Turmeric Extract",
            "dosage": "500mg",
            "frequency": "Once daily with food",
            "duration_days": min(21, duration_days)
        })

    if not medications:
        medications = [
            {
                "name": "Ibuprofen (NSAID)",
                "dosage": "400mg",
                "frequency": "Twice daily with meals",
                "duration_days": min(7, duration_days)
            },
            {
                "name": "Acetaminophen / Tylenol",
                "dosage": "500mg",
                "frequency": "As needed for breakthrough pain (max 3/day)",
                "duration_days": min(10, duration_days)
            }
        ]

    # 7. Nutrition Extraction
    nutrition_items = [
        {
            "title": "Protein Synthesis Target",
            "target": "1.6g - 2.0g per kg body weight daily",
            "duration_days": duration_days,
            "example_foods": [
                "Chicken Breast (31g protein / 100g)",
                "Greek Yogurt (10g protein / 100g)",
                "Eggs (6g protein / egg)",
                "Tofu or Edamame (15g protein / cup)",
                "Whey / Plant Protein Shake (25g per scoop)"
            ]
        },
        {
            "title": "Collagen & Vitamin C Co-factors",
            "target": "15g Hydrolyzed Collagen + 500mg Vitamin C 60 mins pre-rehab",
            "duration_days": min(21, duration_days),
            "example_foods": [
                "Bone Broth (1 cup)",
                "Oranges or Bell Peppers (high Vit C)",
                "Strawberries / Kiwis",
                "Hydrolyzed Collagen Peptides Powder"
            ]
        },
        {
            "title": "Anti-Inflammatory Omega-3 Fatty Acids",
            "target": "2,000mg EPA/DHA daily",
            "duration_days": duration_days,
            "example_foods": [
                "Wild Salmon or Sardines (2 servings/week)",
                "Chia Seeds (2 tbsp)",
                "Walnuts (1 handful / 30g)",
                "Flaxseed Oil (1 tbsp)"
            ]
        },
        {
            "title": "Bone & Tissue Mineralization (Calcium & Vit D)",
            "target": "1000mg Calcium + 2000 IU Vit D3 daily",
            "duration_days": duration_days,
            "example_foods": [
                "Fortified Almond/Dairy Milk",
                "Cottage Cheese",
                "Dark Leafy Greens (Spinach/Kale)",
                "Sardines with bones"
            ]
        }
    ]

    # 8. Exercise Extraction based on body part
    exercises: List[Dict[str, Any]] = []
    if body_part == "Knee":
        exercises = [
            {
                "name": "Quad Setting (Isometric Terminal Extension)",
                "target_reps": "3 sets of 10 reps (5s hold)",
                "frequency": "Twice daily",
                "teach_copy": "Sit upright with affected leg extended straight. Contract your thigh muscle to press the back of your knee flat into the towel/floor. Hold firm contract for 5 seconds.",
                "caution": "Do not jerk or force joint hyper-extension; maintain steady breathing throughout hold.",
                "form_cue": "Focus on tightening the vastus medialis (inner quad tear-drop muscle) right above the kneecap.",
                "duration_days": min(14, duration_days)
            },
            {
                "name": "Straight Leg Raise (SLR)",
                "target_reps": "3 sets of 12 reps",
                "frequency": "Once daily",
                "teach_copy": "Lie on back with opposite knee bent. Lock affected knee completely straight, flex foot upward, and slowly lift leg to height of opposite knee. Lower with control.",
                "caution": "Stop if front knee lag occurs (knee bending mid-lift). Keep pelvis glued to mat.",
                "form_cue": "Lock knee straight before lifting off floor. Flex foot toward tibia.",
                "duration_days": duration_days
            },
            {
                "name": "Seated Heel Slides",
                "target_reps": "2 sets of 15 reps",
                "frequency": "Twice daily",
                "teach_copy": "Sit on a firm chair with smooth floor. Gently glide heel backward toward seat using hamstring pulling action until mild stretch is felt. Hold 3s, slide forward.",
                "caution": "Stay within pain-free Range of Motion (ROM < 3/10 discomfort).",
                "form_cue": "Keep heel flat on the slide towel; control the return stroke smoothly.",
                "duration_days": min(21, duration_days)
            },
            {
                "name": "Terminal Knee Extension (TKE) with Band",
                "target_reps": "3 sets of 15 reps",
                "frequency": "Once daily",
                "teach_copy": "Loop resistance band behind knee, attach to anchor in front. Start with knee slightly bent (20 degrees), extend leg straight against band resistance, squeezing quad.",
                "caution": "Do not let band snap knee back violently on return phase.",
                "form_cue": "Drive heel into floor and lock quad fully at top of rep.",
                "duration_days": duration_days
            }
        ]
    elif body_part == "Ankle":
        exercises = [
            {
                "name": "Ankle Alphabet Tracing",
                "target_reps": "2 full alphabets (A-Z)",
                "frequency": "Twice daily",
                "teach_copy": "Elevate leg slightly. Move only at the ankle joint to trace capital letters A through Z in the air with your big toe.",
                "caution": "Do not rotate whole hip or leg; isolate motion to the ankle socket.",
                "form_cue": "Draw full, exaggerated curves for max pain-free range of motion.",
                "duration_days": min(14, duration_days)
            },
            {
                "name": "Resisted Ankle Eversion & Inversion",
                "target_reps": "3 sets of 12 reps each way",
                "frequency": "Once daily",
                "teach_copy": "Wrap loop band around foot. Turn ankle outward against band, hold 1 sec, return slowly.",
                "caution": "Avoid cramping in calf or arch of foot.",
                "form_cue": "Keep lower leg stationary; pivot from ankle joint only.",
                "duration_days": duration_days
            },
            {
                "name": "Single-Leg Balance Cushion Stance",
                "target_reps": "3 sets of 30 seconds",
                "frequency": "Once daily",
                "teach_copy": "Stand on injured foot near wall or chair back for support safety. Lift non-weight-bearing leg. Hold balance with soft knee.",
                "caution": "Keep fingertips close to support surface to catch balance if needed.",
                "form_cue": "Grip floor with big toe, pinky toe, and heel (foot tripod).",
                "duration_days": duration_days
            }
        ]
    else:
        # Default exercises for other body parts
        exercises = [
            {
                "name": "Isometric Contraction Hold",
                "target_reps": "3 sets of 10 reps (5s hold)",
                "frequency": "Twice daily",
                "teach_copy": "Contract target muscle group against stable surface without moving joint. Hold 5 seconds, relax 3 seconds.",
                "caution": "Breathe continuously; do not hold breath during contraction.",
                "form_cue": "Smooth gradual ramp up to 70% max voluntary contraction.",
                "duration_days": min(14, duration_days)
            },
            {
                "name": "Controlled Articular Rotations (CARs)",
                "target_reps": "2 sets of 10 slow circles",
                "frequency": "Once daily",
                "teach_copy": "Rotate affected joint through its complete active anatomical circle slowly and smoothly.",
                "caution": "Stay outside pinch point zones.",
                "form_cue": "Keep surrounding body segments still while joint rotates.",
                "duration_days": duration_days
            },
            {
                "name": "Eccentric Rehabilitation Load",
                "target_reps": "3 sets of 10 slow reps (4s eccentric)",
                "frequency": "Once daily",
                "teach_copy": "Perform loading phase, then lower weight/body slowly over 4 full seconds.",
                "caution": "Stop if sharp pain exceeds 3/10 on NRS scale.",
                "form_cue": "Count '1-one-thousand, 2-one-thousand, 3-one-thousand, 4-one-thousand' on lowering.",
                "duration_days": duration_days
            }
        ]

    return {
        "diagnosis": diagnosis,
        "body_part": body_part,
        "phase": phase,
        "recovery_duration_days": duration_days,
        "recovery_duration_source": duration_source,
        "hydration_target_glasses": hydration_target,
        "medications": medications,
        "nutrition_items": nutrition_items,
        "exercises": exercises
    }
