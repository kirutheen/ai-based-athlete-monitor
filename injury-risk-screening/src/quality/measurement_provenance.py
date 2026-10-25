"""
Measurement Provenance Engine for DriftGuard.

Provides explicit tracking of the origin, measurement status, algorithm version,
and confidence of every biomechanical metric extracted in DriftGuard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MeasurementSource(str, Enum):
    DIRECTLY_MEASURED = "DIRECTLY_MEASURED"
    POSE_DERIVED = "POSE_DERIVED"
    ESTIMATED = "ESTIMATED"
    SIMULATED = "SIMULATED"
    HARDWARE_MEASURED = "HARDWARE_MEASURED"
    REFERENCE_MEASURED = "REFERENCE_MEASURED"
    CLINICIAN_LABELED = "CLINICIAN_LABELED"
    VALIDATED = "VALIDATED"
    NOT_CURRENTLY_MEASURED = "NOT_CURRENTLY_MEASURED"


class MeasurementStatus(str, Enum):
    ESTIMATED_PROXY = "ESTIMATED_PROXY"
    VALIDATED_METRIC = "VALIDATED_METRIC"
    SIMULATED_METRIC = "SIMULATED_METRIC"
    HARDWARE_PENDING = "HARDWARE_PENDING"
    UNAVAILABLE = "UNAVAILABLE"


class ValidationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    PARTIALLY_VALIDATED = "PARTIALLY_VALIDATED"
    NOT_VALIDATED = "NOT_VALIDATED"
    SIMULATED_ONLY = "SIMULATED_ONLY"


class LabelType(str, Enum):
    RULE_BASED_LABEL = "RULE_BASED_LABEL"
    ML_PREDICTION = "ML_PREDICTION"
    ANOMALY_FLAG = "ANOMALY_FLAG"
    CLINICIAN_LABEL = "CLINICIAN_LABEL"
    REFERENCE_LABEL = "REFERENCE_LABEL"
    VALIDATED_CLINICAL_OUTCOME = "VALIDATED_CLINICAL_OUTCOME"


@dataclass
class MeasurementProvenance:
    """
    Metadata object tracking the provenance of a specific measurement.
    Answers: Where did this number come from, how was it computed, and how reliable is it?
    """
    feature_name: str
    value: float
    unit: str
    side: str = "Bilateral"  # "Left", "Right", "Bilateral", "Trunk", "Pelvis"
    measurement_source: MeasurementSource = MeasurementSource.POSE_DERIVED
    measurement_method: str = "MediaPipe BlazePose 3D Kinematic Model"
    algorithm_version: str = "v2.1.0"
    camera_configuration: str = "Monocular RGB"
    model_version: str = "pose_landmarker_full.task"
    quality_score: float = 1.0  # 0.0 to 1.0
    confidence: float = 0.85     # 0.0 to 1.0
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    is_simulated: bool = False
    is_reference: bool = False
    timestamp: Optional[float] = None
    session_id: Optional[str] = None
    trial_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "value": self.value,
            "unit": self.unit,
            "side": self.side,
            "measurement_source": self.measurement_source.value if isinstance(self.measurement_source, Enum) else self.measurement_source,
            "measurement_method": self.measurement_method,
            "algorithm_version": self.algorithm_version,
            "camera_configuration": self.camera_configuration,
            "model_version": self.model_version,
            "quality_score": round(self.quality_score, 3),
            "confidence": round(self.confidence, 3),
            "validation_status": self.validation_status.value if isinstance(self.validation_status, Enum) else self.validation_status,
            "is_simulated": self.is_simulated,
            "is_reference": self.is_reference,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "trial_id": self.trial_id,
        }
