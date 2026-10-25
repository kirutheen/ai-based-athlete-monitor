"""
Clinical PDF Report Generator for DriftGuard.
Generates professional, printable PDF clinical screening summaries.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_clinical_pdf_report(
    session_data: Dict[str, Any],
    athlete_data: Dict[str, Any],
    flags_data: List[Dict[str, Any]],
    notes_data: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Generates a clinical PDF screening report at output_path and returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()

    # Custom Color Palette
    primary_color = colors.HexColor("#0F172A")    # Dark Navy
    accent_color = colors.HexColor("#2563EB")     # Blue Accent
    success_color = colors.HexColor("#166534")    # Dark Green
    warning_color = colors.HexColor("#B45309")    # Dark Amber
    danger_color = colors.HexColor("#991B1B")     # Dark Red
    gray_bg = colors.HexColor("#F8FAFC")
    border_color = colors.HexColor("#E2E8F0")

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerCustom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # Header Title & Disclaimer Banner
    story.append(Paragraph("<b>DriftGuard Clinical</b> | Movement Risk Screening Report", title_style))
    story.append(Paragraph("AI-Assisted Biomechanical Assessment & Baseline Comparison Platform", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceAfter=10))

    disclaimer_text = (
        "<b>NON-DIAGNOSTIC SCREENING AID:</b> This report presents computer-vision movement screening proxies "
        "and baseline comparisons. It is intended solely for clinical decision support by licensed doctors "
        "and physiotherapists, and does NOT constitute a formal medical diagnosis or ACL injury confirmation."
    )
    disc_table = Table(
        [[Paragraph(disclaimer_text, disclaimer_style)]],
        colWidths=[540],
    )
    disc_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
            ("BORDER", (0, 0), (-1, -1), 0.5, colors.HexColor("#FCA5A5")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(disc_table)
    story.append(Spacer(1, 6))

    hw_text = (
        "<b>MEASUREMENT PROVENANCE & HARDWARE DISCLOSURE:</b> Camera Setup: Monocular RGB (MediaPipe 3D) | IMU Status: <i>IMU integration available — hardware not connected</i> | Kinematics: <i>Pose-Derived Proxies</i> | Mocap Validation: <i>Not Validated against Mocap Reference</i>"
    )
    hw_table = Table([[Paragraph(hw_text, disclaimer_style)]], colWidths=[540])
    hw_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
            ("BORDER", (0, 0), (-1, -1), 0.5, colors.HexColor("#86EFAC")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(hw_table)
    story.append(Spacer(1, 10))

    disc_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
            ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#BFDBFE")),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(disc_table)
    story.append(Spacer(1, 12))

    # Section 1: Patient & Session Meta
    story.append(Paragraph("1. Athlete & Session Metadata", h2_style))
    ath_name = athlete_data.get("display_name", "Unknown Athlete")
    ath_id = athlete_data.get("athlete_id", "N/A")
    sport = athlete_data.get("sport", session_data.get("sport_id", "Basketball"))
    leg = athlete_data.get("dominant_leg", "Right")

    session_id = session_data.get("session_id", "N/A")
    rec_at = session_data.get("recorded_at", datetime.now().isoformat())
    if isinstance(rec_at, str):
        rec_at = rec_at.split("T")[0]

    overall_band = session_data.get("overall_risk_band", "Low")
    band_bg = success_color if overall_band == "Low" else (warning_color if overall_band == "Moderate" else danger_color)

    meta_grid = [
        [
            Paragraph(f"<b>Athlete Name:</b> {ath_name}", body_style),
            Paragraph(f"<b>Session ID:</b> {session_id}", body_style),
        ],
        [
            Paragraph(f"<b>Athlete ID:</b> {ath_id}", body_style),
            Paragraph(f"<b>Assessment Date:</b> {rec_at}", body_style),
        ],
        [
            Paragraph(f"<b>Sport Profile:</b> {sport.capitalize()}", body_style),
            Paragraph(f"<b>Camera Setup:</b> {session_data.get('camera_setup', 'Frontal')}", body_style),
        ],
        [
            Paragraph(f"<b>Dominant Leg:</b> {leg.capitalize()}", body_style),
            Paragraph(f"<b>Overall Risk Band:</b> <font color='{band_bg.hexval()}'><b>{overall_band.upper()}</b></font>", body_style),
        ],
    ]
    meta_table = Table(meta_grid, colWidths=[270, 270])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), gray_bg),
            ("BORDER", (0, 0), (-1, -1), 0.5, border_color),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Section 2: Key Biomechanical Metrics Summary
    story.append(Paragraph("2. Observed Kinematic Summary (Mean / Event Peak)", h2_style))
    metrics = session_data.get("metrics", {})
    fppa_l = metrics.get("fppa_left_deg", 0.0)
    fppa_r = metrics.get("fppa_right_deg", 0.0)
    flex_l = metrics.get("knee_flexion_left_deg", 0.0)
    flex_r = metrics.get("knee_flexion_right_deg", 0.0)
    valg_lsi = metrics.get("valgus_asymmetry_lsi_pct", 0.0)
    flex_lsi = metrics.get("flexion_asymmetry_lsi_pct", 0.0)
    t_lean = metrics.get("trunk_lean_deg", 0.0)

    metrics_table_data = [
        [Paragraph("<b>Biomechanical Metric</b>", body_style), Paragraph("<b>Left Limb</b>", body_style), Paragraph("<b>Right Limb</b>", body_style), Paragraph("<b>Limb Symmetry (LSI)</b>", body_style)],
        [Paragraph("Knee FPPA Valgus Proxy", body_style), Paragraph(f"{fppa_l:.1f}°", body_style), Paragraph(f"{fppa_r:.1f}°", body_style), Paragraph(f"{valg_lsi:.1f}%", body_style)],
        [Paragraph("Sagittal Knee Flexion", body_style), Paragraph(f"{flex_l:.1f}°", body_style), Paragraph(f"{flex_r:.1f}°", body_style), Paragraph(f"{flex_lsi:.1f}%", body_style)],
        [Paragraph("Trunk Lateral Lean", body_style), Paragraph(f"{t_lean:.1f}°", body_style), Paragraph("N/A", body_style), Paragraph("N/A", body_style)],
    ]
    metrics_table = Table(metrics_table_data, colWidths=[180, 120, 120, 120])
    metrics_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("GRID", (0, 0), (-1, -1), 0.5, border_color),
            ("PADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(metrics_table)
    story.append(Spacer(1, 12))

    # Section 3: Confirmed Risk Flags & Deviations
    story.append(Paragraph("3. Confirmed Risk Flags & Literature Evidence", h2_style))
    if not flags_data:
        story.append(Paragraph("<i>No high or moderate risk flags confirmed for this screening session. Movement mechanics remain within personal baseline tolerances.</i>", body_style))
    else:
        flag_rows = [[Paragraph("<b>Feature</b>", body_style), Paragraph("<b>Risk Band</b>", body_style), Paragraph("<b>Deviation</b>", body_style), Paragraph("<b>Clinical Detail / Citation</b>", body_style)]]
        for f in flags_data:
            band = f.get("band", "Moderate")
            band_str = f"<font color='{danger_color.hexval() if band=='High' else warning_color.hexval()}'><b>{band}</b></font>"
            dev_str = f"+{f.get('deviation_iqr', 0.0):.1f} IQR" if f.get('deviation_iqr') else "N/A"
            detail = f"<b>{f.get('headline', '')}</b><br/>{f.get('detail', '')}<br/><font color='#64748B'>Citation: {f.get('citation', 'N/A')}</font>"

            flag_rows.append([
                Paragraph(f.get("feature_name", "feature"), body_style),
                Paragraph(band_str, body_style),
                Paragraph(dev_str, body_style),
                Paragraph(detail, body_style),
            ])
        flag_table = Table(flag_rows, colWidths=[110, 70, 80, 280])
        flag_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
                ("GRID", (0, 0), (-1, -1), 0.5, border_color),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        story.append(flag_table)

    story.append(Spacer(1, 14))

    # Section 4: Clinician Findings & Treatment Cues
    story.append(Paragraph("4. Clinician Observations & Neuromuscular Retraining Notes", h2_style))
    if not notes_data:
        story.append(Paragraph("<i>No clinician notes recorded for this session yet.</i>", body_style))
    else:
        for n in notes_data:
            n_type = n.get("note_type", "General").capitalize()
            c_name = n.get("clinician_name", "Dr. Clinician")
            c_text = n.get("content", "")
            note_box = [
                [Paragraph(f"<b>Note Type:</b> {n_type} | <b>Author:</b> {c_name}", body_style)],
                [Paragraph(f"{c_text}", body_style)],
            ]
            nt_table = Table(note_box, colWidths=[540])
            nt_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), gray_bg),
                    ("BORDER", (0, 0), (-1, -1), 0.5, border_color),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ])
            )
            story.append(nt_table)
            story.append(Spacer(1, 6))

    story.append(Spacer(1, 20))

    # Signature Block
    sig_data = [
        [
            Paragraph("<b>Assessing Clinician Signature:</b> ___________________________", body_style),
            Paragraph("<b>Date:</b> ______________", body_style),
        ]
    ]
    sig_table = Table(sig_data, colWidths=[360, 180])
    story.append(KeepTogether([sig_table]))

    doc.build(story)
    return output_path
