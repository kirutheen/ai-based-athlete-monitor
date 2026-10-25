"""
Coach-facing Streamlit dashboard -- deliberately a much thinner view than
doctor_view.py.

Design intent (from the earlier doctor-vs-coach differentiation
discussion in this project): a coach needs to know "who needs to see a
doctor" and roster-level session completion, not the clinical detail
(raw FPPA degrees, IQR deviations, biomechanics citations) that only a
clinician is positioned to interpret. Showing a coach the same raw
numbers a doctor sees risks them over- or under-reacting to a number
they don't have the training to contextualize -- the whole point of the
project's doctor-first framing. This view intentionally hides band
magnitude detail and clinical citations; it shows status only.

Run with:
    streamlit run src/dashboard/coach_view.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("DRIFTGUARD_API_URL", "http://localhost:8000")

st.set_page_config(page_title="DriftGuard -- Team Overview", layout="wide")


def _get(path: str, **params):
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


st.title("DriftGuard -- Team Overview")
st.caption(
    "Session completion and clinical-review status for your roster. Clinical detail is "
    "visible only to the care team -- if an athlete is flagged, the right next step is "
    "getting them in front of a doctor, not interpreting the number yourself."
)

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    api_up = health.status_code == 200
except requests.exceptions.RequestException:
    api_up = False

if not api_up:
    st.error(f"Cannot reach the DriftGuard API at {API_URL}.")
    st.stop()

athletes = _get("/athletes")
if not athletes:
    st.info("No athletes registered yet.")
    st.stop()

rows = []
for athlete in athletes:
    aid = athlete["athlete_id"]
    events = _get(f"/athletes/{aid}/events")
    flags = _get(f"/athletes/{aid}/flags")
    unreviewed = [f for f in flags if f["reviewed_by_clinician_id"] is None]
    has_active_flag = len(unreviewed) > 0
    rows.append(
        {
            "Athlete": athlete["display_name"],
            "Team / School": athlete.get("team_or_school") or "--",
            "Sessions recorded": len(set(e["session_id"] for e in events)),
            "Status": "See a doctor" if has_active_flag else "On track",
        }
    )

roster_df = pd.DataFrame(rows)


def _highlight_status(val: str) -> str:
    return "background-color: #ffcccc" if val == "See a doctor" else "background-color: #ccffcc"


st.dataframe(roster_df.style.map(_highlight_status, subset=["Status"]), hide_index=True, use_container_width=True)

n_needs_doctor = sum(1 for r in rows if r["Status"] == "See a doctor")
if n_needs_doctor:
    st.warning(f"{n_needs_doctor} athlete(s) have a confirmed flag awaiting clinician review.")
else:
    st.success("No athletes currently flagged for clinician follow-up.")
