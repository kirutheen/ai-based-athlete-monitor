"""
Clinician-facing Streamlit dashboard.

Run with:
    streamlit run src/dashboard/doctor_view.py

Talks to the FastAPI backend (src/api/main.py) over HTTP -- set
DRIFTGUARD_API_URL if it's not running on the default
http://localhost:8000. This file contains no modeling or scoring logic
itself; it only renders what the API returns, which keeps the clinical
logic in one auditable place (src/baseline, src/explain) rather than
duplicated into the UI layer.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("DRIFTGUARD_API_URL", "http://localhost:8000")

st.set_page_config(page_title="DriftGuard -- Clinician Dashboard", layout="wide")


def _get(path: str, **params):
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, json_body: dict):
    resp = requests.post(f"{API_URL}{path}", json=json_body, timeout=10)
    resp.raise_for_status()
    return resp.json()


st.title("DriftGuard -- Clinician Dashboard")
st.caption(
    "Camera-based knee-injury-risk screening. Every flag below is scored against the "
    "athlete's OWN baseline, not a population norm, and requires persistence across "
    "multiple sessions before being shown here -- a single unusual event never appears."
)

try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    api_up = health.status_code == 200
except requests.exceptions.RequestException:
    api_up = False

if not api_up:
    st.error(
        f"Cannot reach the DriftGuard API at {API_URL}. Start it with "
        f"`uvicorn src.api.main:app --reload` and refresh this page."
    )
    st.stop()

tab_queue, tab_athlete = st.tabs(["Flag Queue", "Athlete Detail"])

with tab_queue:
    st.subheader("Unreviewed risk flags")
    show_all = st.checkbox("Show reviewed flags too", value=False)
    flags = _get("/flags", unreviewed_only=not show_all)

    if not flags:
        st.info("No flags to review right now.")
    else:
        for flag in flags:
            band_color = {"Moderate": "orange", "High": "red"}.get(flag["band"], "gray")
            with st.container(border=True):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**:{band_color}[{flag['band']}]** -- {flag['headline']}")
                    st.write(flag["detail"])
                    if flag.get("citation"):
                        st.caption(f"Reference: {flag['citation']}")
                    st.caption(f"Athlete: {flag['athlete_id']} | Feature: {flag['feature_name']}")
                with col2:
                    if flag["reviewed_by_clinician_id"] is None:
                        if st.button("Mark reviewed", key=f"review_{flag['flag_id']}"):
                            _post(
                                f"/flags/{flag['flag_id']}/review",
                                {"clinician_id": "demo_clinician", "note": None},
                            )
                            st.rerun()
                    else:
                        st.caption(f"Reviewed by {flag['reviewed_by_clinician_id']}")

with tab_athlete:
    st.subheader("Athlete trend")
    athletes = _get("/athletes")
    if not athletes:
        st.info("No athletes registered yet.")
    else:
        names = {a["athlete_id"]: a["display_name"] for a in athletes}
        selected_id = st.selectbox(
            "Select athlete", options=list(names.keys()), format_func=lambda aid: names[aid]
        )

        events = _get(f"/athletes/{selected_id}/events")
        baselines = _get(f"/athletes/{selected_id}/baselines")

        if not events:
            st.info("No recorded landing events for this athlete yet.")
        else:
            df = pd.DataFrame(events).sort_values("landing_timestamp_s")

            st.markdown("**Personal baseline status**")
            baseline_df = pd.DataFrame(baselines)
            if not baseline_df.empty:
                st.dataframe(
                    baseline_df[["feature_name", "n_events", "established", "median_value", "iqr_value"]],
                    hide_index=True,
                )

            st.markdown("**Event-level trend (FPPA, degrees)**")
            chart_df = df[["landing_timestamp_s", "fppa_left", "fppa_right"]].set_index("landing_timestamp_s")
            st.line_chart(chart_df)

            st.markdown("**Event-level trend (knee flexion, degrees)**")
            chart_df2 = df[["landing_timestamp_s", "knee_flexion_left", "knee_flexion_right"]].set_index(
                "landing_timestamp_s"
            )
            st.line_chart(chart_df2)

            athlete_flags = _get(f"/athletes/{selected_id}/flags")
            st.markdown(f"**Flag history ({len(athlete_flags)})**")
            if athlete_flags:
                st.dataframe(
                    pd.DataFrame(athlete_flags)[["band", "feature_name", "headline", "reviewed_by_clinician_id"]],
                    hide_index=True,
                )
