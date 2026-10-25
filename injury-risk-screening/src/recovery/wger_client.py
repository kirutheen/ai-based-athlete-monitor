"""
Wger.de Exercise API Media Proxy with Caching and Graceful Local Fallbacks.
"""

from __future__ import annotations
import urllib.request
import urllib.parse
import json
import time
from typing import Dict, Any, Optional

# In-memory TTL Cache: { exercise_name_lower: (timestamp, result_dict) }
_MEDIA_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour


def _generate_fallback_svg(exercise_name: str) -> str:
    """Generates an inline SVG placeholder graphic for exercises when network is offline."""
    clean_name = exercise_name.strip()
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" width="100%" height="100%">
  <rect width="400" height="300" fill="#F8F4EC" rx="16"/>
  <circle cx="200" cy="110" r="45" fill="#6E8F72" opacity="0.15"/>
  <circle cx="200" cy="110" r="28" fill="#6E8F72" opacity="0.3"/>
  <path d="M 180 110 L 220 110 M 200 90 L 200 130" stroke="#6E8F72" stroke-width="4" stroke-linecap="round"/>
  <rect x="60" y="180" width="280" height="6" fill="#D9A860" rx="3"/>
  <text x="200" y="215" text-anchor="middle" fill="#2C3531" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="16" font-weight="600">{clean_name}</text>
  <text x="200" y="240" text-anchor="middle" fill="#6E8F72" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" font-size="13">Exercise Visual Demonstration</text>
</svg>'''
    return svg


def fetch_exercise_media(exercise_name: str) -> Dict[str, Any]:
    """
    Fetches exercise media from Wger API or returns cached/fallback media.
    Guarantees non-null response with image_url or fallback_svg.
    """
    key = exercise_name.strip().lower()
    now = time.time()

    # Check cache first
    if key in _MEDIA_CACHE:
        cached_time, cached_val = _MEDIA_CACHE[key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_val

    fallback_svg = _generate_fallback_svg(exercise_name)
    fallback_res = {
        "exercise_name": exercise_name,
        "image_url": None,
        "fallback_svg": fallback_svg,
        "source": "fallback_svg"
    }

    try:
        # Search wger for exercise
        query = urllib.parse.quote(exercise_name.split("(")[0].strip())
        search_url = f"https://wger.de/api/v2/exercise/search/?term={query}"
        
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "DriftGuard-AthleteRecovery/1.0", "Accept": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode('utf-8'))
                suggestions = data.get("suggestions", [])
                if suggestions:
                    first_item = suggestions[0]
                    # Check images
                    ex_id = first_item.get("data", {}).get("id")
                    if ex_id:
                        img_url = f"https://wger.de/api/v2/exerciseimage/?exercise={ex_id}"
                        img_req = urllib.request.Request(img_url, headers={"User-Agent": "DriftGuard-AthleteRecovery/1.0"})
                        with urllib.request.urlopen(img_req, timeout=3) as img_resp:
                            if img_resp.status == 200:
                                img_data = json.loads(img_resp.read().decode('utf-8'))
                                results = img_data.get("results", [])
                                if results and len(results) > 0:
                                    img_link = results[0].get("image")
                                    if img_link:
                                        res = {
                                            "exercise_name": exercise_name,
                                            "image_url": img_link,
                                            "fallback_svg": fallback_svg,
                                            "source": "wger"
                                        }
                                        _MEDIA_CACHE[key] = (now, res)
                                        return res
    except Exception:
        pass  # Graceful fallback to inline SVG on offline / timeout / 404

    _MEDIA_CACHE[key] = (now, fallback_res)
    return fallback_res
