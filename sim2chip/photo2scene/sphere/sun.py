"""Where the sun was when the sphere was taken, from its time, rough position and compass heading (capture.py's
sphere.json), instead of estimating it from shadows (docs/conocimiento_photosphere.md, light). Solar position by the
NOAA formulas (Spencer's series for the declination and the equation of time), good to a fraction of a degree,
which is far finer than a phone's compass. Pure Python.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

from . import capture

MIN_ELEVATION_DEG = 2.0  # under this the sun is down or behind the horizon's haze: sky light only


def solar_position(when: datetime, lat: float, lon: float) -> tuple[float, float]:
    """Elevation and azimuth (degrees; azimuth clockwise from north) of the sun at an aware datetime."""
    t = when.astimezone(timezone.utc)
    day = t.timetuple().tm_yday
    hours = t.hour + t.minute / 60 + t.second / 3600
    g = 2 * math.pi / 365 * (day - 1 + (hours - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                       - 0.040849 * math.sin(2 * g))
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    true_solar_min = hours * 60 + eqtime + 4 * lon
    ha = math.radians(true_solar_min / 4 - 180)
    phi = math.radians(lat)
    cos_zen = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cos_zen)))
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(phi) - math.tan(decl) * math.cos(phi))) + 180
    return 90 - math.degrees(zen), az % 360


def sun_for(info: dict) -> dict | None:
    """The sun of a sphere (sphere.json) in its levelled frame: elevation, azimuth, its heading in the panorama
    (degrees to the right of the panorama's centre) and the direction its light travels (unit, levelled frame).
    None when time, position or heading is missing; "up": False when it is under MIN_ELEVATION_DEG."""
    if not all(k in info for k in ("taken_at", "lat", "lon", "heading_deg")):
        return None
    when = datetime.fromisoformat(info["taken_at"])
    if when.tzinfo is None:
        return None
    elev, az = solar_position(when, info["lat"], info["lon"])
    yaw = (az - info["heading_deg"]) % 360
    toward = capture.rotation(yaw, -elev) @ np.array([0.0, 0.0, 1.0])
    return {"elevation_deg": round(elev, 2), "azimuth_deg": round(az, 2), "yaw_deg": round(yaw, 2),
            "travel_level": (-toward).round(5).tolist(), "up": elev >= MIN_ELEVATION_DEG}
