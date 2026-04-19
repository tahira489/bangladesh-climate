"""
Runs once, saves data to climate.db, exits.
Called every hour by GitHub Actions.
"""
import requests
import sqlite3
import os
from datetime import datetime

TOKEN = os.getenv("AQICN_TOKEN", "")

CITIES = {
    "Dhaka":      "@7218",
    "Chittagong": "@9058",
    "Sylhet":     "@9061",
    "Barisal":    "@9062",
    "Rangpur":    "@9063",
}
COORDS = {
    "Dhaka":      (23.8103, 90.4125),
    "Chittagong": (22.3569, 91.7832),
    "Sylhet":     (24.8949, 91.8687),
    "Barisal":    (22.7010, 90.3535),
    "Rangpur":    (25.7439, 89.2752),
}

conn = sqlite3.connect("climate.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        city TEXT, timestamp TEXT,
        aqi REAL, pm25 REAL, temp REAL,
        humidity REAL, rainfall REAL, wind REAL,
        flood_risk TEXT
    )
""")
conn.commit()

print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC] Collecting...")

for city, cid in CITIES.items():
    try:
        r = requests.get(
            f"https://api.waqi.info/feed/{cid}/?token={TOKEN}",
            timeout=10
        ).json()
        if r["status"] != "ok":
            print(f"  {city}: AQICN error — {r.get('data','unknown')}")
            continue

        d    = r["data"]
        aqi  = float(d.get("aqi") or 0)
        pm25 = float((d.get("iaqi") or {}).get("pm25", {}).get("v") or 0)
        temp = float((d.get("iaqi") or {}).get("t",    {}).get("v") or 0)
        wind = float((d.get("iaqi") or {}).get("w",    {}).get("v") or 0)

        lat, lon = COORDS[city]
        wr = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=precipitation,relativehumidity_2m&forecast_days=1",
            timeout=10
        ).json()
        rain = round(sum(wr["hourly"]["precipitation"][:24]), 1)
        hum  = int(wr["hourly"]["relativehumidity_2m"][12])

        score = 0
        if rain > 80:   score += 3
        elif rain > 40: score += 2
        elif rain > 20: score += 1
        if aqi > 150:   score += 1
        if hum > 85:    score += 1
        risk = ("Critical" if score >= 4 else "High" if score >= 3
                else "Moderate" if score >= 2 else "Low")

        conn.execute(
            "INSERT INTO readings "
            "(city,timestamp,aqi,pm25,temp,humidity,rainfall,wind,flood_risk) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (city, datetime.utcnow().isoformat(), aqi, pm25, temp, hum, rain, wind, risk)
        )
        print(f"  {city}: AQI={aqi} Rain={rain}mm Risk={risk}")

    except Exception as e:
        print(f"  {city}: ERROR — {e}")

conn.commit()
conn.close()
print("Done.")
