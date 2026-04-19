import streamlit as st
import sqlite3
import os
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
import requests
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bangladesh Climate Intelligence",
    page_icon="🌊",
    layout="wide",
)

# Try Streamlit secrets first (cloud), then .env (local)
try:
    AQICN_TOKEN = st.secrets["AQICN_TOKEN"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    AQICN_TOKEN = os.getenv("AQICN_TOKEN", "")
    GROQ_KEY = os.getenv("GROQ_API_KEY", "")

DB_PATH = "climate.db"

CITIES = ["Dhaka", "Chittagong", "Sylhet", "Barisal", "Rangpur"]
CITY_IDS = {
    "Dhaka": "@7218", "Chittagong": "@9058",
    "Sylhet": "@9061", "Barisal": "@9062", "Rangpur": "@9063",
}
COORDS = {
    "Dhaka":      (23.8103, 90.4125),
    "Chittagong": (22.3569, 91.7832),
    "Sylhet":     (24.8949, 91.8687),
    "Barisal":    (22.7010, 90.3535),
    "Rangpur":    (25.7439, 89.2752),
}
RISK_COLOR = {"Critical": "#A32D2D", "High": "#854F0B", "Moderate": "#3B6D11", "Low": "#0C447C"}
RISK_EMOJI = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}

st.markdown("""
<style>
[data-testid="stMetric"] {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 16px 20px;
}
</style>
""", unsafe_allow_html=True)


# ── DATABASE ──────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()


def query(sql, params=()):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def get_latest():
    return query("""
        SELECT * FROM readings
        WHERE timestamp = (
            SELECT MAX(r2.timestamp) FROM readings r2 WHERE r2.city = readings.city
        )
        ORDER BY city
    """)


def get_history(city):
    return query(
        "SELECT * FROM readings WHERE city=? ORDER BY timestamp DESC LIMIT 168",
        (city,)
    )


def get_stats():
    return query("""
        SELECT city,
            ROUND(AVG(aqi),1) avg_aqi, ROUND(MAX(aqi),1) max_aqi,
            ROUND(MIN(aqi),1) min_aqi, ROUND(AVG(rainfall),1) avg_rain,
            ROUND(MAX(rainfall),1) max_rain, COUNT(*) readings
        FROM readings GROUP BY city ORDER BY avg_aqi DESC
    """)


def total_readings():
    df = query("SELECT COUNT(*) as n FROM readings")
    return int(df["n"].iloc[0]) if not df.empty else 0


# ── LIVE DATA FETCH ────────────────────────────────────────────────────────────
def fetch_and_save():
    """Fetch live data from APIs and save to DB. Called when user clicks Refresh."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    saved = 0
    for city, cid in CITY_IDS.items():
        try:
            # AQI
            r = requests.get(
                f"https://api.waqi.info/feed/{cid}/?token={AQICN_TOKEN}",
                timeout=10
            ).json()
            if r["status"] != "ok":
                continue
            d = r["data"]
            aqi  = float(d.get("aqi") or 0)
            pm25 = float((d.get("iaqi") or {}).get("pm25", {}).get("v") or 0)
            temp = float((d.get("iaqi") or {}).get("t",    {}).get("v") or 0)
            wind = float((d.get("iaqi") or {}).get("w",    {}).get("v") or 0)

            # Weather
            lat, lon = COORDS[city]
            wr = requests.get(
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&hourly=precipitation,relativehumidity_2m&forecast_days=1",
                timeout=10
            ).json()
            rain = round(sum(wr["hourly"]["precipitation"][:24]), 1)
            hum  = int(wr["hourly"]["relativehumidity_2m"][12])

            # Flood risk score
            score = 0
            if rain > 80:    score += 3
            elif rain > 40:  score += 2
            elif rain > 20:  score += 1
            if aqi > 150:    score += 1
            if hum > 85:     score += 1
            risk = ("Critical" if score >= 4 else "High" if score >= 3
                    else "Moderate" if score >= 2 else "Low")

            conn.execute(
                "INSERT INTO readings (city,timestamp,aqi,pm25,temp,humidity,rainfall,wind,flood_risk) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (city, datetime.utcnow().isoformat(), aqi, pm25, temp, hum, rain, wind, risk)
            )
            saved += 1
        except Exception as e:
            st.warning(f"Could not fetch {city}: {e}")

    conn.commit()
    conn.close()
    return saved


# ── PREDICTIONS (rule-based, no scikit-learn needed) ──────────────────────────
def predict_aqi(aqi, humidity, wind):
    # Simple heuristic: pollution drops with wind, rises with humidity
    factor = 1.0
    if wind > 20:     factor -= 0.12
    elif wind > 10:   factor -= 0.06
    if humidity > 85: factor += 0.05
    return int(aqi * factor * 0.93)


def predict_flood(rainfall, aqi, humidity):
    score = 0
    if rainfall > 80:   score += 3
    elif rainfall > 40: score += 2
    elif rainfall > 20: score += 1
    if aqi > 150:       score += 1
    if humidity > 85:   score += 1
    if score >= 4:  return "Critical", 91
    if score >= 3:  return "High",     78
    if score >= 2:  return "Moderate", 82
    return "Low", 88


def combined_risk(aqi, flood_risk):
    if aqi > 150 and flood_risk in ("High", "Critical"): return "Critical"
    if aqi > 100 and flood_risk in ("High", "Critical"): return "High"
    if aqi > 150: return "High"
    return flood_risk


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("## 🌊 Bangladesh Climate Intelligence")
st.markdown("Real-time AQI · Flood risk · AI environmental predictions")

n = total_readings()
col_a, col_b, col_c = st.columns([2, 1, 1])
col_a.caption(f"Database: **{n} readings** collected")

if col_b.button("🔄 Fetch live data now"):
    with st.spinner("Fetching from AQICN and Open-Meteo..."):
        saved = fetch_and_save()
    if saved > 0:
        st.success(f"Saved {saved} city readings!")
        st.rerun()
    else:
        st.error("No data saved. Check your AQICN token in Secrets.")

if n == 0:
    st.warning("No data yet — click **Fetch live data now** above to load your first readings.")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 AI Predictions", "💬 AI Agent"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    city = st.selectbox("Select city", CITIES, key="dash_city")
    latest_df = get_latest()

    if latest_df.empty:
        st.info("Click **Fetch live data now** at the top to load data.")
    else:
        row = latest_df[latest_df["city"] == city]
        if row.empty:
            st.warning(f"No data for {city} yet.")
        else:
            r = row.iloc[0]
            ts = str(r["timestamp"])[:16].replace("T", "  ")
            st.caption(f"Last updated: {ts} UTC")

            aqi = float(r["aqi"])
            aqi_label = (
                "Good" if aqi <= 50 else "Moderate" if aqi <= 100 else
                "Unhealthy for groups" if aqi <= 150 else "Unhealthy" if aqi <= 200 else
                "Very unhealthy" if aqi <= 300 else "Hazardous"
            )
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("AQI",          int(aqi),                    aqi_label)
            c2.metric("PM2.5 µg/m³",  int(r["pm25"]),              "WHO limit: 15")
            c3.metric("Rainfall",      f"{r['rainfall']:.1f} mm")
            c4.metric("Flood risk",    r["flood_risk"])
            c5.metric("Temperature",   f"{r['temp']:.0f}°C")
            c6.metric("Humidity",      f"{r['humidity']:.0f}%")

    st.divider()

    # History chart
    hist_df = get_history(city)
    if not hist_df.empty:
        hist_df = hist_df.iloc[::-1].reset_index(drop=True)
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["aqi"],
            name="AQI", line=dict(color="#E24B4A", width=2),
            fill="tozeroy", fillcolor="rgba(226,75,74,0.08)"
        ))
        fig.add_trace(go.Scatter(
            x=hist_df["timestamp"], y=hist_df["rainfall"],
            name="Rainfall mm", line=dict(color="#378ADD", width=2),
            yaxis="y2", fill="tozeroy", fillcolor="rgba(55,138,221,0.08)"
        ))
        fig.update_layout(
            title=f"{city} — AQI & Rainfall history",
            height=300,
            margin=dict(t=40,b=20,l=0,r=0),
            yaxis=dict(title="AQI"),
            yaxis2=dict(title="Rainfall mm", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("History chart appears after multiple data fetches.")

    st.divider()
    st.subheader("All districts — combined risk")
    if not latest_df.empty:
        cols = st.columns(len(latest_df))
        for i, (_, drow) in enumerate(latest_df.iterrows()):
            cr = combined_risk(float(drow["aqi"]), drow["flood_risk"])
            with cols[i]:
                st.markdown(f"**{drow['city']}**")
                st.markdown(
                    f"<span style='color:{RISK_COLOR.get(cr,'#333')};font-weight:600'>"
                    f"{RISK_EMOJI.get(cr,'')} {cr}</span>",
                    unsafe_allow_html=True
                )
                st.caption(f"AQI {int(drow['aqi'])} · Rain {drow['rainfall']:.0f}mm")

    st.divider()
    st.subheader("Historical statistics")
    stats_df = get_stats()
    if not stats_df.empty:
        stats_df.columns = ["City","Avg AQI","Max AQI","Min AQI","Avg Rain mm","Max Rain mm","Readings"]
        st.dataframe(stats_df, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Environmental predictions")
    st.caption("Rule-based predictions using current environmental data — no ML library needed")

    pred_city = st.selectbox("City", CITIES, key="pred_city")
    latest_df2 = get_latest()

    if latest_df2.empty:
        st.info("Fetch data first using the button at the top.")
    else:
        row2 = latest_df2[latest_df2["city"] == pred_city]
        if not row2.empty:
            r2 = row2.iloc[0]
            aqi2  = float(r2["aqi"])
            rain2 = float(r2["rainfall"])
            hum2  = float(r2["humidity"])
            temp2 = float(r2["temp"])
            wind2 = float(r2["wind"])

            pred_aqi = predict_aqi(aqi2, hum2, wind2)
            pred_fl, conf = predict_flood(rain2, aqi2, hum2)

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted AQI (next hour)", pred_aqi)
            c2.metric("Predicted flood risk",       pred_fl)
            c3.metric("Confidence",                 f"{conf}%")

            st.divider()
            st.subheader("72-hour forecast")
            forecast = [
                {"label": "Today",     "aqi": int(aqi2),           "rain": rain2},
                {"label": "+24 hours", "aqi": int(aqi2 * 0.94),    "rain": round(rain2 * 0.85, 1)},
                {"label": "+48 hours", "aqi": int(aqi2 * 0.89),    "rain": round(rain2 * 0.70, 1)},
            ]
            fc_cols = st.columns(3)
            for i, fc in enumerate(forecast):
                with fc_cols[i]:
                    color = "#A32D2D" if fc["aqi"] > 150 else "#854F0B" if fc["aqi"] > 100 else "#3B6D11"
                    st.markdown(f"**{fc['label']}**")
                    st.markdown(
                        f"<span style='font-size:28px;font-weight:600;color:{color}'>"
                        f"{fc['aqi']}</span> AQI",
                        unsafe_allow_html=True
                    )
                    st.caption(f"Rain: {fc['rain']} mm")

            st.divider()
            st.subheader("Other environmental indicators")
            heat_idx    = round(temp2 + 0.33 * (hum2 / 100 * 6.105) - 4.0, 1)
            disease     = ("High"     if r2["flood_risk"] == "Critical" and rain2 > 60
                           else "Moderate" if rain2 > 40 else "Low")
            crop_risk   = min(int((rain2 / 150 * 60) + (aqi2 / 300 * 40)), 100)

            c1, c2, c3 = st.columns(3)
            c1.metric("Heat index",              f"{heat_idx}°C")
            c2.metric("Waterborne disease risk",  disease)
            c3.metric("Crop damage risk",         f"{crop_risk}%")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI AGENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Climate AI Agent")
    st.caption("Powered by Google Gemini (free) · Knows your live data")

    if not GROQ_KEY:
        st.error("GROQ_API_KEY not set. Add it to Streamlit Secrets.")
        st.stop()

    groq_client = Groq(api_key=GROQ_KEY)

    # Build live context
    live_df = get_latest()
    if not live_df.empty:
        lines = ["Current live Bangladesh environmental data:"]
        for _, dr in live_df.iterrows():
            lines.append(
                f"- {dr['city']}: AQI {int(dr['aqi'])}, "
                f"PM2.5 {int(dr['pm25'])}µg/m³, "
                f"Rainfall {dr['rainfall']:.1f}mm, "
                f"Flood risk {dr['flood_risk']}, "
                f"Temp {dr['temp']:.0f}°C, "
                f"Humidity {dr['humidity']:.0f}%"
            )
        live_ctx = "\n".join(lines)
    else:
        live_ctx = "No live data yet. Use your expert Bangladesh climate knowledge."

    SYSTEM = f"""You are an expert AI climate scientist specialising in Bangladesh.

{live_ctx}

You predict and explain: AQI trends, flood probabilities, waterborne disease risk, crop damage, heat stress, policy interventions.
Give specific numbers, timeframes, confidence levels. Reference the live data above.
Keep responses under 200 words. Write in clear prose."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("**Quick questions:**")
    q1, q2, q3, q4 = st.columns(4)
    if q1.button("Predict Dhaka AQI tomorrow"):
        st.session_state.pending = "Predict Dhaka's AQI for tomorrow based on current data. Give a specific number."
    if q2.button("Worst risk districts?"):
        st.session_state.pending = "Which 3 districts have the worst combined air and flood risk right now?"
    if q3.button("Flood interventions?"):
        st.session_state.pending = "What are the 3 best interventions to reduce flood risk in Bangladesh?"
    if q4.button("20-year outlook?"):
        st.session_state.pending = "How will climate change affect Bangladesh floods and AQI over 20 years?"

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask anything about Bangladesh climate, AQI, floods...")
    if not prompt and "pending" in st.session_state:
        prompt = st.session_state.pop("pending")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    chat_messages = [{"role": "system", "content": SYSTEM}]
                    for m in st.session_state.messages:
                        chat_messages.append({"role": m["role"], "content": m["content"]})
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=chat_messages,
                        max_tokens=400,
                    )
                    reply = response.choices[0].message.content
                    st.write(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Gemini error: {e}")

    if st.session_state.messages:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
