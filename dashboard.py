import os
import glob
import pandas as pd
import pydeck as pdk
import streamlit as st
import altair as alt

st.set_page_config(page_title="US Accidents 2016–2023", layout="wide")

DEFAULT_DASH_BASE = r"D:\Project\big_data\data_lake\serving\dashboard\us_accidents_dashboard_csv"
DEFAULT_GRAPH_BASE = r"D:\Project\big_data\data_lake\serving\graph_csv\us_accidents_graph_csv"

# =========================================================
# STYLE (Cards)
# =========================================================
st.markdown(
    """
<style>
:root{
  --card-bg: rgba(17,24,39,.60);  /* slate-900 */
  --card-bd: rgba(148,163,184,.18); /* slate-400 */
  --card-tx: rgba(226,232,240,.92); /* slate-200 */
  --card-sub: rgba(148,163,184,.92);
  --shadow: 0 10px 30px rgba(0,0,0,.25);
}
.card {
  background: var(--card-bg);
  border: 1px solid var(--card-bd);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
}
.card .top {
  display:flex; align-items:center; justify-content:space-between; gap:10px;
  margin-bottom: 8px;
}
.card .title{
  font-size: 12px; letter-spacing:.3px; color: var(--card-sub);
  text-transform: uppercase;
}
.card .icon{
  font-size: 18px; opacity:.9;
}
.card .value{
  font-size: 26px; font-weight: 800; color: var(--card-tx);
  line-height: 1.1;
}
.card .subtitle{
  margin-top: 6px;
  font-size: 12px; color: var(--card-sub);
}
.small-note{
  color: rgba(148,163,184,.95);
  font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

def render_cards(cards, ncols=3):
    """
    cards: list of dict {title, value, subtitle, icon}
    ncols: 2/3/4
    """
    if not cards:
        return
    ncols = max(1, int(ncols))
    cols = st.columns(ncols)
    for i, c in enumerate(cards[:ncols]):
        with cols[i]:
            st.markdown(
                f"""
<div class="card">
  <div class="top">
    <div class="title">{c.get("title","")}</div>
    <div class="icon">{c.get("icon","")}</div>
  </div>
  <div class="value">{c.get("value","-")}</div>
  <div class="subtitle">{c.get("subtitle","")}</div>
</div>
""",
                unsafe_allow_html=True,
            )

# =========================================================
# HARD-CODE MAP: State name + lat/lon
# =========================================================
STATE_LATLON = [
    ("Alabama", 32.806671, -86.791130), ("Alaska", 61.370716, -152.404419),
    ("Arizona", 33.729759, -111.431221), ("Arkansas", 34.969704, -92.373123),
    ("California", 36.116203, -119.681564), ("Colorado", 39.059811, -105.311104),
    ("Connecticut", 41.597782, -72.755371), ("Delaware", 39.318523, -75.507141),
    ("Florida", 27.766279, -81.686783), ("Georgia", 33.040619, -83.643074),
    ("Hawaii", 21.094318, -157.498337), ("Idaho", 44.240459, -114.478828),
    ("Illinois", 40.349457, -88.986137), ("Indiana", 39.849426, -86.258278),
    ("Iowa", 42.011539, -93.210526), ("Kansas", 38.526600, -96.726486),
    ("Kentucky", 37.668140, -84.670067), ("Louisiana", 31.169546, -91.867805),
    ("Maine", 44.693947, -69.381927), ("Maryland", 39.063946, -76.802101),
    ("Massachusetts", 42.230171, -71.530106), ("Michigan", 43.326618, -84.536095),
    ("Minnesota", 45.694454, -93.900192), ("Mississippi", 32.741646, -89.678696),
    ("Missouri", 38.456085, -92.288368), ("Montana", 46.921925, -110.454353),
    ("Nebraska", 41.125370, -98.268082), ("Nevada", 38.313515, -117.055374),
    ("New Hampshire", 43.452492, -71.563896), ("New Jersey", 40.298904, -74.521011),
    ("New Mexico", 34.840515, -106.248482), ("New York", 42.165726, -74.948051),
    ("North Carolina", 35.630066, -79.806419), ("North Dakota", 47.528912, -99.784012),
    ("Ohio", 40.388783, -82.764915), ("Oklahoma", 35.565342, -96.928917),
    ("Oregon", 44.572021, -122.070938), ("Pennsylvania", 40.590752, -77.209755),
    ("Rhode Island", 41.680893, -71.511780), ("South Carolina", 33.856892, -80.945007),
    ("South Dakota", 44.299782, -99.438828), ("Tennessee", 35.747845, -86.692345),
    ("Texas", 31.054487, -97.563461), ("Utah", 40.150032, -111.862434),
    ("Vermont", 44.045876, -72.710686), ("Virginia", 37.769337, -78.169968),
    ("Washington", 47.400902, -121.490494), ("West Virginia", 38.491226, -80.954453),
    ("Wisconsin", 44.268543, -89.616508), ("Wyoming", 42.755966, -107.302490),
]
STATE_MAP_DF = pd.DataFrame(STATE_LATLON, columns=["StateName", "lat", "lon"])

STATE_CODE_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia"
}

# =========================
# Helpers
# =========================
def norm_path(s: str) -> str:
    if s is None:
        return ""
    return s.strip().strip('"').strip("'")


def ensure_child_folder(base: str, child: str) -> str:
    base = norm_path(base)
    cand = os.path.join(base, child)
    return cand if os.path.isdir(cand) else base


def read_spark_csv_folder(folder_path: str) -> pd.DataFrame:
    """Spark CSV output folder: part-*.csv + _SUCCESS (+ .crc)."""
    if not folder_path or not os.path.isdir(folder_path):
        return pd.DataFrame()
    files = glob.glob(os.path.join(folder_path, "part-*.csv"))
    if not files:
        files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if not f.endswith(".crc")]
    if not files:
        return pd.DataFrame()
    files = sorted(files)
    return pd.read_csv(files[0])


def safe_int_series(s):
    try:
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    except:
        return s


def safe_float_series(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except:
        return s


def pick_col(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None
    m = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in m:
            return m[c.lower()]
    return None


def to_numeric(df: pd.DataFrame, col: str):
    if col and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def drop_if_exists(df: pd.DataFrame, cols):
    if df is None or df.empty:
        return df
    exist = [c for c in cols if c in df.columns]
    return df.drop(columns=exist) if exist else df


def severity_label_en(x):
    try:
        xi = int(float(x))
    except:
        return str(x)
    mapping = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
    return mapping.get(xi, f"Level {xi}")


def clean_state_to_name(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    up = s.str.upper()
    return up.map(STATE_CODE_TO_NAME).fillna(s)


def pie_vega(df_counts: pd.DataFrame, label_col: str, value_col: str, title: str):
    if df_counts is None or df_counts.empty:
        st.info("No data for pie chart.")
        return

    data = df_counts[[label_col, value_col]].copy()
    data[label_col] = data[label_col].astype(str)
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)

    spec = {
        "width": "container",
        "height": 320,
        "data": {"values": data.to_dict(orient="records")},
        "mark": {"type": "arc"},
        "encoding": {
            "theta": {"field": value_col, "type": "quantitative"},
            "color": {"field": label_col, "type": "nominal", "legend": {"title": ""}},
            "tooltip": [
                {"field": label_col, "type": "nominal", "title": "Category"},
                {"field": value_col, "type": "quantitative", "title": "Cases"},
            ],
        },
        "title": {"text": title, "fontSize": 14},
    }
    st.vega_lite_chart(spec, use_container_width=True)


# =========================
# Load data once (ACTUAL + PRED)
# =========================
DASH_BASE = ensure_child_folder(DEFAULT_DASH_BASE, "us_accidents_dashboard_csv")
GRAPH_BASE = ensure_child_folder(DEFAULT_GRAPH_BASE, "us_accidents_graph_csv")

# ---- predicted (model results) ----
trend_pred = read_spark_csv_folder(os.path.join(DASH_BASE, "trend_month_pred_severity"))
hotspots_pred = read_spark_csv_folder(os.path.join(DASH_BASE, "hotspots_city_risk"))
sev_weather_pred = read_spark_csv_folder(os.path.join(DASH_BASE, "severity_by_weather"))
sev_hour_pred = read_spark_csv_folder(os.path.join(DASH_BASE, "severity_by_hour"))

# ---- actual (historical / ground truth) ----
trend_actual = read_spark_csv_folder(os.path.join(DASH_BASE, "trend_month_actual_severity"))
hotspots_actual = read_spark_csv_folder(os.path.join(DASH_BASE, "hotspots_city_actual"))
sev_weather_actual = read_spark_csv_folder(os.path.join(DASH_BASE, "severity_by_weather_actual"))
sev_hour_actual = read_spark_csv_folder(os.path.join(DASH_BASE, "severity_by_hour_actual"))

# graph
GRAPH_EDGE_VIEWS = {
    "City ↔ Weather": "edges_city_weather",
    "City ↔ Hour": "edges_city_hour",
    "Hour ↔ Weather": "edges_hour_weather",
}

edges_full = pd.DataFrame()

trend = pd.DataFrame()
hotspots = pd.DataFrame()
sev_weather = pd.DataFrame()
sev_hour = pd.DataFrame()

# =========================
# Sidebar
# =========================
st.title("US Accidents 2016–2023")
st.sidebar.title("Dashboard")

dash = st.sidebar.radio(
    "Select View",
    [
        "Analytics Overview (Actual Data)",
        "ML & Graph Insights (Model Results)",
    ],
    index=0
)

if dash.startswith("Analytics"):
    trend = trend_actual
    hotspots = hotspots_actual
    sev_weather = sev_weather_actual
    sev_hour = sev_hour_actual
else:
    trend = trend_pred
    hotspots = hotspots_pred
    sev_weather = sev_weather_pred
    sev_hour = sev_hour_pred

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

# Cards per panel
cards_per_panel = 3

# Year
year_opts = []
year_col_tr = pick_col(trend, ["year"])
if not trend.empty and year_col_tr:
    trend[year_col_tr] = safe_int_series(trend[year_col_tr])
    year_opts = sorted([int(x) for x in trend[year_col_tr].dropna().unique().tolist()])

default_year = [2023, 2022] if 2023 in year_opts else ([year_opts[-1]] if year_opts else [])
year_sel = st.sidebar.multiselect("Year", year_opts, default=default_year)

# State (codes)
state_opts = []
state_col_hs = pick_col(hotspots, ["state", "State"])
if not hotspots.empty and state_col_hs:
    state_opts = sorted([str(x).upper() for x in hotspots[state_col_hs].dropna().unique().tolist()])
state_sel = st.sidebar.multiselect(
    "State",
    state_opts,
    default=state_opts[:8] if len(state_opts) > 8 else state_opts
)

# Weather category
weather_opts = []
bucket_col_sw = pick_col(sev_weather, ["weather_bucket", "weather"])
if not sev_weather.empty and bucket_col_sw:
    weather_opts = sorted([str(x) for x in sev_weather[bucket_col_sw].dropna().unique().tolist()])
weather_sel = st.sidebar.multiselect("Weather Category", weather_opts, default=weather_opts)

topn_hotspots = 50
topn_state_map = 20
topn_edges = 50
topn_hubs = 50

st.sidebar.markdown("---")
st.sidebar.subheader("Graph View")

graph_view = st.sidebar.selectbox(
    "Connectivity type",
    list(GRAPH_EDGE_VIEWS.keys()),
    index=0
)

edges_folder = GRAPH_EDGE_VIEWS[graph_view]
edges_full = read_spark_csv_folder(os.path.join(GRAPH_BASE, edges_folder))

# =========================================================
# Severity col selection (Actual vs Predicted)
# =========================================================
def get_severity_col_for_mode(df: pd.DataFrame, severity_mode: str):
    """
    severity_mode:
      - "ANALYTICS" -> prefer actual severity
      - "ML"        -> prefer predicted severity if exists
    """
    if df is None or df.empty:
        return None, None

    actual_candidates = ["severity", "Severity", "actual_severity", "label_severity"]
    pred_candidates = ["pred_severity", "predicted_severity", "severity_pred"]

    if severity_mode == "ANALYTICS":
        col = pick_col(df, actual_candidates) or pick_col(df, pred_candidates)
        return col, "Severity"

    col = pick_col(df, pred_candidates) or pick_col(df, actual_candidates)
    return col, "Predicted Severity (Model)"

# =========================================================
# Sections (CHART FIRST, TABLE LATER)
# =========================================================
def render_trend(severity_mode: str):
    with st.expander("📈 Monthly Trend", expanded=True):
        if trend.empty:
            st.warning("trend_month_* not found.")
            return

        df = drop_if_exists(trend.copy(), ["avg_conf", "Avg_conf", "AVG_CONF"])

        year_col = pick_col(df, ["year"])
        month_col = pick_col(df, ["month"])
        n_col = pick_col(df, ["n", "count", "total"])
        state_col = pick_col(df, ["State", "state"])
        sev_col, sev_label = get_severity_col_for_mode(df, severity_mode)

        if year_col and year_sel:
            df[year_col] = safe_int_series(df[year_col])
            df = df[df[year_col].isin(year_sel)]
        if state_col and state_sel:
            df[state_col] = df[state_col].astype(str).str.upper()
            df = df[df[state_col].isin(state_sel)]
        if n_col:
            df = to_numeric(df, n_col)

        total_cases = int(df[n_col].fillna(0).sum()) if n_col else 0

        # Peak month (by total cases)
        peak_month = "-"
        if month_col and n_col and not df.empty:
            agg = df.groupby(month_col, as_index=False)[n_col].sum().sort_values(n_col, ascending=False)
            if not agg.empty:
                peak_month = str(agg.iloc[0][month_col])

        # High severity share (if severity exists)
        hi_share = "-"
        if sev_col and n_col and not df.empty:
            tmp = df[[sev_col, n_col]].copy()
            tmp[sev_col] = pd.to_numeric(tmp[sev_col], errors="coerce")
            tmp[n_col] = pd.to_numeric(tmp[n_col], errors="coerce").fillna(0)
            total = tmp[n_col].sum()
            if total > 0:
                hi = tmp[tmp[sev_col] >= 3][n_col].sum()
                hi_share = f"{(hi/total)*100:.1f}%"

        cards = [
            {"title": "Total Cases", "value": f"{total_cases:,}", "subtitle": "After filters (year/state)", "icon": "🧾"},
            {"title": "Peak Month", "value": peak_month, "subtitle": "Highest cases month", "icon": "📌"},
            {"title": "High Severity Share", "value": hi_share, "subtitle": "Severity ≥ 3", "icon": "⚠️"},
            {"title": "Mode", "value": ("Actual" if severity_mode=="ANALYTICS" else "Model"), "subtitle": sev_label, "icon": "🧠"},
        ]
        render_cards(cards, ncols=cards_per_panel)

        st.markdown("##### Charts")
        if month_col and sev_col and n_col:
            tmp = df[[month_col, sev_col, n_col]].copy()
            tmp[sev_col] = tmp[sev_col].apply(severity_label_en)
            piv = tmp.pivot_table(index=month_col, columns=sev_col, values=n_col, aggfunc="sum").fillna(0).sort_index()
            st.caption("Cases by month (split by severity)")
            st.line_chart(piv)
        elif month_col and n_col:
            agg = df.groupby(month_col, as_index=False)[n_col].sum().sort_values(month_col)
            st.caption("Cases by month (total)")
            st.line_chart(agg.set_index(month_col)[n_col])
        else:
            st.info("Need month + cases (n) columns to plot the trend.")

        st.markdown("##### Table")
        show = df.copy()
        if sev_col and sev_col in show.columns:
            show[sev_col] = show[sev_col].apply(severity_label_en)

        rename_map = {}
        if year_col: rename_map[year_col] = "Year"
        if month_col: rename_map[month_col] = "Month"
        if sev_col: rename_map[sev_col] = sev_label
        if n_col: rename_map[n_col] = "Cases"
        if state_col: rename_map[state_col] = "State"
        st.dataframe(show.rename(columns=rename_map), use_container_width=True, height=380)


def render_hotspot(severity_mode: str):
    with st.expander("🔥 City Hotspots (Risk Priority)", expanded=True):
        if hotspots.empty:
            st.warning("hotspots_city_* not found.")
            return

        df = drop_if_exists(hotspots.copy(), ["avg_conf", "Avg_conf", "AVG_CONF"])

        state_col = pick_col(df, ["State", "state"])
        city_col = pick_col(df, ["City", "city"])
        year_col = pick_col(df, ["year"])
        n_col = pick_col(df, ["n", "count", "total"])
        risk_col = pick_col(df, ["risk_score", "risk"])
        avg_col = pick_col(df, ["avg_pred_sev", "avg_sev", "avg_severity"])

        if year_col and year_sel:
            df[year_col] = safe_int_series(df[year_col])
            df = df[df[year_col].isin(year_sel)]
        if state_col and state_sel:
            df[state_col] = df[state_col].astype(str).str.upper()
            df = df[df[state_col].isin(state_sel)]

        if n_col: df = to_numeric(df, n_col)
        if avg_col: df = to_numeric(df, avg_col)
        if risk_col: df = to_numeric(df, risk_col)

        if not risk_col and avg_col and n_col:
            df["risk_score"] = df[avg_col].fillna(0) * df[n_col].fillna(0)
            risk_col = "risk_score"

        if risk_col:
            df_show = df.sort_values(risk_col, ascending=False).head(topn_hotspots)
        else:
            df_show = df.head(topn_hotspots)

        top_city = str(df_show.iloc[0][city_col]) if (not df_show.empty and city_col) else "-"
        total_cases = int(df[n_col].fillna(0).sum()) if n_col else 0
        top_risk = "-"
        if risk_col and not df_show.empty:
            try:
                top_risk = f"{float(df_show.iloc[0][risk_col]):,.1f}"
            except:
                top_risk = str(df_show.iloc[0][risk_col])

        avg_sev_overall = "-"
        if avg_col and n_col and not df.empty:
            # weighted avg severity
            try:
                w = (df[avg_col].fillna(0) * df[n_col].fillna(0)).sum()
                s = df[n_col].fillna(0).sum()
                if s > 0:
                    avg_sev_overall = f"{(w/s):.2f}"
            except:
                pass

        cards = [
            {"title": "Top City", "value": top_city, "subtitle": "Highest priority (risk)", "icon": "🏙️"},
            {"title": "Total Cases", "value": f"{total_cases:,}", "subtitle": "After filters", "icon": "🧾"},
            {"title": "Top Risk Score", "value": top_risk, "subtitle": "Cases × Avg Severity", "icon": "🔥"},
            {"title": "Avg Severity", "value": avg_sev_overall, "subtitle": "Weighted average", "icon": "📊"},
        ]
        render_cards(cards, ncols=cards_per_panel)

        st.caption(
            "Risk Score explanation: **Risk Score = Cases × Average Severity**. "
            "It prioritizes locations with both high volume and high severity."
        )

        st.markdown("##### Chart")
        if city_col and n_col:
            top_vol = df.groupby(city_col, as_index=False)[n_col].sum().sort_values(n_col, ascending=False).head(10)
            top_vol = top_vol.rename(columns={city_col: "City", n_col: "Cases"}).set_index("City")
            st.caption("Top 10 by Cases")
            st.bar_chart(top_vol)

        st.markdown("##### Table")
        rename_map = {}
        if city_col: rename_map[city_col] = "City"
        if state_col: rename_map[state_col] = "State"
        if year_col: rename_map[year_col] = "Year"
        if n_col: rename_map[n_col] = "Cases"
        if avg_col: rename_map[avg_col] = "Avg Severity"
        if risk_col: rename_map[risk_col] = "Risk Score (Cases × Avg Severity)"
        st.dataframe(df_show.rename(columns=rename_map), use_container_width=True, height=380)


def render_weather(severity_mode: str):
    with st.expander("🌦️ Weather Drivers", expanded=True):
        if sev_weather.empty:
            st.warning("severity_by_weather* not found.")
            return

        df = drop_if_exists(sev_weather.copy(), ["avg_conf", "Avg_conf", "AVG_CONF"])

        state_col = pick_col(df, ["State", "state"])
        year_col = pick_col(df, ["year"])
        bucket_col = pick_col(df, ["weather_bucket", "weather"])
        n_col = pick_col(df, ["n", "count", "total"])
        sev_col, sev_label = get_severity_col_for_mode(df, severity_mode)

        if year_col and year_sel:
            df[year_col] = safe_int_series(df[year_col])
            df = df[df[year_col].isin(year_sel)]
        if state_col and state_sel:
            df[state_col] = df[state_col].astype(str).str.upper()
            df = df[df[state_col].isin(state_sel)]
        if bucket_col and weather_sel:
            df[bucket_col] = df[bucket_col].astype(str)
            df = df[df[bucket_col].isin(weather_sel)]
        if n_col:
            df = to_numeric(df, n_col)

        total_cases = int(df[n_col].fillna(0).sum()) if n_col else 0
        n_buckets = int(df[bucket_col].nunique()) if bucket_col and not df.empty else 0

        # top weather by volume
        top_weather = "-"
        if bucket_col and n_col and not df.empty:
            agg = df.groupby(bucket_col, as_index=False)[n_col].sum().sort_values(n_col, ascending=False)
            if not agg.empty:
                top_weather = str(agg.iloc[0][bucket_col])

        # high severity share overall (>=3)
        hi_share = "-"
        if sev_col and n_col and not df.empty:
            tmp = df[[sev_col, n_col]].copy()
            tmp[sev_col] = pd.to_numeric(tmp[sev_col], errors="coerce")
            tmp[n_col] = pd.to_numeric(tmp[n_col], errors="coerce").fillna(0)
            total = tmp[n_col].sum()
            if total > 0:
                hi = tmp[tmp[sev_col] >= 3][n_col].sum()
                hi_share = f"{(hi/total)*100:.1f}%"

        cards = [
            {"title": "Total Cases", "value": f"{total_cases:,}", "subtitle": "After filters", "icon": "🧾"},
            {"title": "Weather Buckets", "value": f"{n_buckets:,}", "subtitle": "Unique categories", "icon": "🗂️"},
            {"title": "Top Weather", "value": top_weather, "subtitle": "Highest volume", "icon": "🌧️"},
            {"title": "High Severity Share", "value": hi_share, "subtitle": "Severity ≥ 3", "icon": "⚠️"},
        ]
        render_cards(cards, ncols=cards_per_panel)

        st.markdown("##### Charts")
        c1, c2 = st.columns(2)

        with c1:
            if bucket_col and sev_col and n_col:
                tmp = df[[bucket_col, sev_col, n_col]].copy()
                tmp[sev_col] = pd.to_numeric(tmp[sev_col], errors="coerce")
                total = tmp.groupby(bucket_col, as_index=False)[n_col].sum().rename(columns={n_col: "total_n"})
                hi = tmp[tmp[sev_col] >= 3].groupby(bucket_col, as_index=False)[n_col].sum().rename(columns={n_col: "hi_n"})
                merged = total.merge(hi, on=bucket_col, how="left").fillna(0)
                merged["high_sev_share_pct"] = (merged["hi_n"] / merged["total_n"].replace(0, pd.NA)) * 100
                merged = merged.dropna(subset=["high_sev_share_pct"]).sort_values("high_sev_share_pct", ascending=False).head(12)
                st.caption("Share of High Severity (≥3) by Weather")
                st.bar_chart(merged.set_index(bucket_col)["high_sev_share_pct"])
            else:
                st.info("Need weather_bucket + severity + cases (n).")

        with c2:
            if bucket_col and n_col:
                vol = (
                    df.groupby(bucket_col, as_index=False)[n_col]
                      .sum()
                      .sort_values(n_col, ascending=False)
                      .head(8)
                      .rename(columns={bucket_col: "Weather", n_col: "Cases"})
                )
                pie_vega(vol, "Weather", "Cases", "Volume by Weather (Top 8)")
            else:
                st.info("Need weather_bucket + cases (n).")

        st.markdown("##### Table")
        show = df.copy()
        if sev_col and sev_col in show.columns:
            show[sev_col] = show[sev_col].apply(severity_label_en)

        rename_map = {}
        if bucket_col: rename_map[bucket_col] = "Weather Category"
        if sev_col: rename_map[sev_col] = sev_label
        if n_col: rename_map[n_col] = "Cases"
        if year_col: rename_map[year_col] = "Year"
        if state_col: rename_map[state_col] = "State"
        st.dataframe(show.rename(columns=rename_map), use_container_width=True, height=380)


def render_hour(severity_mode: str):
    with st.expander("🕒 Time Drivers (Hour of Day)", expanded=True):
        if sev_hour.empty:
            st.warning("severity_by_hour* not found.")
            return

        df = drop_if_exists(sev_hour.copy(), ["avg_conf", "Avg_conf", "AVG_CONF"])

        hour_col = pick_col(df, ["hour"])
        year_col = pick_col(df, ["year"])
        state_col = pick_col(df, ["State", "state"])
        bucket_col = pick_col(df, ["weather_bucket", "weather"])
        n_col = pick_col(df, ["n", "count", "total"])
        sev_col, sev_label = get_severity_col_for_mode(df, severity_mode)

        if year_col and year_sel:
            df[year_col] = safe_int_series(df[year_col])
            df = df[df[year_col].isin(year_sel)]
        if state_col and state_sel:
            df[state_col] = df[state_col].astype(str).str.upper()
            df = df[df[state_col].isin(state_sel)]
        if bucket_col and weather_sel:
            df[bucket_col] = df[bucket_col].astype(str)
            df = df[df[bucket_col].isin(weather_sel)]
        if hour_col:
            df[hour_col] = safe_int_series(df[hour_col])
        if n_col:
            df = to_numeric(df, n_col)

        total = int(df[n_col].fillna(0).sum()) if n_col else 0

        peak_hour = "-"
        if hour_col and n_col and not df.empty:
            agg = df.groupby(hour_col, as_index=False)[n_col].sum().sort_values(n_col, ascending=False)
            if not agg.empty:
                peak_hour = str(agg.iloc[0][hour_col])

        hi_share = "-"
        if sev_col and n_col and not df.empty:
            tmp = df[[sev_col, n_col]].copy()
            tmp[sev_col] = pd.to_numeric(tmp[sev_col], errors="coerce")
            tmp[n_col] = pd.to_numeric(tmp[n_col], errors="coerce").fillna(0)
            total_n = tmp[n_col].sum()
            if total_n > 0:
                hi = tmp[tmp[sev_col] >= 3][n_col].sum()
                hi_share = f"{(hi/total_n)*100:.1f}%"

        cards = [
            {"title": "Total Cases", "value": f"{total:,}", "subtitle": "After filters", "icon": "🧾"},
            {"title": "Peak Hour", "value": peak_hour, "subtitle": "Highest volume hour", "icon": "⏱️"},
            {"title": "High Severity Share", "value": hi_share, "subtitle": "Severity ≥ 3", "icon": "⚠️"},
            {"title": "Mode", "value": ("Actual" if severity_mode=="ANALYTICS" else "Model"), "subtitle": sev_label, "icon": "🧠"},
        ]
        render_cards(cards, ncols=cards_per_panel)

        st.markdown("##### Chart")
        if hour_col and sev_col and n_col:
            tmp = df[[hour_col, sev_col, n_col]].copy()
            tmp[sev_col] = tmp[sev_col].apply(severity_label_en)
            piv = tmp.pivot_table(index=hour_col, columns=sev_col, values=n_col, aggfunc="sum").fillna(0).sort_index()
            st.caption("Hourly pattern (cases) by severity")
            st.line_chart(piv)
        elif hour_col and n_col:
            agg = df.groupby(hour_col, as_index=False)[n_col].sum().sort_values(hour_col)
            st.caption("Hourly pattern (total cases)")
            st.line_chart(agg.set_index(hour_col)[n_col])
        else:
            st.info("Need hour + cases (n) to plot.")

        st.markdown("##### Table")
        show = df.copy()
        if sev_col and sev_col in show.columns:
            show[sev_col] = show[sev_col].apply(severity_label_en)

        rename_map = {}
        if hour_col: rename_map[hour_col] = "Hour"
        if sev_col: rename_map[sev_col] = sev_label
        if n_col: rename_map[n_col] = "Cases"
        if year_col: rename_map[year_col] = "Year"
        if state_col: rename_map[state_col] = "State"
        if bucket_col: rename_map[bucket_col] = "Weather Category"
        st.dataframe(show.rename(columns=rename_map), use_container_width=True, height=380)


def render_map():
    with st.expander("🗺️ Map (Top States by Cases)", expanded=True):
        if hotspots.empty:
            st.warning("hotspots_city_* not found.")
            return

        df = drop_if_exists(hotspots.copy(), ["avg_conf", "Avg_conf", "AVG_CONF"])

        state_col = pick_col(df, ["State", "state"])
        year_col = pick_col(df, ["year"])
        n_col = pick_col(df, ["n", "count", "total"])

        if not state_col or not n_col:
            st.warning("Missing required columns for map (state / cases).")
            return

        if year_col and year_sel:
            df[year_col] = safe_int_series(df[year_col])
            df = df[df[year_col].isin(year_sel)]

        if state_sel:
            df[state_col] = df[state_col].astype(str).str.upper()
            df = df[df[state_col].isin(state_sel)]

        df = to_numeric(df, n_col)
        if df.empty:
            st.warning("No data after filters.")
            return

        tmp = df[[state_col, n_col]].copy()
        tmp[state_col] = clean_state_to_name(tmp[state_col])
        tmp[n_col] = pd.to_numeric(tmp[n_col], errors="coerce").fillna(0)

        agg = tmp.groupby(state_col, as_index=False)[n_col].sum()
        agg = agg.rename(columns={state_col: "StateName", n_col: "Cases"})

        mapdf = (
            agg.merge(STATE_MAP_DF, on="StateName", how="inner")
               .sort_values("Cases", ascending=False)
               .head(topn_state_map)
               .copy()
        )

        if mapdf.empty:
            st.warning("State-to-lat/lon join failed.")
            return

        # cards
        top_state = str(mapdf.iloc[0]["StateName"]) if not mapdf.empty else "-"
        top_cases = f"{int(mapdf.iloc[0]['Cases']):,}" if not mapdf.empty else "-"
        covered = f"{len(mapdf):,}/{len(STATE_MAP_DF):,}"

        cards = [
            {"title": "Top State", "value": top_state, "subtitle": "Highest cases (in map view)", "icon": "🏁"},
            {"title": "Top Cases", "value": top_cases, "subtitle": "Cases in top state", "icon": "🧾"},
            {"title": "States Shown", "value": covered, "subtitle": "Joined with lat/lon", "icon": "🗺️"},
            {"title": "Map View", "value": "Scatter", "subtitle": "Bubble size ~ sqrt(cases)", "icon": "🫧"},
        ]
        render_cards(cards, ncols=cards_per_panel)

        mapdf["tooltip"] = mapdf["StateName"] + "\nCases: " + mapdf["Cases"].map(lambda x: f"{int(x):,}")
        mapdf["radius"] = (mapdf["Cases"].clip(lower=0) ** 0.5)

        picked_state = None
        if state_sel and len(state_sel) == 1:
            picked_state = STATE_CODE_TO_NAME.get(state_sel[0].upper(), state_sel[0])

        base_layer = pdk.Layer(
            "ScatterplotLayer",
            data=mapdf,
            get_position='[lon, lat]',
            get_radius="radius",
            radius_scale=120,
            radius_min_pixels=4,
            radius_max_pixels=40,
            pickable=True,
            auto_highlight=True,
            get_fill_color=[80, 160, 255, 120],
            get_line_color=[200, 200, 200, 160],
            line_width_min_pixels=1,
        )

        layers = [base_layer]

        if picked_state:
            sel = mapdf[mapdf["StateName"] == picked_state].copy()
            if not sel.empty:
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=sel,
                        get_position='[lon, lat]',
                        get_radius="radius",
                        radius_scale=180,
                        radius_min_pixels=10,
                        radius_max_pixels=70,
                        pickable=True,
                        get_fill_color=[255, 90, 90, 180],
                        get_line_color=[255, 255, 255, 220],
                        line_width_min_pixels=2,
                    )
                )

        deck = pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=39.5, longitude=-98.35, zoom=3.2, pitch=0),
            layers=layers,
            tooltip={"text": "{tooltip}"},
        )

        st.markdown("##### Map")
        st.pydeck_chart(deck, use_container_width=True)

        st.markdown("##### State Ranking (Cases)")
        view_tbl = mapdf[["StateName", "Cases"]].rename(columns={"StateName": "State"})
        st.bar_chart(view_tbl.set_index("State")["Cases"])

        st.markdown("##### Table")
        st.dataframe(view_tbl, use_container_width=True, height=260)


def heatmap_pair(
    df_edges: pd.DataFrame,
    a_col: str,
    b_col: str,
    w_col: str,
    top_a: int = 15,
    top_b: int = 12,
    a_title: str = "A",
    b_title: str = "B",
):
    if df_edges is None or df_edges.empty:
        st.info("No data for heatmap.")
        return

    d = df_edges[[a_col, b_col, w_col]].copy()
    d[a_col] = d[a_col].astype(str)
    d[b_col] = d[b_col].astype(str)
    d[w_col] = pd.to_numeric(d[w_col], errors="coerce").fillna(0)

    topA = (
        d.groupby(a_col, as_index=False)[w_col].sum()
         .sort_values(w_col, ascending=False)
         .head(top_a)[a_col].tolist()
    )
    d = d[d[a_col].isin(topA)]

    topB = (
        d.groupby(b_col, as_index=False)[w_col].sum()
         .sort_values(w_col, ascending=False)
         .head(top_b)[b_col].tolist()
    )
    d = d[d[b_col].isin(topB)]

    d = d.groupby([a_col, b_col], as_index=False)[w_col].sum()

    a_order = d.groupby(a_col)[w_col].sum().sort_values(ascending=False).index.tolist()
    b_order = d.groupby(b_col)[w_col].sum().sort_values(ascending=False).index.tolist()

    chart = (
        alt.Chart(d)
        .mark_rect()
        .encode(
            x=alt.X(f"{b_col}:N", title=b_title, sort=b_order),
            y=alt.Y(f"{a_col}:N", title=a_title, sort=a_order),
            color=alt.Color(f"{w_col}:Q", title="Co-occurrence (count)"),
            tooltip=[
                alt.Tooltip(a_col, title=a_title),
                alt.Tooltip(b_col, title=b_title),
                alt.Tooltip(w_col, title="Count"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)


def heatmap_city_hour(df_edges: pd.DataFrame, src_col: str, dst_col: str, w_col: str, top_cities: int = 15):
    if df_edges is None or df_edges.empty:
        st.info("No data for heatmap.")
        return

    d = df_edges[[src_col, dst_col, w_col]].copy()
    d[src_col] = d[src_col].astype(str)
    d[dst_col] = pd.to_numeric(d[dst_col], errors="coerce")
    d[w_col] = pd.to_numeric(d[w_col], errors="coerce").fillna(0)

    d = d.dropna(subset=[dst_col])
    d[dst_col] = d[dst_col].astype(int)

    top = (
        d.groupby(src_col, as_index=False)[w_col]
         .sum()
         .sort_values(w_col, ascending=False)
         .head(top_cities)[src_col]
         .tolist()
    )
    d = d[d[src_col].isin(top)]
    d = d.groupby([src_col, dst_col], as_index=False)[w_col].sum()

    order_city = d.groupby(src_col)[w_col].sum().sort_values(ascending=False).index.tolist()
    hour_order = list(range(0, 24))

    chart = (
        alt.Chart(d)
        .mark_rect()
        .encode(
            x=alt.X(f"{dst_col}:O", title="Hour (0–23)", sort=hour_order),
            y=alt.Y(f"{src_col}:N", title="City", sort=order_city),
            color=alt.Color(f"{w_col}:Q", title="Co-occurrence (count)"),
            tooltip=[
                alt.Tooltip(src_col, title="City"),
                alt.Tooltip(dst_col, title="Hour"),
                alt.Tooltip(w_col, title="Count"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)


def render_graph():
    with st.expander("🕸️ Graph Insights (Connectivity)", expanded=True):
        if edges_full.empty:
            st.warning(f"Graph edges not found for view: {graph_view} (folder: {edges_folder}).")
            return

        e = drop_if_exists(edges_full.copy(), ["avg_conf", "Avg_conf", "AVG_CONF"])

        cols = {c.lower(): c for c in e.columns}

        def col(name):
            return cols.get(name.lower())

        src_col = col("src") or col("source") or col("from")
        dst_col = col("dst") or col("target") or col("to")
        w_col = col("cooc") or col("weight") or col("count") or col("n")

        year_col = col("year")
        state_col = col("state") or col("src_state") or col("dst_state")
        bucket_col = col("weather_bucket")

        if not src_col or not dst_col or not w_col:
            st.warning("Edge schema not recognized. Need columns like src, dst, cooc/weight.")
            st.dataframe(e.head(30), use_container_width=True)
            return

        if year_col and year_sel:
            e[year_col] = safe_int_series(e[year_col])
            e = e[e[year_col].isin(year_sel)]

        if state_col and state_sel:
            e[state_col] = e[state_col].astype(str).str.upper()
            e = e[e[state_col].isin(state_sel)]

        if bucket_col and weather_sel:
            e[bucket_col] = e[bucket_col].astype(str)
            e = e[e[bucket_col].isin(weather_sel)]

        e[w_col] = safe_float_series(e[w_col])

        if e.empty:
            st.warning("No edges after filters. Loosen filters.")
            return

        e_top = e.sort_values(w_col, ascending=False).head(topn_edges)

        # Cards summary
        n_edges = len(e_top)
        strongest = "-"
        if not e_top.empty:
            try:
                strongest = f"{float(e_top.iloc[0][w_col]):,.0f}"
            except:
                strongest = str(e_top.iloc[0][w_col])

        unique_src = int(e_top[src_col].nunique()) if src_col in e_top.columns else 0
        unique_dst = int(e_top[dst_col].nunique()) if dst_col in e_top.columns else 0

        cards = [
            {"title": "Connectivity View", "value": graph_view, "subtitle": f"Folder: {edges_folder}", "icon": "🕸️"},
            {"title": "Top Edges", "value": f"{n_edges:,}", "subtitle": "Shown after filters", "icon": "🔗"},
            {"title": "Strongest Link", "value": strongest, "subtitle": "Max co-occurrence", "icon": "💪"},
            {"title": "Coverage", "value": f"{unique_src:,}×{unique_dst:,}", "subtitle": "Unique src × dst", "icon": "📦"},
        ]
        render_cards(cards, ncols=cards_per_panel)

        st.caption(f"Connectivity view: **{graph_view}**.")

        # ===== Chart FIRST (Heatmap) =====
        if ("City" in graph_view) and ("Hour" in graph_view):
            st.markdown("##### Heatmap — City vs Hour (Co-occurrence)")
            try:
                heatmap_city_hour(e_top, src_col=src_col, dst_col=dst_col, w_col=w_col, top_cities=15)
            except Exception as ex:
                st.warning(f"Heatmap failed: {ex}")
                st.info("Check that dst is numeric hour (0-23) and altair is installed.")
        elif ("City" in graph_view) and ("Weather" in graph_view):
            st.markdown("##### Heatmap — City vs Weather (Co-occurrence)")
            try:
                heatmap_pair(
                    e_top,
                    a_col=src_col,
                    b_col=dst_col,
                    w_col=w_col,
                    top_a=15,
                    top_b=10,
                    a_title="City",
                    b_title="Weather",
                )
            except Exception as ex:
                st.warning(f"Heatmap failed: {ex}")
        elif ("Hour" in graph_view) and ("Weather" in graph_view):
            st.markdown("##### Heatmap — Hour vs Weather (Co-occurrence)")
            try:
                heatmap_pair(
                    e_top,
                    a_col=src_col,
                    b_col=dst_col,
                    w_col=w_col,
                    top_a=24,
                    top_b=10,
                    a_title="Hour",
                    b_title="Weather",
                )
            except Exception as ex:
                st.warning(f"Heatmap failed: {ex}")

        st.markdown("##### Table — Top Edges (Strongest Connections)")
        st.dataframe(e_top, use_container_width=True, height=280)


# =========================================================
# PAGE
# =========================================================
if dash.startswith("Analytics"):
    st.header("Analytics Overview")

    render_trend("ANALYTICS")
    render_hotspot("ANALYTICS")
    render_weather("ANALYTICS")
    render_map()
    render_hour("ANALYTICS")
else:
    st.header("ML & Graph Insights")

    st.subheader("Insight 1 — Risk Hotspots")
    render_hotspot("ML")

    st.subheader("Insight 2 — Drivers: Weather & Time")
    render_weather("ML")
    render_hour("ML")

    st.subheader("Insight 3 — Connectivity (Graph Analytics)")
    render_graph()
