"""
Taxi Trip and Revenue Analysis Dashboard
=========================================
Dataset  : https://www.kaggle.com/datasets/itskoustavdas/indian-bike-rider-daily-operational-dataset-2025
Frontend : Streamlit  (Python UI library)
Backend  : Pandas, NumPy
Charts   : Plotly Express / Plotly Graph Objects
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# OLS trendlines require statsmodels — optional, falls back gracefully
try:
    import statsmodels  # noqa: F401
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Taxi Trip & Revenue Analysis",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; margin: 0; }
    .kpi-label { font-size: 0.82rem; color: #6b7280; text-transform: uppercase;
                 letter-spacing: 0.07em; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem; font-weight: 600; color: #1e3a5f;
        border-bottom: 2px solid #3b82f6; padding-bottom: 6px;
        margin-bottom: 14px; margin-top: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: #e9eef6; border-radius: 8px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px; padding: 8px 20px;
        font-weight: 500; font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #3b82f6 !important; color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING  (cached once)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(filepath: str):
    df = pd.read_csv(filepath)

    # Datetime columns
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["month"]      = df["date"].dt.to_period("M").astype(str)
    df["week"]       = df["date"].dt.isocalendar().week.astype(int)
    df["day_name"]   = df["date"].dt.day_name()
    df["month_name"] = df["date"].dt.strftime("%b %Y")

    # Gross revenue per row
    gross = df["earnings"] + df["peak_bonus"] + df["waiting_earnings"]

    # Derived financials
    df["net_earnings"]   = gross - df["fuel_cost"]
    df["revenue_per_km"] = df["earnings"] / df["distance_km"].replace(0, np.nan)
    df["profit_margin"]  = df["net_earnings"] / gross.replace(0, np.nan) * 100

    # Ride efficiency
    rides_safe = df["rides"].replace(0, np.nan)
    df["on_time_rate"] = df["on_time_rides"]   / rides_safe * 100
    df["cancel_rate"]  = df["cancelled_rides"] / rides_safe * 100

    # Pre-compute all heavy aggregations ─────────────────────────────────
    # 1. city summary
    city_summary = df.groupby("location", sort=False).agg(
        Rides      =("rides",            "sum"),
        Revenue    =("earnings",         "sum"),
        Net_Profit =("net_earnings",     "sum"),
        Avg_Rating =("customer_rating",  "mean"),
        Riders     =("rider_id",         "nunique"),
    ).reset_index().sort_values("Revenue", ascending=False)

    # 2. revenue components by city
    rev_city = df.groupby("location", sort=False).agg(
        Base_Earnings =("earnings",        "sum"),
        Peak_Bonus    =("peak_bonus",       "sum"),
        Waiting_Earn  =("waiting_earnings", "sum"),
        Fuel_Cost     =("fuel_cost",        "sum"),
    ).reset_index()

    # 3. profit margin by city
    margin_city = df.groupby("location", sort=False)["profit_margin"].mean().reset_index()

    # 4. trip outcomes by city
    trip_city = df.groupby("location", sort=False).agg(
        On_Time   =("on_time_rides",   "sum"),
        Cancelled =("cancelled_rides", "sum"),
        Missed    =("missed_rides",    "sum"),
        Total     =("rides",           "sum"),
    ).reset_index()
    trip_city["Cancel_Rate"] = trip_city["Cancelled"] / trip_city["Total"] * 100
    trip_city["OT_Rate"]     = trip_city["On_Time"]   / trip_city["Total"] * 100

    # 5. rider aggregation
    rider_agg = df.groupby("rider_id", sort=False).agg(
        Total_Rides   =("rides",            "sum"),
        Gross_Revenue =("earnings",         "sum"),
        Net_Earnings  =("net_earnings",     "sum"),
        Avg_Rating    =("customer_rating",  "mean"),
        Avg_Speed     =("avg_speed_kmph",   "mean"),
        On_Time_Rate  =("on_time_rate",     "mean"),
        Cancel_Rate   =("cancel_rate",      "mean"),
        Total_Distance=("distance_km",      "sum"),
        Shifts        =("date",             "nunique"),
        City          =("location",         lambda x: x.mode()[0]),
    ).reset_index()

    # 6. daily aggregation
    daily = df.groupby("date", sort=True).agg(
        Total_Rides   =("rides",          "sum"),
        Gross_Revenue =("earnings",       "sum"),
        Net_Profit    =("net_earnings",   "sum"),
        Fuel_Cost     =("fuel_cost",      "sum"),
        Avg_Rating    =("customer_rating","mean"),
    ).reset_index()

    # 7. day-of-week
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = df.groupby("day_name", sort=False).agg(
        Avg_Rides   =("rides",           "mean"),
        Avg_Revenue =("earnings",        "mean"),
        Avg_Rating  =("customer_rating", "mean"),
    ).reindex(day_order).reset_index()

    # 8. monthly city heatmap pivot
    monthly_city = df.groupby(["month_name","location"], sort=False)["net_earnings"].sum().reset_index()
    pivot = monthly_city.pivot(index="location", columns="month_name", values="net_earnings").fillna(0)

    # 9. city radar (normalised)
    city_radar = df.groupby("location", sort=False).agg(
        Avg_Rating    =("customer_rating", "mean"),
        On_Time_Rate  =("on_time_rate",    "mean"),
        Avg_Speed     =("avg_speed_kmph",  "mean"),
        Rev_per_km    =("revenue_per_km",  "mean"),
        Profit_Margin =("profit_margin",   "mean"),
    ).reset_index()
    for m in ["Avg_Rating","On_Time_Rate","Avg_Speed","Rev_per_km","Profit_Margin"]:
        mn, mx = city_radar[m].min(), city_radar[m].max()
        city_radar[f"{m}_n"] = (city_radar[m] - mn) / (mx - mn + 1e-9)

    # 10. small scatter samples (pre-sampled for speed)
    scatter_ovh  = df.sample(min(3000, len(df)), random_state=7)[
        ["distance_km","overhead_distance_km","location"]].copy()
    scatter_earn = df.sample(min(3000, len(df)), random_state=42)[
        ["fuel_cost","net_earnings","rides","location","rider_id","date","distance_km"]].copy()
    scatter_speed = rider_agg.sample(min(2000, len(rider_agg)), random_state=3)[
        ["Avg_Speed","Gross_Revenue","Total_Distance","City"]].copy()
    scatter_rating = rider_agg.sample(min(2000, len(rider_agg)), random_state=1)[
        ["Avg_Rating","Net_Earnings","Total_Rides","City","rider_id","Shifts"]].copy()
    scatter_wh   = df.sample(min(3000, len(df)), random_state=99)[
        ["work_hours","net_earnings","location"]].copy()

    return dict(
        df=df,
        city_summary=city_summary,
        rev_city=rev_city,
        margin_city=margin_city,
        trip_city=trip_city,
        rider_agg=rider_agg,
        daily=daily,
        dow=dow,
        pivot=pivot,
        city_radar=city_radar,
        scatter_ovh=scatter_ovh,
        scatter_earn=scatter_earn,
        scatter_speed=scatter_speed,
        scatter_rating=scatter_rating,
        scatter_wh=scatter_wh,
        min_date=df["date"].min().date(),
        max_date=df["date"].max().date(),
        all_cities=sorted(df["location"].dropna().unique().tolist()),
        rating_min=float(df["customer_rating"].min()),
        rating_max=float(df["customer_rating"].max()),
    )


DATA = load_data("BikeRidersInfoSheet.csv")
df_full      = DATA["df"]
all_cities   = DATA["all_cities"]
min_date     = DATA["min_date"]
max_date     = DATA["max_date"]
rating_min   = DATA["rating_min"]
rating_max   = DATA["rating_max"]

# ─────────────────────────────────────────────
# SIDEBAR FILTERS  (no external image fetch)
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🛵 Dashboard Filters")

    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    selected_cities = st.multiselect(
        "City / Location",
        options=all_cities,
        default=all_cities,
    )

    rating_range = st.slider(
        "Customer Rating",
        min_value=round(rating_min, 1),
        max_value=round(rating_max, 1),
        value=(round(rating_min, 1), round(rating_max, 1)),
        step=0.1,
    )

    st.divider()
    st.caption("Data: BikeRidersInfoSheet.csv")
    st.caption(f"Total records: {len(df_full):,}")

# ─────────────────────────────────────────────
# APPLY FILTERS  (only on raw df when user changes something)
# ─────────────────────────────────────────────
start_dt = pd.Timestamp(date_range[0]) if len(date_range) >= 1 else df_full["date"].min()
end_dt   = pd.Timestamp(date_range[1]) if len(date_range) == 2 else df_full["date"].max()

filters_are_default = (
    len(date_range) == 2
    and date_range[0] == min_date
    and date_range[1] == max_date
    and set(selected_cities) == set(all_cities)
    and round(rating_range[0], 1) == round(rating_min, 1)
    and round(rating_range[1], 1) == round(rating_max, 1)
)

if filters_are_default:
    # Use pre-computed aggregations — fastest path
    df           = df_full
    city_summary = DATA["city_summary"]
    rev_city     = DATA["rev_city"]
    margin_city  = DATA["margin_city"]
    trip_city    = DATA["trip_city"]
    rider_agg    = DATA["rider_agg"]
    daily        = DATA["daily"]
    dow          = DATA["dow"]
    pivot        = DATA["pivot"]
    city_radar   = DATA["city_radar"]
    scatter_ovh  = DATA["scatter_ovh"]
    scatter_earn = DATA["scatter_earn"]
    scatter_speed= DATA["scatter_speed"]
    scatter_rating=DATA["scatter_rating"]
    scatter_wh   = DATA["scatter_wh"]
else:
    # Re-compute only on filtered slice
    df = df_full[
        (df_full["date"] >= start_dt) &
        (df_full["date"] <= end_dt) &
        (df_full["location"].isin(selected_cities)) &
        (df_full["customer_rating"] >= rating_range[0]) &
        (df_full["customer_rating"] <= rating_range[1])
    ].copy()

    city_summary = df.groupby("location", sort=False).agg(
        Rides      =("rides",           "sum"),
        Revenue    =("earnings",        "sum"),
        Net_Profit =("net_earnings",    "sum"),
        Avg_Rating =("customer_rating", "mean"),
        Riders     =("rider_id",        "nunique"),
    ).reset_index().sort_values("Revenue", ascending=False)

    rev_city = df.groupby("location", sort=False).agg(
        Base_Earnings=("earnings",        "sum"),
        Peak_Bonus   =("peak_bonus",       "sum"),
        Waiting_Earn =("waiting_earnings", "sum"),
        Fuel_Cost    =("fuel_cost",        "sum"),
    ).reset_index()

    margin_city = df.groupby("location", sort=False)["profit_margin"].mean().reset_index()

    trip_city = df.groupby("location", sort=False).agg(
        On_Time   =("on_time_rides",   "sum"),
        Cancelled =("cancelled_rides", "sum"),
        Missed    =("missed_rides",    "sum"),
        Total     =("rides",           "sum"),
    ).reset_index()
    trip_city["Cancel_Rate"] = trip_city["Cancelled"] / trip_city["Total"] * 100
    trip_city["OT_Rate"]     = trip_city["On_Time"]   / trip_city["Total"] * 100

    rider_agg = df.groupby("rider_id", sort=False).agg(
        Total_Rides   =("rides",           "sum"),
        Gross_Revenue =("earnings",        "sum"),
        Net_Earnings  =("net_earnings",    "sum"),
        Avg_Rating    =("customer_rating", "mean"),
        Avg_Speed     =("avg_speed_kmph",  "mean"),
        On_Time_Rate  =("on_time_rate",    "mean"),
        Cancel_Rate   =("cancel_rate",     "mean"),
        Total_Distance=("distance_km",     "sum"),
        Shifts        =("date",            "nunique"),
        City          =("location",        lambda x: x.mode()[0]),
    ).reset_index()

    daily = df.groupby("date", sort=True).agg(
        Total_Rides   =("rides",          "sum"),
        Gross_Revenue =("earnings",       "sum"),
        Net_Profit    =("net_earnings",   "sum"),
        Fuel_Cost     =("fuel_cost",      "sum"),
        Avg_Rating    =("customer_rating","mean"),
    ).reset_index()

    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = df.groupby("day_name", sort=False).agg(
        Avg_Rides   =("rides",           "mean"),
        Avg_Revenue =("earnings",        "mean"),
        Avg_Rating  =("customer_rating", "mean"),
    ).reindex(day_order).reset_index()

    monthly_city = df.groupby(["month_name","location"], sort=False)["net_earnings"].sum().reset_index()
    pivot = monthly_city.pivot(index="location", columns="month_name", values="net_earnings").fillna(0)

    city_radar = df.groupby("location", sort=False).agg(
        Avg_Rating    =("customer_rating", "mean"),
        On_Time_Rate  =("on_time_rate",    "mean"),
        Avg_Speed     =("avg_speed_kmph",  "mean"),
        Rev_per_km    =("revenue_per_km",  "mean"),
        Profit_Margin =("profit_margin",   "mean"),
    ).reset_index()
    for m in ["Avg_Rating","On_Time_Rate","Avg_Speed","Rev_per_km","Profit_Margin"]:
        mn, mx = city_radar[m].min(), city_radar[m].max()
        city_radar[f"{m}_n"] = (city_radar[m] - mn) / (mx - mn + 1e-9)

    n = min(3000, len(df))
    scatter_ovh   = df.sample(n, random_state=7)[["distance_km","overhead_distance_km","location"]].copy()
    scatter_earn  = df.sample(n, random_state=42)[["fuel_cost","net_earnings","rides","location","rider_id","date","distance_km"]].copy()
    scatter_wh    = df.sample(n, random_state=99)[["work_hours","net_earnings","location"]].copy()
    nr = min(2000, len(rider_agg))
    scatter_speed = rider_agg.sample(nr, random_state=3)[["Avg_Speed","Gross_Revenue","Total_Distance","City"]].copy()
    scatter_rating= rider_agg.sample(nr, random_state=1)[["Avg_Rating","Net_Earnings","Total_Rides","City","rider_id","Shifts"]].copy()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🛵 Taxi Trip & Revenue Analysis Dashboard")
st.markdown(
    f"**Records:** {len(df):,} &nbsp;|&nbsp; "
    f"**Cities:** {', '.join(selected_cities)} &nbsp;|&nbsp; "
    f"**Period:** {start_dt.date()} → {end_dt.date()}"
)
st.divider()

# ─────────────────────────────────────────────
# KPI helper
# ─────────────────────────────────────────────
def kpi(col, value, label, fmt=",.0f", prefix="", suffix="", color="#3b82f6"):
    formatted = f"{prefix}{value:{fmt}}{suffix}"
    col.markdown(
        f"""<div class="kpi-card" style="border-left-color:{color};">
            <p class="kpi-value">{formatted}</p>
            <p class="kpi-label">{label}</p>
        </div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "💰 Revenue Analysis",
    "🚗 Trip Analysis",
    "🏆 Rider Performance",
    "📅 Time Trends",
])

# ════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Key Performance Indicators</p>', unsafe_allow_html=True)

    total_rides    = int(df["rides"].sum())
    total_earnings = df["earnings"].sum() + df["peak_bonus"].sum() + df["waiting_earnings"].sum()
    net_profit     = df["net_earnings"].sum()
    avg_rating     = df["customer_rating"].mean()
    total_distance = df["distance_km"].sum()
    unique_riders  = df["rider_id"].nunique()
    total_fuel     = df["fuel_cost"].sum()
    avg_rides_day  = daily["Total_Rides"].mean() if not daily.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, total_rides,    "Total Rides",         color="#3b82f6")
    kpi(c2, total_earnings, "Gross Revenue (₹)",   prefix="₹", color="#10b981")
    kpi(c3, net_profit,     "Net Profit (₹)",      prefix="₹", color="#f59e0b")
    kpi(c4, avg_rating,     "Avg Customer Rating", fmt=".2f",  color="#8b5cf6")

    st.markdown("<br/>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    kpi(c5, total_distance, "Total Distance (km)", fmt=",.1f", suffix=" km", color="#06b6d4")
    kpi(c6, unique_riders,  "Unique Riders",       color="#ef4444")
    kpi(c7, total_fuel,     "Total Fuel Cost (₹)", prefix="₹", color="#f97316")
    kpi(c8, avg_rides_day,  "Avg Rides / Day",     fmt=".1f",  color="#64748b")

    st.divider()
    st.markdown('<p class="section-header">City-level Summary</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(city_summary, x="location", y="Revenue", color="location",
                     title="Gross Revenue by City",
                     labels={"location":"City","Revenue":"Revenue (₹)"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.pie(city_summary, values="Rides", names="location",
                      title="Ride Share by City",
                      color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Filtered Dataset Preview</p>', unsafe_allow_html=True)
    st.dataframe(df.head(200), use_container_width=True, height=300)

# ════════════════════════════════════════════
# TAB 2 — REVENUE ANALYSIS
# ════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Revenue Breakdown</p>', unsafe_allow_html=True)

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(name="Base Earnings",    x=rev_city["location"], y=rev_city["Base_Earnings"],  marker_color="#3b82f6"))
    fig_rev.add_trace(go.Bar(name="Peak Bonus",       x=rev_city["location"], y=rev_city["Peak_Bonus"],     marker_color="#10b981"))
    fig_rev.add_trace(go.Bar(name="Waiting Earnings", x=rev_city["location"], y=rev_city["Waiting_Earn"],   marker_color="#f59e0b"))
    fig_rev.add_trace(go.Bar(name="Fuel Cost",        x=rev_city["location"], y=-rev_city["Fuel_Cost"],     marker_color="#ef4444"))
    fig_rev.update_layout(barmode="relative", title="Earnings Components vs Fuel Cost by City",
                          xaxis_title="City", yaxis_title="Amount (₹)",
                          plot_bgcolor="white", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_rev, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_m = px.bar(margin_city.sort_values("profit_margin", ascending=True),
                       x="profit_margin", y="location", orientation="h",
                       title="Average Profit Margin by City (%)",
                       labels={"profit_margin":"Profit Margin (%)","location":"City"},
                       color="profit_margin", color_continuous_scale="RdYlGn")
        fig_m.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
        st.plotly_chart(fig_m, use_container_width=True)
    with c2:
        fig_rpk = px.box(df, x="location", y="revenue_per_km",
                         title="Revenue per km Distribution by City",
                         labels={"revenue_per_km":"Revenue/km (₹)","location":"City"},
                         color="location", color_discrete_sequence=px.colors.qualitative.Set2)
        fig_rpk.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig_rpk, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Earnings vs Fuel Cost Scatter</p>', unsafe_allow_html=True)
    fig_sc = px.scatter(scatter_earn, x="fuel_cost", y="net_earnings",
                        color="location", size="rides",
                        hover_data=["rider_id","date","distance_km"],
                        title="Net Earnings vs Fuel Cost (bubble = rides count)",
                        labels={"fuel_cost":"Fuel Cost (₹)","net_earnings":"Net Earnings (₹)"},
                        opacity=0.6, color_discrete_sequence=px.colors.qualitative.Set1)
    fig_sc.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Peak Bonus Impact</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        df["has_peak_bonus"] = df["peak_bonus"] > 0
        bonus_compare = df.groupby("has_peak_bonus").agg(
            Avg_Earnings=("earnings",     "mean"),
            Avg_Net     =("net_earnings", "mean"),
        ).reset_index()
        bonus_compare["Bonus Type"] = bonus_compare["has_peak_bonus"].map(
            {True:"With Peak Bonus", False:"Without Peak Bonus"})
        fig_bp = px.bar(
            bonus_compare.melt(id_vars="Bonus Type", value_vars=["Avg_Earnings","Avg_Net"]),
            x="Bonus Type", y="value", color="variable", barmode="group",
            title="Avg Earnings: Peak Bonus vs No Bonus",
            labels={"value":"Amount (₹)","variable":"Metric"},
            color_discrete_sequence=["#3b82f6","#10b981"])
        fig_bp.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_bp, use_container_width=True)
    with c4:
        fig_hist = px.histogram(df, x="profit_margin", nbins=40,
                                title="Profit Margin Distribution",
                                labels={"profit_margin":"Profit Margin (%)"},
                                color_discrete_sequence=["#3b82f6"])
        fig_hist.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════
# TAB 3 — TRIP ANALYSIS
# ════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Trip Volume & Completion</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_tc = go.Figure()
        fig_tc.add_trace(go.Bar(name="On Time",   x=trip_city["location"], y=trip_city["On_Time"],   marker_color="#10b981"))
        fig_tc.add_trace(go.Bar(name="Cancelled", x=trip_city["location"], y=trip_city["Cancelled"], marker_color="#ef4444"))
        fig_tc.add_trace(go.Bar(name="Missed",    x=trip_city["location"], y=trip_city["Missed"],    marker_color="#f59e0b"))
        fig_tc.update_layout(barmode="group", title="Trip Outcomes by City",
                             xaxis_title="City", yaxis_title="Number of Rides",
                             plot_bgcolor="white")
        st.plotly_chart(fig_tc, use_container_width=True)
    with c2:
        fig_cr = px.bar(trip_city.sort_values("Cancel_Rate", ascending=False),
                        x="location", y=["OT_Rate","Cancel_Rate"], barmode="group",
                        title="On-Time Rate vs Cancellation Rate by City (%)",
                        labels={"value":"Rate (%)","location":"City","variable":"Metric"},
                        color_discrete_map={"OT_Rate":"#10b981","Cancel_Rate":"#ef4444"})
        fig_cr.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_cr, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Distance & Speed Analysis</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig_dist = px.histogram(df, x="distance_km", nbins=50, color="location",
                                title="Distance per Shift Distribution",
                                labels={"distance_km":"Distance (km)"},
                                barmode="overlay", opacity=0.7,
                                color_discrete_sequence=px.colors.qualitative.Set2)
        fig_dist.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_dist, use_container_width=True)
    with c4:
        fig_spd = px.violin(df, x="location", y="avg_speed_kmph", color="location", box=True,
                            title="Speed Distribution by City",
                            labels={"avg_speed_kmph":"Avg Speed (km/h)","location":"City"},
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_spd.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig_spd, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Idle Time & Overhead Distance</p>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        fig_idle = px.box(df, x="location", y="idle_minutes", color="location",
                          title="Idle Minutes Distribution by City",
                          labels={"idle_minutes":"Idle Minutes","location":"City"},
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_idle.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig_idle, use_container_width=True)
    with c6:
        fig_ovh = px.scatter(scatter_ovh, x="distance_km", y="overhead_distance_km",
                             color="location", opacity=0.5,
                             title="Overhead vs Actual Distance",
                             labels={"distance_km":"Actual Distance (km)","overhead_distance_km":"Overhead (km)"},
                             trendline="ols" if _HAS_STATSMODELS else None,
                             color_discrete_sequence=px.colors.qualitative.Set1)
        fig_ovh.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_ovh, use_container_width=True)

# ════════════════════════════════════════════
# TAB 4 — RIDER PERFORMANCE
# ════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">Top Riders</p>', unsafe_allow_html=True)

    top_n = st.slider("Show Top N Riders by Net Earnings", 5, 50, 15)
    top_riders = rider_agg.nlargest(top_n, "Net_Earnings")

    c1, c2 = st.columns(2)
    with c1:
        fig_tr = px.bar(top_riders.sort_values("Net_Earnings"),
                        x="Net_Earnings", y="rider_id", orientation="h", color="City",
                        title=f"Top {top_n} Riders by Net Earnings",
                        labels={"Net_Earnings":"Net Earnings (₹)","rider_id":"Rider ID"},
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig_tr.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_tr, use_container_width=True)
    with c2:
        fig_rat = px.scatter(scatter_rating, x="Avg_Rating", y="Net_Earnings",
                             color="City", size="Total_Rides",
                             hover_data=["rider_id","Shifts"],
                             title="Customer Rating vs Net Earnings",
                             labels={"Avg_Rating":"Avg Customer Rating","Net_Earnings":"Net Earnings (₹)"},
                             opacity=0.7, color_discrete_sequence=px.colors.qualitative.Set1)
        fig_rat.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_rat, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Rating & Cancellation Distribution</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig_rathist = px.histogram(df, x="customer_rating", nbins=30, color="location",
                                   title="Customer Rating Distribution by City",
                                   labels={"customer_rating":"Customer Rating"},
                                   barmode="overlay", opacity=0.75,
                                   color_discrete_sequence=px.colors.qualitative.Set2)
        fig_rathist.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_rathist, use_container_width=True)
    with c4:
        fig_cancel_rid = px.histogram(rider_agg, x="Cancel_Rate", nbins=30,
                                      title="Cancellation Rate Distribution Across Riders",
                                      labels={"Cancel_Rate":"Cancellation Rate (%)"},
                                      color_discrete_sequence=["#ef4444"])
        fig_cancel_rid.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_cancel_rid, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Speed vs Earnings & Radar Comparison</p>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        fig_spd_earn = px.scatter(scatter_speed, x="Avg_Speed", y="Gross_Revenue",
                                  color="City", size="Total_Distance",
                                  title="Average Speed vs Gross Revenue",
                                  labels={"Avg_Speed":"Avg Speed (km/h)","Gross_Revenue":"Gross Revenue (₹)"},
                                  opacity=0.65, color_discrete_sequence=px.colors.qualitative.Pastel1)
        fig_spd_earn.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_spd_earn, use_container_width=True)
    with c6:
        metrics = ["Avg_Rating","On_Time_Rate","Avg_Speed","Rev_per_km","Profit_Margin"]
        cats    = ["Avg Rating","On-Time Rate","Avg Speed","Rev/km","Profit Margin"]
        colors  = px.colors.qualitative.Set2
        fig_radar = go.Figure()
        for i, row in city_radar.iterrows():
            vals = [row[f"{m}_n"] for m in metrics] + [row[f"{metrics[0]}_n"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]],
                fill="toself", name=row["location"],
                line_color=colors[i % len(colors)],
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            title="City Performance Radar (Normalised)", showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Rider Leaderboard</p>', unsafe_allow_html=True)
    lb = top_riders[["rider_id","City","Total_Rides","Gross_Revenue",
                      "Net_Earnings","Avg_Rating","On_Time_Rate","Shifts"]].copy()
    lb = lb.rename(columns={
        "rider_id":"Rider ID","City":"City","Total_Rides":"Rides",
        "Gross_Revenue":"Gross Rev (₹)","Net_Earnings":"Net Earn (₹)",
        "Avg_Rating":"Rating","On_Time_Rate":"On-Time %","Shifts":"Shifts",
    })
    lb["Gross Rev (₹)"] = lb["Gross Rev (₹)"].round(2)
    lb["Net Earn (₹)"]  = lb["Net Earn (₹)"].round(2)
    lb["Rating"]        = lb["Rating"].round(2)
    lb["On-Time %"]     = lb["On-Time %"].round(1)
    st.dataframe(lb.reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════
# TAB 5 — TIME TRENDS
# ════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-header">Daily Revenue & Ride Trends</p>', unsafe_allow_html=True)

    fig_daily = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=("Daily Revenue (₹)","Daily Rides"),
                              vertical_spacing=0.08)
    fig_daily.add_trace(go.Scatter(x=daily["date"], y=daily["Gross_Revenue"], name="Gross Revenue",
                                   line=dict(color="#3b82f6", width=2)), row=1, col=1)
    fig_daily.add_trace(go.Scatter(x=daily["date"], y=daily["Net_Profit"],    name="Net Profit",
                                   line=dict(color="#10b981", width=2)), row=1, col=1)
    fig_daily.add_trace(go.Scatter(x=daily["date"], y=daily["Fuel_Cost"],     name="Fuel Cost",
                                   line=dict(color="#ef4444", width=1.5, dash="dot")), row=1, col=1)
    fig_daily.add_trace(go.Bar(    x=daily["date"], y=daily["Total_Rides"],   name="Total Rides",
                                   marker_color="#93c5fd"), row=2, col=1)
    fig_daily.update_layout(height=500, plot_bgcolor="white", hovermode="x unified")
    st.plotly_chart(fig_daily, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Day-of-Week Patterns</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_dow = px.bar(dow, x="day_name", y="Avg_Revenue",
                         title="Average Revenue by Day of Week",
                         labels={"day_name":"Day","Avg_Revenue":"Avg Revenue (₹)"},
                         color="Avg_Revenue", color_continuous_scale="Blues")
        fig_dow.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
        st.plotly_chart(fig_dow, use_container_width=True)
    with c2:
        fig_dow2 = px.line(dow, x="day_name", y=["Avg_Rides","Avg_Rating"],
                           title="Avg Rides & Rating by Day of Week",
                           labels={"day_name":"Day","value":"Value","variable":"Metric"},
                           markers=True,
                           color_discrete_map={"Avg_Rides":"#3b82f6","Avg_Rating":"#f59e0b"})
        fig_dow2.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_dow2, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Monthly City Revenue Heatmap</p>', unsafe_allow_html=True)
    if not pivot.empty:
        fig_heat = px.imshow(pivot,
                             title="Monthly Net Earnings Heatmap (₹) by City",
                             labels={"x":"Month","y":"City","color":"Net Earnings (₹)"},
                             color_continuous_scale="YlOrRd", aspect="auto")
        fig_heat.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-header">Work Hours vs Earnings Correlation</p>', unsafe_allow_html=True)
    fig_wh = px.scatter(scatter_wh, x="work_hours", y="net_earnings",
                        color="location", opacity=0.55,
                        trendline="ols" if _HAS_STATSMODELS else None,
                        title="Work Hours vs Net Earnings per Shift",
                        labels={"work_hours":"Work Hours","net_earnings":"Net Earnings (₹)"},
                        color_discrete_sequence=px.colors.qualitative.Set2)
    fig_wh.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_wh, use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:0.8rem;'>"
    "🛵 Taxi Trip & Revenue Analysis Dashboard &nbsp;|&nbsp; "
    "Built with Streamlit &amp; Plotly &nbsp;|&nbsp; "
    "Data: BikeRidersInfoSheet.csv"
    "</p>",
    unsafe_allow_html=True,
)
