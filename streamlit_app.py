"""
GRU Energy Audit Analyzer — Streamlit Web App
==============================================

A web interface for energy auditors to analyze customer utility data
before conducting on-site surveys.

Run locally:  streamlit run streamlit_app.py
Deploy:       Push to GitHub and connect to Streamlit Cloud
"""

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import core analysis modules
from gru_audit_v2 import (
    Config,
    TemperatureCache,
    MeterLoader,
    MeterFeatures,
    WeatherAnomalyDetector,
    AMILoader,
    AMIFeatures,
    detect_change_point,
    compute_degree_days,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="GRU Energy Audit Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize config
if "cfg" not in st.session_state:
    st.session_state.cfg = Config()

cfg = st.session_state.cfg


# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #2E86AB;
    }
    .anomaly-high {
        background: #ffebee;
        border-left-color: #C73E1D;
    }
    .anomaly-low {
        background: #e3f2fd;
        border-left-color: #457B9D;
    }
    .checklist-item {
        padding: 8px 0;
        border-bottom: 1px solid #eee;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_temperature_cached(start_date, end_date):
    """Fetch and cache temperature data."""
    cache = TemperatureCache(tempfile.gettempdir())
    return cache.get_or_fetch(start_date, end_date)


def create_consumption_chart(feats: dict, division: str):
    """Create consumption history chart."""
    df = feats["df"]
    unit = feats["unit"]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    normal = df[~df["anomaly"]]
    anomaly = df[df["anomaly"]]
    
    ax.bar(normal["mr_date"], normal["consumption"], width=20,
           color="#A3B18A", alpha=0.8, label="Normal")
    ax.bar(anomaly["mr_date"], anomaly["consumption"], width=20,
           color="#C73E1D", alpha=0.9, label="Anomaly")
    
    ax.plot(feats["rolling_avg"].index, feats["rolling_avg"].values,
            color="#1D3557", linewidth=2, linestyle="--", label="3-Period Avg")
    
    ax.set_title(f"{division} Consumption History", fontsize=12, fontweight="bold")
    ax.set_ylabel(unit)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    
    return fig


def create_weather_anomaly_chart(anomaly_result: dict, division: str):
    """Create weather-normalized anomaly chart."""
    df = anomaly_result["df"]
    
    if df.empty:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Top: Actual vs Predicted
    ax1.fill_between(df["mr_date"], df["ci_lower"], df["ci_upper"],
                    alpha=0.3, color="#A8DADC", label="95% CI")
    ax1.plot(df["mr_date"], df["predicted_daily"], 
            color="#1D3557", linewidth=2, linestyle="--", label="Predicted")
    
    normal = df[~df["anomaly"]]
    high = df[df["anomaly_high"]]
    low = df[df["anomaly_low"]]
    
    ax1.scatter(normal["mr_date"], normal["actual_daily"],
               color="#A3B18A", s=50, zorder=5, label="Normal")
    ax1.scatter(high["mr_date"], high["actual_daily"],
               color="#C73E1D", s=80, zorder=6, label="High", marker="^")
    ax1.scatter(low["mr_date"], low["actual_daily"],
               color="#457B9D", s=80, zorder=6, label="Low", marker="v")
    
    ax1.set_title(f"{division} — Weather-Normalized Analysis", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Daily Usage")
    ax1.legend(fontsize=8, loc="upper left")
    
    # Bottom: Z-scores
    colors = ["#C73E1D" if z > cfg.residual_z_threshold
             else "#457B9D" if z < -cfg.residual_z_threshold
             else "#A3B18A" for z in df["residual_z"]]
    
    ax2.bar(df["mr_date"], df["residual_z"], width=20, color=colors, alpha=0.7)
    ax2.axhline(cfg.residual_z_threshold, color="#C73E1D", linestyle="--", linewidth=1.5)
    ax2.axhline(-cfg.residual_z_threshold, color="#457B9D", linestyle="--", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_ylabel("Z-Score")
    ax2.set_title("Residual Z-Score", fontsize=10)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    return fig


def create_ami_profile_chart(feats: dict, unit: str):
    """Create AMI hourly profile chart."""
    hourly = feats["hourly_profile"]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    colors = ["#C73E1D" if h == feats["peak_hour"] else "#6A4C93" for h in range(24)]
    ax.bar(hourly.index, hourly.values, color=colors, alpha=0.85, width=0.7)
    
    ax.axhline(feats["base_kwh"], color="#2A9D8F", linewidth=2,
              linestyle="--", label=f"Base ({feats['base_kw']:.2f} kW)")
    
    ax.set_title(f"Hourly Load Profile (Load Factor: {feats['load_factor']:.1%})",
                fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(f"{unit}/interval")
    ax.set_xticks(range(24))
    ax.legend(fontsize=8)
    plt.tight_layout()
    
    return fig


def display_metric_card(label, value, unit="", delta=None, is_anomaly=False):
    """Display a styled metric card."""
    css_class = "metric-card"
    if is_anomaly:
        css_class += " anomaly-high" if delta and delta > 0 else " anomaly-low"
    
    delta_html = ""
    if delta is not None:
        color = "#C73E1D" if delta > 0 else "#457B9D"
        delta_html = f'<span style="color: {color}; font-size: 0.9rem;">({delta:+.1f}%)</span>'
    
    if isinstance(value, float):
        value_str = f"{value:,.2f}"
    else:
        value_str = str(value)
    
    st.markdown(f"""
    <div class="{css_class}">
        <div style="font-size: 0.85rem; color: #666;">{label}</div>
        <div style="font-size: 1.5rem; font-weight: 600;">{value_str} {unit} {delta_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/lightning.svg", width=50)
    st.title("⚡ GRU Audit Tool")
    st.caption("Pre-Survey Analysis")
    
    st.divider()
    
    # File uploads
    st.subheader("📁 Upload Files")
    
    meter_file = st.file_uploader(
        "Customer Meter File",
        type=["xlsx", "xls"],
        help="GRU format with Consumption History tab"
    )
    
    ami_file = st.file_uploader(
        "AMI Interval Data (Optional)",
        type=["xlsx", "xls"],
        help="15-minute interval data"
    )
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    
    z_threshold = st.slider(
        "Anomaly Z-Threshold",
        min_value=1.5,
        max_value=4.0,
        value=2.5,
        step=0.1,
        help="Higher = fewer anomalies flagged"
    )
    cfg.residual_z_threshold = z_threshold
    
    min_periods = st.slider(
        "Min History Periods",
        min_value=4,
        max_value=12,
        value=8,
        help="Minimum billing periods for analysis"
    )
    cfg.min_history_periods = min_periods
    
    st.divider()
    
    # Info
    st.caption("v2.0 | Weather-Normalized Analysis")
    st.caption("© GRU Energy Audit Team")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">⚡ Energy Audit Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Pre-Survey Report Tool for Energy Auditors</p>', unsafe_allow_html=True)

if meter_file is None:
    # Welcome screen
    st.info("👈 Upload a customer meter file to begin analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Meter Analysis
        - Consumption history
        - Anomaly detection
        - Data quality scoring
        """)
    
    with col2:
        st.markdown("""
        ### 🌡️ Weather-Normalized
        - HDD/CDD modeling
        - Residual analysis
        - Persistent anomalies
        """)
    
    with col3:
        st.markdown("""
        ### ⚡ AMI Analysis
        - Load factor
        - Peak demand
        - TOU patterns
        """)
    
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS FILES
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading data..."):
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(meter_file.getvalue())
            tmp_path = tmp.name
        
        # Load meter data
        loader = MeterLoader(tmp_path)
        df_all = loader.load()
        customer = loader.get_customer_info()
        
        # Clean up
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOMER INFO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.markdown(f"### 👤 {customer.get('name', 'Unknown')}")
    st.caption(f"Account: {customer.get('account', 'N/A')}")

with col2:
    st.markdown(f"**Address:** {customer.get('address', 'N/A')}")
    st.caption(customer.get('city', ''))

with col3:
    st.markdown(f"**Own/Rent:** {customer.get('own_rent', 'N/A')}")
    st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d')}")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# FETCH TEMPERATURE
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner("Fetching weather data..."):
    start = df_all["mr_date"].min() - pd.Timedelta(days=35)
    end = df_all["mr_date"].max()
    df_temp = fetch_temperature_cached(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

weather_detector = WeatherAnomalyDetector(df_temp) if df_temp is not None else None

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS TABS
# ═══════════════════════════════════════════════════════════════════════════════

divisions = df_all["division"].unique().tolist()
tab_names = divisions + (["AMI Analysis"] if ami_file else []) + ["📋 Checklist"]

tabs = st.tabs(tab_names)

recommendations = []

# ═══════════════════════════════════════════════════════════════════════════════
# DIVISION TABS
# ═══════════════════════════════════════════════════════════════════════════════

for idx, division in enumerate(divisions):
    with tabs[idx]:
        df_div = loader.get_division(division)
        
        if df_div.empty:
            st.warning(f"No data available for {division}")
            continue
        
        # Compute features
        feats = MeterFeatures(df_div).compute()
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            display_metric_card("Total Reads", feats["n_reads"])
        with col2:
            display_metric_card("Daily Average", feats["daily_avg"], f"{feats['unit']}/day")
        with col3:
            display_metric_card("Peak Period", feats["peak"], feats["unit"])
        with col4:
            display_metric_card("Anomalies", feats["n_anomalies"], is_anomaly=feats["n_anomalies"] > 0)
        with col5:
            display_metric_card("Quality Score", feats["quality_score"], "/100")
        
        st.markdown("---")
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("📊 Consumption History")
            fig = create_consumption_chart(feats, division)
            st.pyplot(fig)
            plt.close(fig)
        
        with chart_col2:
            if weather_detector:
                st.subheader("🌡️ Weather-Normalized Analysis")
                anomaly_result = weather_detector.analyze(df_div, division)
                
                if not anomaly_result["df"].empty:
                    summary = anomaly_result["summary"]
                    
                    # Summary metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("High Anomalies", summary["n_anomaly_high"])
                    m2.metric("Low Anomalies", summary["n_anomaly_low"])
                    m3.metric("Persistent High", summary["n_persistent_high"])
                    m4.metric("Latest Z-Score", f"{summary['latest_z']:.2f}" if summary['latest_z'] else "N/A")
                    
                    if summary["n_persistent_high"] > 0:
                        recommendations.append(f"🔴 {division}: Persistent high usage — check equipment efficiency")
                else:
                    st.info("Insufficient data for weather-normalized analysis")
            else:
                st.warning("Temperature data unavailable")
        
        # Weather anomaly chart (full width)
        if weather_detector and not anomaly_result["df"].empty:
            st.subheader("📈 Anomaly Detection Details")
            fig2 = create_weather_anomaly_chart(anomaly_result, division)
            if fig2:
                st.pyplot(fig2)
                plt.close(fig2)
        
        # Change-point detection
        change = detect_change_point(df_div)
        if change:
            st.warning(f"""
            ⚠️ **Usage Regime Change Detected**
            - Date: {change['change_date'].strftime('%Y-%m-%d')}
            - Direction: {change['direction'].capitalize()}
            - Change: {change['pct_change']:+.1f}%
            """)
            recommendations.append(
                f"🟠 {division}: Usage changed {change['pct_change']:+.0f}% around {change['change_date'].strftime('%Y-%m')}"
            )
        
        # Anomaly table
        if feats["n_anomalies"] > 0:
            with st.expander(f"View {feats['n_anomalies']} Anomaly Periods"):
                anomaly_df = feats["df"][feats["df"]["anomaly"]][["mr_date", "consumption", "days", "avg_daily"]]
                anomaly_df["mr_date"] = anomaly_df["mr_date"].dt.strftime("%Y-%m-%d")
                st.dataframe(anomaly_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# AMI TAB
# ═══════════════════════════════════════════════════════════════════════════════

if ami_file:
    ami_tab_idx = len(divisions)
    with tabs[ami_tab_idx]:
        st.subheader("⚡ AMI Interval Analysis")
        
        try:
            # Save and load AMI file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(ami_file.getvalue())
                ami_tmp_path = tmp.name
            
            ami_loader = AMILoader(ami_tmp_path)
            df_ami = ami_loader.load()
            ami_feats = AMIFeatures(df_ami).compute()
            
            os.unlink(ami_tmp_path)
            
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                display_metric_card("Base Load", ami_feats["base_kw"], "kW")
            with col2:
                display_metric_card("Peak Demand", ami_feats["peak_kw"], "kW")
            with col3:
                display_metric_card("Load Factor", ami_feats["load_factor"] * 100, "%")
            with col4:
                display_metric_card("Peak Hour", f"{ami_feats['peak_hour']}:00")
            with col5:
                display_metric_card("TOU Ratio", ami_feats["tou_ratio"])
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🕐 Hourly Load Profile")
                fig = create_ami_profile_chart(ami_feats, ami_loader.unit)
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                st.subheader("📅 Daily Totals")
                daily = ami_feats["daily_series"]
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ["#C73E1D" if d == ami_feats["peak_day"].date() else "#2E86AB" 
                         for d in daily.index]
                ax.bar(daily.index, daily.values, color=colors, alpha=0.85)
                ax.axhline(ami_feats["daily_avg"], color="#F18F01", linewidth=2,
                          linestyle="--", label=f"Avg ({ami_feats['daily_avg']:.1f})")
                ax.set_ylabel(f"{ami_loader.unit}/day")
                ax.legend()
                fig.autofmt_xdate()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            # Recommendations
            if ami_feats["load_factor"] < 0.3:
                recommendations.append("🟠 Low load factor (<30%) — peaky demand pattern")
            if ami_feats["tou_ratio"] > 2:
                recommendations.append("🟠 High on-peak usage — consider TOU rate optimization")
                
        except Exception as e:
            st.error(f"Error loading AMI data: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# CHECKLIST TAB
# ═══════════════════════════════════════════════════════════════════════════════

checklist_tab_idx = len(divisions) + (1 if ami_file else 0)
with tabs[checklist_tab_idx]:
    st.subheader("📋 Pre-Survey Checklist")
    
    # Analysis-based recommendations
    if recommendations:
        st.markdown("### 🔍 Analysis Findings")
        for rec in recommendations:
            st.markdown(f"- {rec}")
        st.markdown("---")
    
    # Standard checklist
    st.markdown("### ✅ Standard Inspection Items")
    
    checklist_items = [
        ("Verify thermostat type and settings", False),
        ("Check air filter condition", False),
        ("Inspect windows and doors for air leaks", False),
        ("Document appliance ages and conditions", False),
        ("Review lighting (LED opportunities)", False),
        ("Check water heater temperature setting", False),
        ("Inspect ductwork for leaks", False),
        ("Verify HVAC maintenance history", False),
    ]
    
    col1, col2 = st.columns(2)
    
    for idx, (item, _) in enumerate(checklist_items):
        col = col1 if idx < len(checklist_items) // 2 else col2
        with col:
            st.checkbox(item, key=f"check_{idx}")
    
    st.markdown("---")
    
    # Notes section
    st.markdown("### 📝 Survey Notes")
    notes = st.text_area("Add notes for the on-site survey:", height=150)
    
    # Export button
    if st.button("📄 Generate Report Summary"):
        summary = f"""
# Pre-Survey Report Summary
**Customer:** {customer.get('name', 'N/A')}
**Account:** {customer.get('account', 'N/A')}
**Address:** {customer.get('address', 'N/A')}, {customer.get('city', '')}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Analysis Findings
{chr(10).join('- ' + r for r in recommendations) if recommendations else 'No significant issues detected.'}

## Notes
{notes if notes else 'None'}
        """
        st.download_button(
            label="⬇️ Download Summary",
            data=summary,
            file_name=f"{customer.get('account', 'report')}_summary.md",
            mime="text/markdown"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.divider()
st.caption("GRU Energy Audit Analyzer v2.0 | Weather-normalized analysis powered by Open-Meteo API")
