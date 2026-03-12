"""
Energy Audit Analyzer
Professional energy consumption analysis tool for auditors.

Features:
- Meter data analysis with anomaly detection
- AMI (interval) data analysis with load profiles
- Weather-normalized consumption analysis
- Advanced fractal analysis (Hurst exponent)
- PDF report export

Author: Energy Audit Tools
Version: 2.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor
import requests
import io
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Energy Audit Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color palette
COLORS = {
    "primary": "#1e3a5f",      # Dark blue
    "secondary": "#3498db",    # Light blue
    "accent": "#e67e22",       # Orange
    "success": "#27ae60",      # Green
    "warning": "#f39c12",      # Yellow
    "danger": "#c0392b",       # Red
    "neutral": "#7f8c8d",      # Gray
    "light": "#ecf0f1",        # Light gray
    "dark": "#2c3e50",         # Dark gray
}

# Chart style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#cccccc',
    'grid.color': '#e0e0e0',
    'grid.alpha': 0.5,
})

# Custom CSS for light professional theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3a5f;
    }
    
    /* Cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e8f4fd;
        border-left: 4px solid #3498db;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .warning-box {
        background-color: #fef9e7;
        border-left: 4px solid #f39c12;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .success-box {
        background-color: #eafaf1;
        border-left: 4px solid #27ae60;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .danger-box {
        background-color: #fdedec;
        border-left: 4px solid #c0392b;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Section headers */
    .section-header {
        background-color: #1e3a5f;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        margin: 20px 0 15px 0;
    }
    
    /* Tables */
    .dataframe {
        font-size: 12px;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def info_box(text, box_type="info"):
    """Display styled info box."""
    st.markdown(f'<div class="{box_type}-box">{text}</div>', unsafe_allow_html=True)

def section_header(text):
    """Display section header."""
    st.markdown(f'<div class="section-header"><strong>{text}</strong></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER SHEET INFO
# ═══════════════════════════════════════════════════════════════════════════════

def get_master_sheet_info(file_obj):
    """Extract customer information from Master Sheet."""
    try:
        ms = pd.read_excel(file_obj, sheet_name="Master Sheet", header=None)
        
        def safe_get(row, col):
            try:
                val = ms.iloc[row, col]
                return str(val).strip() if pd.notna(val) else None
            except Exception:
                return None
        
        # Auto-detect row offset
        row_offset = 0
        cell_0_6 = safe_get(0, 6)
        if cell_0_6 and not any(c.isdigit() for c in str(cell_0_6)):
            row_offset = 1
        
        def get(row, col):
            return safe_get(row + row_offset, col)
        
        info = {
            "account": get(0, 6),
            "customer_name": get(1, 6),
            "own_rent": get(2, 6),
            "community": get(3, 6),
            "address": get(4, 6),
            "city_town": get(5, 6),
            "gru_rep": get(6, 2),
            "survey_date": get(7, 2),
            "survey_time": get(8, 2),
            "results_sent_to": get(9, 2),
        }
        
        # Clean survey date
        if info["survey_date"] and "00:00:00" in str(info["survey_date"]):
            try:
                info["survey_date"] = pd.to_datetime(info["survey_date"]).strftime("%m/%d/%Y")
            except Exception:
                pass
        
        return info
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# METER LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class MeterLoader:
    """
    Load and clean meter consumption data.
    
    Expected Format:
    - Excel file with "Consumption" sheet (or sheet containing "consumption" in name)
    - Required columns: Division, MR Date, Days, Consumption
    - Optional columns: MR Reason, MR Type, MR Unit, Avg (daily average)
    """
    
    COLUMN_MAP = {
        "Division": "division", "division": "division",
        "Device": "device", "device": "device",
        "MR Reason": "mr_reason", "mr reason": "mr_reason", "Reason": "mr_reason",
        "MR Type": "mr_type", "mr type": "mr_type", "Type": "mr_type",
        "MR Date": "mr_date", "mr date": "mr_date", "Date": "mr_date", "Read Date": "mr_date",
        "Days": "days", "days": "days", "Billing Days": "days",
        "MR Result": "mr_result", "mr result": "mr_result",
        "MR Unit": "mr_unit", "mr unit": "mr_unit", "Unit": "mr_unit",
        "Consumption": "consumption", "consumption": "consumption", "Usage": "consumption",
        "Avg.": "avg_daily", "Avg": "avg_daily", "avg": "avg_daily", "Daily Avg": "avg_daily",
    }
    
    NON_READ_REASONS = {3}
    NON_READ_TYPES = {"automatic estimation"}
    
    def __init__(self, file_obj):
        self.file_obj = file_obj
        self.df = None
        self.has_mr_reason = False
        self.debug_info = {}
    
    def _find_sheet(self, xl):
        """Find the consumption data sheet."""
        self.debug_info["available_sheets"] = xl.sheet_names
        
        # Priority order for sheet selection
        for name in xl.sheet_names:
            name_lower = name.lower()
            if "consumption" in name_lower:
                return name
        
        # Fallback: look for sheets with data
        for name in xl.sheet_names:
            name_lower = name.lower()
            if name_lower not in ["master sheet", "info", "summary", "notes"]:
                # Check if sheet has data
                try:
                    test_df = pd.read_excel(xl, sheet_name=name, nrows=5)
                    if len(test_df.columns) >= 3:
                        return name
                except:
                    continue
        
        raise ValueError(
            f"No consumption sheet found.\n"
            f"Available sheets: {xl.sheet_names}\n"
            f"Expected: Sheet named 'Consumption' or containing 'consumption' in name."
        )
    
    def _find_header_row(self, xl, sheet):
        """
        Find the header row containing column names.
        Returns the row index where headers are found.
        """
        # Read first 10 rows to find headers
        preview = pd.read_excel(xl, sheet_name=sheet, header=None, nrows=10)
        
        for i in range(min(10, len(preview))):
            row_values = [str(x).strip() for x in preview.iloc[i].values if pd.notna(x)]
            row_str = " ".join(row_values).lower()
            
            # Check for key column names
            if "division" in row_str or ("date" in row_str and "consumption" in row_str):
                self.debug_info["header_row"] = i
                return i
        
        # Fallback: assume first row
        self.debug_info["header_row"] = 0
        return 0
    
    def load_and_clean(self):
        """Load and clean meter data with robust error handling."""
        try:
            xl = pd.ExcelFile(self.file_obj)
            sheet = self._find_sheet(xl)
            self.debug_info["selected_sheet"] = sheet
            
            header_row = self._find_header_row(xl, sheet)
            
            df = pd.read_excel(xl, sheet_name=sheet, header=header_row)
            df.columns = [str(c).strip() for c in df.columns]
            self.debug_info["original_columns"] = df.columns.tolist()
            
            # Map columns
            rename_map = {}
            for orig_col in df.columns:
                col_clean = orig_col.strip()
                if col_clean in self.COLUMN_MAP:
                    rename_map[orig_col] = self.COLUMN_MAP[col_clean]
            
            df.rename(columns=rename_map, inplace=True)
            self.debug_info["mapped_columns"] = df.columns.tolist()
            
            # Validate required columns
            required = ["mr_date", "consumption"]
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                # Try to find alternative column names
                suggestions = []
                for col in df.columns:
                    col_lower = col.lower()
                    if "date" in col_lower:
                        suggestions.append(f"'{col}' might be the date column")
                    if any(kw in col_lower for kw in ["usage", "kwh", "consumption", "reading"]):
                        suggestions.append(f"'{col}' might be the consumption column")
                
                raise ValueError(
                    f"Missing required columns: {missing}\n"
                    f"Columns found: {df.columns.tolist()}\n"
                    f"Suggestions:\n" + "\n".join(f"  - {s}" for s in suggestions) if suggestions else ""
                )
            
            # Parse dates
            if "mr_date" in df.columns:
                df["mr_date"] = pd.to_datetime(df["mr_date"], errors="coerce")
                invalid_dates = df["mr_date"].isna().sum()
                if invalid_dates > 0:
                    self.debug_info["invalid_dates"] = invalid_dates
            
            # Parse numeric columns
            for col in ["days", "consumption", "avg_daily"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Check for mr_reason column
            self.has_mr_reason = "mr_reason" in df.columns
            
            # Filter out non-reads
            original_count = len(df)
            
            if self.has_mr_reason:
                df = df[~df["mr_reason"].isin(self.NON_READ_REASONS)]
            
            if "mr_type" in df.columns:
                df["mr_type_lower"] = df["mr_type"].astype(str).str.lower()
                df = df[~df["mr_type_lower"].isin(self.NON_READ_TYPES)]
                df = df.drop(columns=["mr_type_lower"])
            
            # Drop rows with missing essential data
            df = df.dropna(subset=["mr_date", "consumption"])
            
            self.debug_info["rows_before_filter"] = original_count
            self.debug_info["rows_after_filter"] = len(df)
            
            if len(df) == 0:
                raise ValueError(
                    f"No valid data rows after cleaning.\n"
                    f"Original rows: {original_count}\n"
                    f"Check that 'MR Date' and 'Consumption' columns have valid data."
                )
            
            self.df = df.reset_index(drop=True)
            return self.df
            
        except Exception as e:
            if "Missing required columns" in str(e) or "No consumption sheet" in str(e):
                raise
            
            debug_str = "\n".join([f"  {k}: {v}" for k, v in self.debug_info.items()])
            raise ValueError(
                f"Meter Load Error: {str(e)}\n\n"
                f"Debug Info:\n{debug_str}\n\n"
                f"Expected format:\n"
                f"  - Excel file with 'Consumption' sheet\n"
                f"  - Required columns: Division, MR Date, Days, Consumption\n"
                f"  - Optional: Master Sheet with customer info"
            )
    
    def get_division(self, division):
        """Get data for a specific division (Electricity, Water, Gas)."""
        if self.df is None:
            return pd.DataFrame()
        
        if "division" not in self.df.columns:
            # If no division column, return all data
            return self.df.copy()
        
        mask = self.df["division"].str.lower().str.contains(division.lower(), na=False)
        return self.df[mask].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# METER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class MeterFeatures:
    """Compute features from meter data."""
    
    def __init__(self, df):
        self.df = df.copy().sort_values("mr_date").reset_index(drop=True)
    
    def compute_features(self):
        df = self.df
        
        avg_read_interval = df["days"].mean()
        total_consumption = df["consumption"].sum()
        total_days = df["days"].sum()
        overall_daily_avg = total_consumption / total_days if total_days > 0 else None
        peak_consumption = df["consumption"].max()
        base_consumption = df["consumption"].quantile(0.05)
        period_series = df.set_index("mr_date")["consumption"]
        rolling_avg = period_series.rolling(window=3, min_periods=1).mean()
        
        # Isolation Forest anomaly detection
        iso_cols = [c for c in ["consumption", "days", "avg_daily"] if c in df.columns]
        iso_data = df[iso_cols].dropna()
        df["anomaly"] = False
        
        if len(iso_data) >= 5:
            preds = IsolationForest(contamination=0.05, random_state=42).fit_predict(iso_data)
            df.loc[iso_data.index, "anomaly"] = (preds == -1)
        
        n_anomalies = int(df["anomaly"].sum())
        unit = df["mr_unit"].iloc[0] if "mr_unit" in df.columns else ""
        
        # Data quality score
        quality_score = self._compute_quality_score(df)
        
        return {
            "avg_read_interval": avg_read_interval,
            "total_consumption": total_consumption,
            "total_days": total_days,
            "overall_daily_avg": overall_daily_avg,
            "peak_consumption": peak_consumption,
            "base_consumption": base_consumption,
            "period_series": period_series,
            "rolling_avg": rolling_avg,
            "n_anomalies": n_anomalies,
            "unit": unit,
            "quality_score": quality_score,
            "df_with_anomalies": df,
        }
    
    def _compute_quality_score(self, df):
        score = 100
        
        # Missing values penalty
        missing_pct = df.isnull().mean().mean() * 100
        score -= min(missing_pct * 2, 20)
        
        # Irregular intervals penalty
        if "days" in df.columns:
            days_std = df["days"].std()
            if days_std > 10:
                score -= min(days_std, 20)
        
        # Too few readings penalty
        if len(df) < 12:
            score -= (12 - len(df)) * 2
        
        return max(0, min(100, round(score)))


# ═══════════════════════════════════════════════════════════════════════════════
# AMI LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class AMILoader:
    """
    Load AMI interval data with auto-format detection.
    
    Supported Formats:
    - Format A: Combined timestamp "Jan 15, 2025 - 2:30 PM EST" (values in Wh, divide by 1000)
    - Format B: Combined timestamp "01/12/2026 00:15 EST" (values in kWh)
    - Format C: Separate Date/Time columns with 12-hour time "12:00 am" (values as-is)
    - Format D: Simple datetime column with numeric values
    """
    
    SHEET_MAP = {
        "ELECTRIC": "Electric", "Electric": "Electric",
        "WATER": "Water", "Water": "Water", 
        "GAS": "Gas", "Gas": "Gas",
    }
    UNITS = {"Electric": "kWh", "Water": "Gal", "Gas": "CCF"}
    
    def __init__(self, file_obj):
        self.file_obj = file_obj
        self.df = None
        self.util_type = None
        self.unit = None
        self.format_detected = None
        self.debug_info = {}
    
    def _detect_utility_type(self, xl, filename=""):
        """Detect utility type from sheet names or filename."""
        filename_lower = str(filename).lower()
        
        # Check filename first
        if "water" in filename_lower:
            return "Water"
        elif "gas" in filename_lower:
            return "Gas"
        elif "electric" in filename_lower:
            return "Electric"
        
        # Check sheet names
        for name in xl.sheet_names:
            name_upper = name.upper()
            if "WATER" in name_upper:
                return "Water"
            elif "GAS" in name_upper:
                return "Gas"
            elif "ELECTRIC" in name_upper:
                return "Electric"
        
        return "Electric"  # Default
    
    def _find_header_row(self, preview):
        """Find the header row by looking for date/time keywords."""
        for i in range(min(10, len(preview))):
            row_values = [str(x).lower().strip() for x in preview.iloc[i].values if pd.notna(x)]
            row_str = " ".join(row_values)
            
            # Check for header keywords
            if any(kw in row_str for kw in ["date", "time", "timestamp", "datetime"]):
                return i
        
        return 0  # Default to first row
    
    def _identify_columns(self, df):
        """
        Identify date, time, and value columns.
        Returns: (date_col, time_col, value_col, format_type)
        """
        cols_lower = {c: c.lower().strip() for c in df.columns}
        
        date_col = None
        time_col = None
        value_col = None
        
        for col, col_lower in cols_lower.items():
            # Date column
            if col_lower in ["date", "datetime", "date/time", "reading date"]:
                date_col = col
            # Time column (separate)
            elif col_lower in ["time", "reading time"]:
                time_col = col
            # Combined timestamp
            elif "date" in col_lower and "time" in col_lower:
                date_col = col
            # Value columns
            elif col_lower in ["kwh", "kw", "value", "reading", "consumption", "usage", "gal", "gallons", "ccf"]:
                value_col = col
        
        # If no value column found, look for numeric column
        if value_col is None:
            for col in df.columns:
                if col not in [date_col, time_col]:
                    # Check if column has numeric data
                    try:
                        numeric_vals = pd.to_numeric(df[col], errors='coerce')
                        if numeric_vals.notna().sum() > len(df) * 0.5:
                            value_col = col
                            break
                    except:
                        continue
        
        # Determine format type
        if date_col and time_col:
            format_type = "separate_datetime"
        elif date_col:
            format_type = "combined_datetime"
        else:
            format_type = "unknown"
        
        return date_col, time_col, value_col, format_type
    
    def _parse_separate_datetime(self, df, date_col, time_col):
        """Parse separate date and time columns (Format C)."""
        timestamps = []
        
        for _, row in df.iterrows():
            try:
                date_val = row[date_col]
                time_val = str(row[time_col]).strip()
                
                # Handle date
                if isinstance(date_val, pd.Timestamp):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d")
                
                # Handle time (12-hour format with am/pm)
                time_clean = time_val.strip()
                
                # Parse 12-hour time
                try:
                    time_obj = pd.to_datetime(time_clean, format="%I:%M %p")
                    time_str = time_obj.strftime("%H:%M")
                except:
                    try:
                        time_obj = pd.to_datetime(time_clean, format="%I:%M%p")
                        time_str = time_obj.strftime("%H:%M")
                    except:
                        # Try 24-hour format
                        try:
                            time_obj = pd.to_datetime(time_clean, format="%H:%M")
                            time_str = time_obj.strftime("%H:%M")
                        except:
                            time_str = "00:00"
                
                full_ts = pd.to_datetime(f"{date_str} {time_str}")
                timestamps.append(full_ts)
            except Exception as e:
                timestamps.append(pd.NaT)
        
        return pd.Series(timestamps)
    
    def _parse_combined_datetime(self, df, ts_col):
        """Parse combined datetime column (Format A, B, D)."""
        first_val = str(df[ts_col].iloc[0]).strip()
        
        # Remove timezone suffixes
        def clean_tz(x):
            s = str(x).strip()
            for tz in [" EST", " EDT", " CST", " CDT", " PST", " PDT"]:
                s = s.replace(tz, "")
            return s.strip()
        
        # Detect format
        if " - " in first_val:
            # Format A: "Jan 15, 2025 - 2:30 PM EST"
            self.format_detected = "A"
            self.debug_info["multiplier"] = 0.001
            
            def parse_a(x):
                try:
                    s = clean_tz(x).replace(" - ", " ")
                    return pd.to_datetime(s)
                except:
                    return pd.NaT
            
            return df[ts_col].apply(parse_a), 0.001
        
        elif "/" in first_val and len(first_val.split()) >= 2:
            # Format B: "01/12/2026 00:15 EST"
            self.format_detected = "B"
            self.debug_info["multiplier"] = 1.0
            
            def parse_b(x):
                try:
                    s = clean_tz(x)
                    return pd.to_datetime(s, format="%m/%d/%Y %H:%M")
                except:
                    try:
                        return pd.to_datetime(s)
                    except:
                        return pd.NaT
            
            return df[ts_col].apply(parse_b), 1.0
        
        else:
            # Format D: General datetime
            self.format_detected = "D"
            self.debug_info["multiplier"] = 1.0
            
            return pd.to_datetime(df[ts_col], errors="coerce"), 1.0
    
    def load(self):
        """Load and parse AMI data with auto-format detection."""
        try:
            # Get filename for utility detection
            filename = getattr(self.file_obj, 'name', '')
            
            xl = pd.ExcelFile(self.file_obj)
            self.debug_info["sheets"] = xl.sheet_names
            
            # Select sheet
            sheet = xl.sheet_names[0]
            for name in xl.sheet_names:
                if name.upper() in ["ELECTRIC", "WATER", "GAS"]:
                    sheet = name
                    break
            
            self.debug_info["selected_sheet"] = sheet
            
            # Detect utility type
            self.util_type = self._detect_utility_type(xl, filename)
            self.unit = self.UNITS.get(self.util_type, "kWh")
            self.debug_info["util_type"] = self.util_type
            
            # Read preview to find header
            preview = pd.read_excel(xl, sheet_name=sheet, header=None, nrows=15)
            header_row = self._find_header_row(preview)
            self.debug_info["header_row"] = header_row
            
            # Read data
            df = pd.read_excel(xl, sheet_name=sheet, header=header_row)
            df.columns = [str(c).strip() for c in df.columns]
            self.debug_info["columns"] = df.columns.tolist()
            self.debug_info["row_count"] = len(df)
            
            if len(df) == 0:
                raise ValueError(
                    f"No data found after header row {header_row}.\n"
                    f"Columns detected: {df.columns.tolist()}\n"
                    f"Please check that data starts after row {header_row + 1} in Excel."
                )
            
            # Identify columns
            date_col, time_col, value_col, format_type = self._identify_columns(df)
            self.debug_info["date_col"] = date_col
            self.debug_info["time_col"] = time_col
            self.debug_info["value_col"] = value_col
            self.debug_info["format_type"] = format_type
            
            if date_col is None:
                raise ValueError(
                    f"Could not find date/time column.\n"
                    f"Columns found: {df.columns.tolist()}\n"
                    f"Expected: 'Date', 'Time', 'DateTime', or similar.\n"
                    f"First row sample: {df.iloc[0].tolist() if len(df) > 0 else 'N/A'}"
                )
            
            if value_col is None:
                raise ValueError(
                    f"Could not find value column.\n"
                    f"Columns found: {df.columns.tolist()}\n"
                    f"Expected: 'kWh', 'Value', 'Consumption', 'Reading', or numeric column."
                )
            
            # Parse timestamps based on format
            multiplier = 1.0
            
            if format_type == "separate_datetime":
                # Format C: Separate Date and Time columns
                self.format_detected = "C"
                df["timestamp"] = self._parse_separate_datetime(df, date_col, time_col)
                multiplier = 1.0
            else:
                # Combined datetime column
                df["timestamp"], multiplier = self._parse_combined_datetime(df, date_col)
            
            # Parse values
            df["kwh"] = pd.to_numeric(df[value_col], errors="coerce") * multiplier
            
            # Clean up
            df = df[["timestamp", "kwh"]].dropna()
            
            if len(df) == 0:
                raise ValueError(
                    f"No valid data after parsing.\n"
                    f"Date column '{date_col}' sample: {preview.iloc[header_row+1:header_row+3, :].values.tolist()}\n"
                    f"Check date format is parseable."
                )
            
            # Sort chronologically
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            self.df = df
            self.debug_info["final_row_count"] = len(df)
            self.debug_info["date_range"] = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
            
            return df
            
        except Exception as e:
            error_msg = str(e)
            debug_str = "\n".join([f"  {k}: {v}" for k, v in self.debug_info.items()])
            raise ValueError(
                f"AMI Load Error: {error_msg}\n\n"
                f"Debug Info:\n{debug_str}\n\n"
                f"Expected file formats:\n"
                f"  Format A: Combined column 'Jan 15, 2025 - 2:30 PM EST' + value in Wh\n"
                f"  Format B: Combined column '01/12/2026 00:15 EST' + value in kWh\n"
                f"  Format C: Separate 'Date' and 'Time' columns (e.g., '2026-03-11', '12:00 am')\n"
                f"  Format D: Standard datetime column + numeric value column"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# AMI FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class AMIFeatures:
    """Compute features from AMI interval data."""
    
    def __init__(self, df, unit="kWh"):
        self.df = df.copy()
        self.unit = unit
    
    def compute(self):
        df = self.df.sort_values("timestamp")
        
        deltas = df["timestamp"].diff().dropna()
        interval = deltas.mode()[0]
        interval_minutes = int(interval.total_seconds() / 60)
        
        base_load = df["kwh"].quantile(0.05)
        base_load_rate = base_load / (interval_minutes / 60)  # per hour rate
        peak_val = df["kwh"].max()
        peak_rate = peak_val / (interval_minutes / 60)
        
        df["date"] = df["timestamp"].dt.date
        daily_series = df.groupby("date")["kwh"].sum()
        daily_avg = daily_series.mean()
        peak_day = pd.Timestamp(daily_series.idxmax())
        
        df["hour"] = df["timestamp"].dt.hour
        avg_by_hour = df.groupby("hour")["kwh"].mean()
        
        # Load factor (meaningful for electricity)
        total_val = df["kwh"].sum()
        hours = len(df) * interval_minutes / 60
        avg_demand = total_val / hours if hours > 0 else 0
        load_factor = avg_demand / peak_rate if peak_rate > 0 else 0
        
        # Time-of-use ratio (peak 2-7pm vs off-peak)
        peak_hours = df[df["hour"].between(14, 19)]["kwh"].mean()
        off_peak = df[~df["hour"].between(14, 19)]["kwh"].mean()
        tou_ratio = peak_hours / off_peak if off_peak > 0 else 1
        
        return {
            "interval_minutes": interval_minutes,
            "base_load": base_load,
            "base_load_kw": base_load_rate,  # Kept for backward compatibility
            "base_load_rate": base_load_rate,
            "peak_kwh": peak_val,  # Kept for backward compatibility
            "peak_val": peak_val,
            "peak_kw": peak_rate,  # Kept for backward compatibility
            "peak_rate": peak_rate,
            "daily_avg_kwh": daily_avg,  # Kept for backward compatibility
            "daily_avg": daily_avg,
            "daily_series": daily_series,
            "peak_day": peak_day,
            "avg_by_hour": avg_by_hour,
            "load_factor": load_factor,
            "tou_ratio": tou_ratio,
            "df": df,
            "unit": self.unit,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

GAINESVILLE_LAT = 29.6516
GAINESVILLE_LON = -82.3248

@st.cache_data(ttl=3600)
def get_temperature_data(start_date, end_date, lat=GAINESVILLE_LAT, lon=GAINESVILLE_LON):
    """Fetch daily temperature data from Open-Meteo API."""
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "temperature_unit": "fahrenheit",
        "timezone": "America/New_York",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()["daily"]
        
        df_temp = pd.DataFrame({
            "date": pd.to_datetime(data["time"]),
            "temp_max": data["temperature_2m_max"],
            "temp_min": data["temperature_2m_min"],
        })
        df_temp["temp_avg"] = (df_temp["temp_max"] + df_temp["temp_min"]) / 2
        df_temp = df_temp.set_index("date")
        
        return df_temp
    except Exception as e:
        st.warning(f"Could not fetch temperature data: {e}")
        return None


def compute_degree_days(temp_avg, base=65):
    """Compute Heating and Cooling Degree Days."""
    hdd = max(0, base - temp_avg)
    cdd = max(0, temp_avg - base)
    return hdd, cdd


# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class WeatherAnomalyDetector:
    """Weather-normalized anomaly detection using regression."""
    
    def __init__(self, df_meter, df_temp, comfort_base=65, z_threshold=2.5):
        self.df_meter = df_meter.copy()
        self.df_temp = df_temp
        self.comfort_base = comfort_base
        self.z_threshold = z_threshold
        self.model = None
        self.results = None
    
    def run(self):
        df = self.df_meter.sort_values("mr_date").reset_index(drop=True)
        
        # Compute degree days for each billing period
        records = []
        for _, row in df.iterrows():
            end_date = row["mr_date"]
            days = int(row["days"]) if pd.notna(row["days"]) else 30
            start_date = end_date - pd.Timedelta(days=days)
            
            # Get temps for this period
            mask = (self.df_temp.index > start_date) & (self.df_temp.index <= end_date)
            period_temps = self.df_temp.loc[mask, "temp_avg"]
            
            if len(period_temps) == 0:
                continue
            
            hdd = sum(max(0, self.comfort_base - t) for t in period_temps)
            cdd = sum(max(0, t - self.comfort_base) for t in period_temps)
            
            records.append({
                "mr_date": end_date,
                "consumption": row["consumption"],
                "days": days,
                "daily_avg": row["consumption"] / days if days > 0 else 0,
                "hdd": hdd,
                "cdd": cdd,
                "temp_avg": period_temps.mean(),
            })
        
        if len(records) < 6:
            return None
        
        df_analysis = pd.DataFrame(records)
        
        # Fit regression model
        X = df_analysis[["hdd", "cdd"]].values
        y = df_analysis["daily_avg"].values
        
        try:
            self.model = HuberRegressor().fit(X, y)
            df_analysis["predicted"] = self.model.predict(X)
            df_analysis["residual"] = df_analysis["daily_avg"] - df_analysis["predicted"]
            
            residual_std = df_analysis["residual"].std()
            df_analysis["residual_z"] = df_analysis["residual"] / residual_std if residual_std > 0 else 0
            
            df_analysis["anomaly_high"] = df_analysis["residual_z"] > self.z_threshold
            df_analysis["anomaly_low"] = df_analysis["residual_z"] < -self.z_threshold
            df_analysis["anomaly"] = df_analysis["anomaly_high"] | df_analysis["anomaly_low"]
            
            # Persistent anomalies (2+ consecutive)
            df_analysis["persistent_high"] = (
                df_analysis["anomaly_high"] & 
                df_analysis["anomaly_high"].shift(1).fillna(False)
            )
            df_analysis["persistent_low"] = (
                df_analysis["anomaly_low"] & 
                df_analysis["anomaly_low"].shift(1).fillna(False)
            )
            
            self.results = df_analysis
            return df_analysis
            
        except Exception as e:
            st.warning(f"Regression model failed: {e}")
            return None
    
    def get_interpretation(self):
        """Generate auditor interpretation of results."""
        if self.results is None:
            return None
        
        df = self.results
        interpretations = []
        
        # Model fit
        r2 = 1 - (df["residual"].var() / df["daily_avg"].var()) if df["daily_avg"].var() > 0 else 0
        
        if r2 > 0.7:
            interpretations.append({
                "type": "success",
                "title": "Strong Weather Correlation",
                "text": f"Weather explains {r2:.0%} of consumption variance. Building responds predictably to temperature changes."
            })
        elif r2 > 0.4:
            interpretations.append({
                "type": "info",
                "title": "Moderate Weather Correlation",
                "text": f"Weather explains {r2:.0%} of consumption variance. Other factors (occupancy, equipment) also significant."
            })
        else:
            interpretations.append({
                "type": "warning",
                "title": "Weak Weather Correlation",
                "text": f"Weather explains only {r2:.0%} of consumption variance. Usage driven primarily by non-weather factors."
            })
        
        # Anomalies
        n_high = df["anomaly_high"].sum()
        n_low = df["anomaly_low"].sum()
        n_persistent_high = df["persistent_high"].sum()
        
        if n_persistent_high > 0:
            interpretations.append({
                "type": "danger",
                "title": "Persistent High Usage Detected",
                "text": f"{n_persistent_high} consecutive periods with unexpectedly high consumption. Investigate potential issues: HVAC inefficiency, air leaks, equipment malfunction, or occupancy changes."
            })
        elif n_high > 0:
            interpretations.append({
                "type": "warning",
                "title": "Occasional High Usage",
                "text": f"{n_high} periods with higher-than-expected consumption. May indicate sporadic issues or temporary changes."
            })
        
        if n_low > 0:
            interpretations.append({
                "type": "info",
                "title": "Low Usage Periods",
                "text": f"{n_low} periods with lower-than-expected consumption. May indicate vacancy, conservation efforts, or equipment issues."
            })
        
        if n_high == 0 and n_low == 0:
            interpretations.append({
                "type": "success",
                "title": "Consistent Performance",
                "text": "No significant anomalies detected. Consumption patterns are consistent with weather expectations."
            })
        
        return interpretations


# ═══════════════════════════════════════════════════════════════════════════════
# FRACTAL ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class FractalAnalyzer:
    """
    Fractal analysis for energy consumption time series.
    Based on Knowles et al. (2017) - University of Florida.
    
    Implements Detrended Fluctuation Analysis (DFA) for Hurst exponent.
    """
    
    def __init__(self, df_ami, value_col="kwh", time_col="timestamp"):
        self.df = df_ami.copy().sort_values(time_col).reset_index(drop=True)
        self.values = self.df[value_col].values
        self.timestamps = self.df[time_col]
        self.differenced = np.diff(self.values)
    
    def detrended_fluctuation_analysis(self, min_window=4, max_window=None, num_windows=20):
        """
        Perform Detrended Fluctuation Analysis (DFA).
        
        Returns Hurst exponent (H):
        - H < 0.5: Anti-persistent (mean-reverting)
        - H = 0.5: Random walk (uncorrelated)
        - H > 0.5: Persistent (trending)
        """
        data = self.differenced
        N = len(data)
        
        if max_window is None:
            max_window = N // 4
        
        window_sizes = np.unique(np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num_windows
        ).astype(int))
        
        fluctuations = []
        
        for n in window_sizes:
            num_segments = N // n
            if num_segments < 2:
                continue
            
            rms_values = []
            for i in range(num_segments):
                segment = data[i*n:(i+1)*n]
                y = np.cumsum(segment - np.mean(segment))
                x = np.arange(n)
                coeffs = np.polyfit(x, y, 1)
                trend = np.polyval(coeffs, x)
                rms = np.sqrt(np.mean((y - trend) ** 2))
                if rms > 0:
                    rms_values.append(rms)
            
            if rms_values:
                fluctuations.append((n, np.mean(rms_values)))
        
        if len(fluctuations) < 3:
            return {"error": "Insufficient data for DFA"}
        
        windows = np.array([f[0] for f in fluctuations])
        F_n = np.array([f[1] for f in fluctuations])
        
        valid = F_n > 0
        if valid.sum() < 3:
            return {"error": "Insufficient valid fluctuations"}
        
        log_n = np.log(windows[valid])
        log_F = np.log(F_n[valid])
        
        slope, intercept, r_value, p_value, _ = stats.linregress(log_n, log_F)
        
        return {
            "hurst_exponent": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "window_sizes": windows[valid],
            "fluctuations": F_n[valid],
            "log_n": log_n,
            "log_F": log_F,
            "slope": slope,
            "intercept": intercept,
        }
    
    def distribution_analysis(self):
        """Analyze distribution characteristics."""
        raw = self.values
        diff = self.differenced
        
        raw_stats = {
            "mean": np.mean(raw),
            "std": np.std(raw),
            "skewness": stats.skew(raw),
            "kurtosis": stats.kurtosis(raw),
        }
        
        diff_stats = {
            "mean": np.mean(diff),
            "std": np.std(diff),
            "skewness": stats.skew(diff),
            "kurtosis": stats.kurtosis(diff),
        }
        
        ks_raw = stats.kstest(raw, 'norm', args=(np.mean(raw), np.std(raw)))
        ks_diff = stats.kstest(diff, 'norm', args=(np.mean(diff), np.std(diff)))
        
        return {
            "raw": raw_stats,
            "differenced": diff_stats,
            "normality_tests": {
                "raw_ks_pvalue": ks_raw.pvalue,
                "raw_is_normal": ks_raw.pvalue > 0.05,
                "diff_ks_pvalue": ks_diff.pvalue,
                "diff_is_normal": ks_diff.pvalue > 0.05,
            }
        }
    
    def get_interpretation(self, dfa_result, dist_result):
        """Generate auditor interpretation."""
        if "error" in dfa_result:
            return [{"type": "warning", "title": "Analysis Error", "text": dfa_result["error"]}]
        
        H = dfa_result["hurst_exponent"]
        K = dist_result["raw"]["kurtosis"]
        interpretations = []
        
        # Hurst interpretation
        if H < 0.35:
            interpretations.append({
                "type": "info",
                "title": f"Strongly Anti-Persistent Pattern (H = {H:.3f})",
                "text": "Usage strongly mean-reverts. High consumption periods are quickly followed by low periods. This suggests highly variable occupant behavior with frequent manual adjustments."
            })
        elif H < 0.45:
            interpretations.append({
                "type": "info", 
                "title": f"Anti-Persistent Pattern (H = {H:.3f})",
                "text": "Usage tends to bounce back toward average. Indicates variable scheduling or reactive behavior patterns."
            })
        elif H < 0.55:
            interpretations.append({
                "type": "info",
                "title": f"Random/Mixed Pattern (H = {H:.3f})",
                "text": "Usage pattern shows no strong persistence or anti-persistence. Mix of predictable and unpredictable factors."
            })
        elif H < 0.65:
            interpretations.append({
                "type": "success",
                "title": f"Weakly Persistent Pattern (H = {H:.3f})",
                "text": "Usage shows mild trending behavior. Relatively consistent patterns that are easier to predict."
            })
        else:
            interpretations.append({
                "type": "success",
                "title": f"Strongly Persistent Pattern (H = {H:.3f})",
                "text": "Usage trends continue over time. Indicates consistent, automated, or predictable consumption patterns. Easier to baseline and model."
            })
        
        # Kurtosis interpretation
        if K > 5:
            interpretations.append({
                "type": "warning",
                "title": f"High Kurtosis ({K:.1f})",
                "text": "Frequent extreme usage events (spikes/drops). Check for large cycling equipment: HVAC, electric water heater, EV charging. Load shifting opportunities may exist."
            })
        
        # Recommendations
        if H < 0.4:
            interpretations.append({
                "type": "info",
                "title": "Recommended Approach",
                "text": "Focus on behavior-based interventions. Consider programmable thermostat, occupant education, and scheduling optimization. Pre/post comparisons will be challenging due to high variability."
            })
        elif H > 0.6:
            interpretations.append({
                "type": "info",
                "title": "Recommended Approach", 
                "text": "Focus on equipment-based interventions. Building shows predictable patterns suitable for efficiency upgrades. Pre/post comparisons will be reliable."
            })
        
        return interpretations


# ═══════════════════════════════════════════════════════════════════════════════
# CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_meter_consumption_chart(df, title="Consumption History"):
    """Create meter consumption bar chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = [COLORS["danger"] if row["anomaly"] else COLORS["secondary"] 
              for _, row in df.iterrows()]
    
    ax.bar(df["mr_date"], df["consumption"], color=colors, alpha=0.8, width=20)
    
    # Add trend line
    z = np.polyfit(range(len(df)), df["consumption"], 1)
    trend = np.poly1d(z)(range(len(df)))
    ax.plot(df["mr_date"], trend, color=COLORS["accent"], linestyle="--", 
            linewidth=2, label="Trend")
    
    ax.set_xlabel("Meter Read Date")
    ax.set_ylabel(df["mr_unit"].iloc[0] if "mr_unit" in df.columns else "Consumption")
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.legend()
    
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def create_daily_average_chart(df, title="Daily Average Consumption"):
    """Create daily average line chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if "avg_daily" in df.columns:
        ax.plot(df["mr_date"], df["avg_daily"], color=COLORS["primary"], 
                linewidth=2, marker="o", markersize=6)
        ax.fill_between(df["mr_date"], df["avg_daily"], alpha=0.3, color=COLORS["secondary"])
    
    ax.set_xlabel("Meter Read Date")
    ax.set_ylabel("Daily Average")
    ax.set_title(title, fontweight="bold", fontsize=14)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def create_ami_load_shape(df, title="Load Shape", unit="kWh"):
    """Create AMI load shape chart."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(df["timestamp"], df["kwh"], color=COLORS["secondary"], 
            linewidth=0.5, alpha=0.8)
    ax.fill_between(df["timestamp"], df["kwh"], alpha=0.3, color=COLORS["secondary"])
    
    ax.set_xlabel("Date/Time")
    ax.set_ylabel(f"{unit} per Interval")
    ax.set_title(title, fontweight="bold", fontsize=14)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def create_hourly_profile(avg_by_hour, title="Average Hourly Profile", unit="kWh"):
    """Create hourly profile chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    hours = avg_by_hour.index
    values = avg_by_hour.values
    
    # Color by time of day
    colors = []
    for h in hours:
        if 6 <= h < 9 or 17 <= h < 21:
            colors.append(COLORS["accent"])  # Peak
        elif 9 <= h < 17:
            colors.append(COLORS["secondary"])  # Day
        else:
            colors.append(COLORS["primary"])  # Night
    
    ax.bar(hours, values, color=colors, alpha=0.8, edgecolor="white")
    
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(f"Average {unit}")
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["primary"], label="Night (9PM-6AM)"),
        Patch(facecolor=COLORS["secondary"], label="Day (9AM-5PM)"),
        Patch(facecolor=COLORS["accent"], label="Peak (6-9AM, 5-9PM)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    plt.tight_layout()
    return fig


def create_temperature_overlay(df_merged, title="Consumption vs Temperature"):
    """Create temperature overlay chart."""
    fig, ax1 = plt.subplots(figsize=(14, 5))
    
    # Temperature-based coloring
    colors = []
    for t in df_merged["temp_avg"]:
        if t >= 80:
            colors.append(COLORS["danger"])
        elif t <= 55:
            colors.append(COLORS["secondary"])
        else:
            colors.append(COLORS["success"])
    
    ax1.bar(df_merged["date"], df_merged["kwh"], color=colors, alpha=0.7, width=0.8)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily kWh", color=COLORS["primary"])
    ax1.tick_params(axis="y", labelcolor=COLORS["primary"])
    
    # Temperature line
    ax2 = ax1.twinx()
    ax2.plot(df_merged["date"], df_merged["temp_avg"], color=COLORS["accent"], 
             linewidth=2.5, marker="o", markersize=4)
    ax2.axhline(65, color=COLORS["accent"], linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Temperature (F)", color=COLORS["accent"])
    ax2.tick_params(axis="y", labelcolor=COLORS["accent"])
    
    ax1.set_title(title, fontweight="bold", fontsize=14)
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS["danger"], alpha=0.7, label="Hot (>80F)"),
        Patch(facecolor=COLORS["success"], alpha=0.7, label="Mild (55-80F)"),
        Patch(facecolor=COLORS["secondary"], alpha=0.7, label="Cold (<55F)"),
        Line2D([0], [0], color=COLORS["accent"], linewidth=2.5, marker="o", label="Temperature"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8)
    
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def create_weather_anomaly_chart(df_analysis, title="Weather-Normalized Analysis"):
    """Create weather anomaly detection chart."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: Actual vs Predicted
    ax1 = axes[0]
    ax1.plot(df_analysis["mr_date"], df_analysis["daily_avg"], 
             color=COLORS["primary"], linewidth=2, marker="o", label="Actual")
    ax1.plot(df_analysis["mr_date"], df_analysis["predicted"],
             color=COLORS["success"], linewidth=2, linestyle="--", label="Expected")
    ax1.fill_between(df_analysis["mr_date"], 
                     df_analysis["predicted"] - df_analysis["residual"].std() * 2,
                     df_analysis["predicted"] + df_analysis["residual"].std() * 2,
                     alpha=0.2, color=COLORS["success"], label="Normal Range")
    
    ax1.set_ylabel("Daily Average (kWh/day)")
    ax1.set_title(title, fontweight="bold", fontsize=14)
    ax1.legend(loc="upper right")
    
    # Bottom: Residual Z-scores
    ax2 = axes[1]
    colors = [COLORS["danger"] if z > 2.5 else COLORS["secondary"] if z < -2.5 
              else COLORS["neutral"] for z in df_analysis["residual_z"]]
    ax2.bar(df_analysis["mr_date"], df_analysis["residual_z"], color=colors, alpha=0.8)
    ax2.axhline(2.5, color=COLORS["danger"], linestyle="--", linewidth=1.5, label="High Threshold")
    ax2.axhline(-2.5, color=COLORS["secondary"], linestyle="--", linewidth=1.5, label="Low Threshold")
    ax2.axhline(0, color=COLORS["dark"], linewidth=1)
    
    ax2.set_xlabel("Billing Period")
    ax2.set_ylabel("Z-Score")
    ax2.legend(loc="upper right")
    
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def create_dfa_chart(dfa_result, title="Detrended Fluctuation Analysis"):
    """Create DFA scaling plot."""
    if "error" in dfa_result:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, dfa_result["error"], ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scaling plot
    ax1 = axes[0]
    ax1.scatter(dfa_result["log_n"], dfa_result["log_F"], 
                color=COLORS["secondary"], s=60, alpha=0.8, zorder=3)
    
    x_line = np.linspace(dfa_result["log_n"].min(), dfa_result["log_n"].max(), 100)
    y_line = dfa_result["slope"] * x_line + dfa_result["intercept"]
    ax1.plot(x_line, y_line, color=COLORS["danger"], linewidth=2, linestyle="--",
             label=f"H = {dfa_result['hurst_exponent']:.3f} (R2 = {dfa_result['r_squared']:.3f})")
    
    ax1.set_xlabel("log(Window Size)")
    ax1.set_ylabel("log(Fluctuation)")
    ax1.set_title("DFA Scaling Plot", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Hurst interpretation
    ax2 = axes[1]
    H = dfa_result["hurst_exponent"]
    
    h_range = np.linspace(0, 1, 100)
    for i, h in enumerate(h_range[:-1]):
        color = COLORS["secondary"] if h < 0.5 else COLORS["danger"]
        ax2.axvspan(h, h_range[i+1], color=color, alpha=0.2)
    
    ax2.axvline(0.5, color=COLORS["dark"], linewidth=2, linestyle="-", label="Random (H=0.5)")
    ax2.axvline(H, color=COLORS["primary"], linewidth=3, label=f"This Building (H={H:.3f})")
    
    ax2.text(0.25, 0.8, "Anti-Persistent\n(Variable)", ha="center", va="center",
             fontsize=10, transform=ax2.transAxes, color=COLORS["secondary"], fontweight="bold")
    ax2.text(0.75, 0.8, "Persistent\n(Consistent)", ha="center", va="center",
             fontsize=10, transform=ax2.transAxes, color=COLORS["danger"], fontweight="bold")
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Hurst Exponent (H)")
    ax2.set_title("Complexity Classification", fontweight="bold")
    ax2.legend(loc="lower right")
    ax2.set_yticks([])
    
    plt.tight_layout()
    return fig


def create_distribution_chart(values, diff, title="Distribution Analysis"):
    """Create distribution analysis charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Raw histogram
    ax1 = axes[0]
    ax1.hist(values, bins=50, color=COLORS["secondary"], alpha=0.7, edgecolor="white", density=True)
    x = np.linspace(values.min(), values.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, np.mean(values), np.std(values)), 
             color=COLORS["danger"], linewidth=2, linestyle="--", label="Normal fit")
    ax1.set_title("Raw Data Distribution", fontweight="bold")
    ax1.set_xlabel("kWh per interval")
    ax1.set_ylabel("Density")
    ax1.legend()
    
    # Differenced histogram
    ax2 = axes[1]
    ax2.hist(diff, bins=50, color=COLORS["success"], alpha=0.7, edgecolor="white", density=True)
    x_d = np.linspace(diff.min(), diff.max(), 100)
    ax2.plot(x_d, stats.norm.pdf(x_d, np.mean(diff), np.std(diff)),
             color=COLORS["danger"], linewidth=2, linestyle="--", label="Normal fit")
    ax2.set_title("Differenced Data", fontweight="bold")
    ax2.set_xlabel("Change between intervals")
    ax2.set_ylabel("Density")
    ax2.legend()
    
    # Q-Q plot
    ax3 = axes[2]
    stats.probplot(diff, dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot (Differenced)", fontweight="bold")
    ax3.get_lines()[0].set_markerfacecolor(COLORS["secondary"])
    ax3.get_lines()[0].set_markersize(4)
    ax3.get_lines()[1].set_color(COLORS["danger"])
    
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(customer_info, meter_data, ami_data, temp_data, 
                        fractal_data=None, report_type="standard"):
    """Generate PDF report."""
    
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        
        ax.text(0.5, 0.8, "Energy Audit Analyzer", fontsize=28, fontweight="bold",
                ha="center", va="center", color=COLORS["primary"])
        ax.text(0.5, 0.7, "Consumption Analysis Report", fontsize=18,
                ha="center", va="center", color=COLORS["dark"])
        
        if customer_info:
            ax.text(0.5, 0.5, f"Customer: {customer_info.get('customer_name', 'N/A')}", 
                    fontsize=14, ha="center", va="center")
            ax.text(0.5, 0.45, f"Account: {customer_info.get('account', 'N/A')}", 
                    fontsize=12, ha="center", va="center")
            ax.text(0.5, 0.4, f"Address: {customer_info.get('address', 'N/A')}", 
                    fontsize=12, ha="center", va="center")
        
        report_label = "Standard Report (Weather Analysis)" if report_type == "standard" else "Advanced Report (Fractal Analysis)"
        ax.text(0.5, 0.25, report_label, fontsize=12, ha="center", va="center",
                style="italic", color=COLORS["neutral"])
        ax.text(0.5, 0.15, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                fontsize=10, ha="center", va="center", color=COLORS["neutral"])
        
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        
        # Meter charts
        if meter_data is not None:
            for division, df in meter_data.items():
                if not df.empty:
                    # Consumption chart
                    fig = create_meter_consumption_chart(df, f"{division} - Consumption History")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                    
                    # Daily average
                    fig = create_daily_average_chart(df, f"{division} - Daily Average")
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
        
        # AMI charts
        if ami_data is not None:
            df_ami = ami_data["df"]
            feats = ami_data["features"]
            
            fig = create_ami_load_shape(df_ami, "AMI Load Shape")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            
            fig = create_hourly_profile(feats["avg_by_hour"], "Hourly Profile")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        
        # Temperature charts
        if temp_data is not None and "merged" in temp_data:
            fig = create_temperature_overlay(temp_data["merged"], "Temperature Correlation")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            
            if "anomaly_df" in temp_data and temp_data["anomaly_df"] is not None:
                fig = create_weather_anomaly_chart(temp_data["anomaly_df"], "Weather-Normalized Analysis")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        
        # Advanced (fractal) analysis
        if report_type == "advanced" and fractal_data is not None:
            fig = create_dfa_chart(fractal_data["dfa"], "Fractal Analysis - DFA")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            
            fig = create_distribution_chart(
                fractal_data["values"], 
                fractal_data["differenced"],
                "Distribution Analysis"
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        
        # Summary page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        
        ax.text(0.5, 0.95, "Analysis Summary", fontsize=20, fontweight="bold",
                ha="center", va="top", color=COLORS["primary"])
        
        y_pos = 0.85
        summary_items = []
        
        if meter_data:
            for div, df in meter_data.items():
                if not df.empty:
                    feats = MeterFeatures(df).compute_features()
                    summary_items.append(f"{div}:")
                    summary_items.append(f"  - Total: {feats['total_consumption']:,.0f} {feats['unit']}")
                    summary_items.append(f"  - Daily Avg: {feats['overall_daily_avg']:.1f} {feats['unit']}/day")
                    summary_items.append(f"  - Anomalies: {feats['n_anomalies']}")
                    summary_items.append("")
        
        if ami_data:
            feats = ami_data["features"]
            summary_items.append("AMI Analysis:")
            summary_items.append(f"  - Interval: {feats['interval_minutes']} minutes")
            summary_items.append(f"  - Daily Avg: {feats['daily_avg_kwh']:.1f} kWh")
            summary_items.append(f"  - Peak Demand: {feats['peak_kw']:.2f} kW")
            summary_items.append(f"  - Load Factor: {feats['load_factor']:.1%}")
            summary_items.append("")
        
        if fractal_data and "dfa" in fractal_data and "hurst_exponent" in fractal_data["dfa"]:
            H = fractal_data["dfa"]["hurst_exponent"]
            summary_items.append("Complexity Analysis:")
            summary_items.append(f"  - Hurst Exponent: {H:.3f}")
            if H < 0.45:
                summary_items.append("  - Pattern: Anti-persistent (variable)")
            elif H > 0.55:
                summary_items.append("  - Pattern: Persistent (consistent)")
            else:
                summary_items.append("  - Pattern: Random/mixed")
        
        for item in summary_items:
            ax.text(0.1, y_pos, item, fontsize=11, va="top", 
                   fontfamily="monospace" if item.startswith("  ") else "sans-serif")
            y_pos -= 0.04
        
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    
    buffer.seek(0)
    return buffer


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.title("Energy Audit Analyzer")
    st.markdown("*Professional energy consumption analysis for auditors*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Data Upload")
        
        st.subheader("Meter Data")
        meter_file = st.file_uploader(
            "Upload meter reading file (Excel)",
            type=["xlsx", "xls"],
            key="meter"
        )
        
        st.subheader("AMI Data")
        ami_file = st.file_uploader(
            "Upload AMI interval data (Excel)",
            type=["xlsx", "xls"],
            key="ami"
        )
        
        st.markdown("---")
        st.subheader("Settings")
        z_threshold = st.slider("Anomaly Z-Threshold", 1.5, 4.0, 2.5, 0.1)
        comfort_base = st.slider("Comfort Baseline (F)", 60, 72, 65, 1)
    
    # Check if any data uploaded
    if meter_file is None and ami_file is None:
        st.info("Upload meter data and/or AMI data to begin analysis.")
        
        # Show instructions
        with st.expander("How to Use This Tool"):
            st.markdown("""
            **1. Upload Your Data**
            - **Meter File**: Excel file with billing/consumption history (must have 'Consumption' sheet)
            - **AMI File**: Excel file with interval data (15-min or hourly readings)
            - You can upload one or both files
            
            **2. Review Analysis**
            - Navigate through tabs to see different analyses
            - Weather data is automatically fetched for Gainesville, FL
            
            **3. Export Reports**
            - **Standard Report**: Includes consumption charts and weather analysis
            - **Advanced Report**: Adds fractal complexity analysis
            
            **Understanding the Analysis**
            - **Weather Normalization**: Compares actual usage to weather-expected usage
            - **Anomaly Detection**: Flags periods with unusually high or low consumption
            - **Fractal Analysis**: Reveals complexity patterns in energy use behavior
            """)
        return
    
    # Initialize data containers
    customer_info = None
    meter_data = {}
    ami_data = None
    temp_data = {}
    fractal_data = None
    
    # Process meter file
    if meter_file is not None:
        try:
            # Reset file position
            meter_file.seek(0)
            customer_info = get_master_sheet_info(meter_file)
            
            meter_file.seek(0)
            loader = MeterLoader(meter_file)
            loader.load_and_clean()
            
            for div in ["Electricity", "Water", "Gas"]:
                df_div = loader.get_division(div)
                if not df_div.empty:
                    meter_data[div] = df_div
                    
            if not meter_data:
                st.warning("Meter file loaded but no division data found. Check that 'Division' column exists.")
                
        except Exception as e:
            st.error("**Meter File Error**")
            st.code(str(e), language=None)
            st.info("**Troubleshooting Tips:**\n"
                   "1. Ensure file has a sheet named 'Consumption'\n"
                   "2. Required columns: Division, MR Date, Days, Consumption\n"
                   "3. Check that dates are in a recognizable format")
    
    # Process AMI file
    if ami_file is not None:
        try:
            ami_file.seek(0)
            ami_loader = AMILoader(ami_file)
            df_ami = ami_loader.load()
            ami_feats = AMIFeatures(df_ami, unit=ami_loader.unit).compute()
            ami_data = {
                "df": df_ami,
                "features": ami_feats,
                "util_type": ami_loader.util_type,
                "unit": ami_loader.unit,
                "format": ami_loader.format_detected,
            }
            
            # Show format detected
            st.sidebar.success(f"AMI Format: {ami_loader.format_detected}")
            
        except Exception as e:
            st.error("**AMI File Error**")
            st.code(str(e), language=None)
            st.info("**Supported AMI Formats:**\n"
                   "- **Format A:** Timestamp like 'Jan 15, 2025 - 2:30 PM EST'\n"
                   "- **Format B:** Timestamp like '01/12/2026 00:15 EST'\n"
                   "- **Format C:** Separate Date and Time columns\n"
                   "- **Format D:** Standard datetime + numeric value\n\n"
                   "**Required:** Date/Time column + Value column (numeric)")
    
    # Display customer info
    if customer_info and "customer_name" in customer_info:
        st.markdown(f"### {customer_info.get('customer_name', 'Customer')}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Account:** {customer_info.get('account', 'N/A')}")
        with col2:
            st.markdown(f"**Address:** {customer_info.get('address', 'N/A')}")
        with col3:
            st.markdown(f"**Survey Date:** {customer_info.get('survey_date', 'N/A')}")
        st.markdown("---")
    
    # Create tabs
    tab_names = ["Overview"]
    if meter_data:
        tab_names.append("Meter Analysis")
    if ami_data:
        tab_names.append("AMI Analysis")
    if meter_data or ami_data:
        tab_names.append("Temperature Analysis")
    if ami_data:
        tab_names.append("Advanced Analysis")
    tab_names.append("Export Report")
    
    tabs = st.tabs(tab_names)
    tab_index = 0
    
    # ─── OVERVIEW TAB ───────────────────────────────────────────────────────────
    with tabs[tab_index]:
        tab_index += 1
        
        st.header("Analysis Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Summary")
            
            if meter_data:
                st.markdown("**Meter Data:**")
                for div, df in meter_data.items():
                    feats = MeterFeatures(df).compute_features()
                    st.markdown(f"""
                    - **{div}**: {len(df)} readings
                      - Total: {feats['total_consumption']:,.0f} {feats['unit']}
                      - Daily Avg: {feats['overall_daily_avg']:.1f} {feats['unit']}/day
                      - Quality Score: {feats['quality_score']}/100
                    """)
            
            if ami_data:
                feats = ami_data["features"]
                st.markdown(f"""
                **AMI Data ({ami_data['util_type']}):**
                - Intervals: {len(ami_data['df']):,}
                - Date Range: {ami_data['df']['timestamp'].min().date()} to {ami_data['df']['timestamp'].max().date()}
                - Interval: {feats['interval_minutes']} minutes
                - Daily Avg: {feats['daily_avg_kwh']:.1f} {ami_data['unit']}
                """)
        
        with col2:
            st.subheader("Quick Insights")
            
            if meter_data and "Electricity" in meter_data:
                df_elec = meter_data["Electricity"]
                feats = MeterFeatures(df_elec).compute_features()
                
                if feats["n_anomalies"] > 0:
                    info_box(f"Found {feats['n_anomalies']} anomalous billing periods in electricity data that may warrant investigation.", "warning")
                else:
                    info_box("No significant anomalies detected in billing data.", "success")
            
            if ami_data:
                feats = ami_data["features"]
                if feats["load_factor"] < 0.3:
                    info_box(f"Load factor is {feats['load_factor']:.1%}. Low load factor indicates peaky demand - potential for load shifting.", "info")
                elif feats["load_factor"] > 0.5:
                    info_box(f"Load factor is {feats['load_factor']:.1%}. Good load factor indicates consistent usage patterns.", "success")
    
    # ─── METER ANALYSIS TAB ─────────────────────────────────────────────────────
    if meter_data:
        with tabs[tab_index]:
            tab_index += 1
            
            st.header("Meter Data Analysis")
            
            for div, df in meter_data.items():
                st.subheader(div)
                
                feats = MeterFeatures(df).compute_features()
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Consumption", f"{feats['total_consumption']:,.0f} {feats['unit']}")
                with col2:
                    st.metric("Daily Average", f"{feats['overall_daily_avg']:.1f} {feats['unit']}/day")
                with col3:
                    st.metric("Peak Period", f"{feats['peak_consumption']:,.0f} {feats['unit']}")
                with col4:
                    st.metric("Anomalies", feats['n_anomalies'])
                
                # Charts
                fig = create_meter_consumption_chart(feats["df_with_anomalies"], f"{div} - Consumption History")
                st.pyplot(fig)
                plt.close(fig)
                
                fig = create_daily_average_chart(df, f"{div} - Daily Average")
                st.pyplot(fig)
                plt.close(fig)
                
                # Anomaly details
                if feats["n_anomalies"] > 0:
                    with st.expander("View Anomalous Periods"):
                        anomalies = feats["df_with_anomalies"][feats["df_with_anomalies"]["anomaly"]]
                        st.dataframe(anomalies[["mr_date", "consumption", "days", "avg_daily"]].reset_index(drop=True))
                
                st.markdown("---")
    
    # ─── AMI ANALYSIS TAB ───────────────────────────────────────────────────────
    if ami_data:
        with tabs[tab_index]:
            tab_index += 1
            
            st.header(f"AMI Interval Analysis ({ami_data['util_type']})")
            
            df_ami = ami_data["df"]
            feats = ami_data["features"]
            unit = ami_data["unit"]
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Daily Average", f"{feats['daily_avg']:.1f} {unit}")
            with col2:
                st.metric("Peak Interval", f"{feats['peak_val']:.1f} {unit}")
            with col3:
                st.metric("Base Load", f"{feats['base_load']:.2f} {unit}")
            with col4:
                st.metric("Load Factor", f"{feats['load_factor']:.1%}")
            
            # Charts
            st.subheader("Load Shape")
            fig = create_ami_load_shape(df_ami, title=f"{ami_data['util_type']} Load Shape", unit=unit)
            st.pyplot(fig)
            plt.close(fig)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Daily Totals")
                fig, ax = plt.subplots(figsize=(10, 5))
                daily = feats["daily_series"]
                ax.bar(daily.index, daily.values, color=COLORS["secondary"], alpha=0.8)
                ax.axhline(feats["daily_avg"], color=COLORS["accent"], linestyle="--", 
                          linewidth=2, label=f"Avg: {feats['daily_avg']:.1f} {unit}")
                ax.set_xlabel("Date")
                ax.set_ylabel(f"Daily {unit}")
                ax.set_title("Daily Consumption", fontweight="bold")
                ax.legend()
                fig.autofmt_xdate()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                st.subheader("Hourly Profile")
                fig = create_hourly_profile(feats["avg_by_hour"], title=f"Average Hourly {unit}", unit=unit)
                st.pyplot(fig)
                plt.close(fig)
            
            # Interpretation
            st.subheader("Auditor Notes")
            
            if feats["tou_ratio"] > 1.3:
                info_box(f"Peak-to-off-peak ratio is {feats['tou_ratio']:.2f}. Significant peak hour usage - time-of-use rate may benefit from load shifting.", "warning")
            
            if ami_data["util_type"] == "Electric" and feats["base_load_rate"] > 1.0:
                info_box(f"Base load is {feats['base_load_rate']:.2f} kW. Consider investigating always-on loads: refrigeration, phantom loads, pool pumps.", "info")
            elif ami_data["util_type"] == "Water" and feats["base_load"] > 10:
                info_box(f"Base load is {feats['base_load']:.1f} {unit}/interval. Check for potential leaks or continuous water use.", "info")
    
    # ─── TEMPERATURE ANALYSIS TAB ───────────────────────────────────────────────
    if meter_data or ami_data:
        with tabs[tab_index]:
            tab_index += 1
            
            st.header("Temperature Analysis")
            
            # Determine date range
            if ami_data:
                date_min = ami_data["df"]["timestamp"].min()
                date_max = ami_data["df"]["timestamp"].max()
            elif meter_data:
                first_div = list(meter_data.values())[0]
                date_min = first_div["mr_date"].min() - pd.Timedelta(days=35)
                date_max = first_div["mr_date"].max()
            
            # Fetch temperature data
            with st.spinner("Fetching temperature data..."):
                df_temp = get_temperature_data(date_min, date_max)
            
            if df_temp is None:
                st.error("Could not fetch temperature data. Check your internet connection.")
            else:
                temp_data["df"] = df_temp
                
                # AMI temperature analysis
                if ami_data:
                    st.subheader("AMI vs Temperature")
                    
                    daily = ami_data["features"]["daily_series"].reset_index()
                    daily.columns = ["date", "kwh"]
                    daily["date"] = pd.to_datetime(daily["date"])
                    
                    merged = daily.merge(
                        df_temp.reset_index().rename(columns={"index": "date"}),
                        on="date", how="inner"
                    ).dropna()
                    
                    if not merged.empty:
                        temp_data["merged"] = merged
                        
                        fig = create_temperature_overlay(merged, "Daily Usage vs Temperature")
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Correlation
                        if ami_data["util_type"] == "Gas":
                            corr = merged["kwh"].corr(merged["temp_avg"])
                            st.markdown(f"**Temperature Correlation:** r = {corr:.2f}")
                            if corr < -0.5:
                                info_box("Strong negative correlation confirms heating load. Gas usage increases significantly as temperature drops.", "success")
                        else:
                            merged["temp_delta"] = (merged["temp_avg"] - comfort_base).abs()
                            corr = merged["kwh"].corr(merged["temp_delta"])
                            st.markdown(f"**Temperature Sensitivity:** r = {corr:.2f}")
                            if corr > 0.6:
                                info_box("Strong HVAC relationship. Building responds predictably to temperature deviations from comfort baseline.", "success")
                            elif corr > 0.3:
                                info_box("Moderate temperature sensitivity. Some HVAC load but other factors also significant.", "info")
                            else:
                                info_box("Weak temperature sensitivity. Usage driven primarily by non-HVAC loads.", "info")
                
                # Meter weather anomaly analysis
                if meter_data and "Electricity" in meter_data:
                    st.subheader("Weather-Normalized Anomaly Detection")
                    
                    st.markdown("""
                    **What This Shows:**
                    This analysis compares actual electricity consumption to what would be expected based on weather (heating and cooling degree days).
                    Periods that deviate significantly from expectations are flagged as anomalies.
                    """)
                    
                    df_elec = meter_data["Electricity"]
                    detector = WeatherAnomalyDetector(df_elec, df_temp, comfort_base, z_threshold)
                    anomaly_df = detector.run()
                    
                    if anomaly_df is not None:
                        temp_data["anomaly_df"] = anomaly_df
                        temp_data["detector"] = detector
                        
                        fig = create_weather_anomaly_chart(anomaly_df, "Actual vs Expected Consumption")
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Interpretations
                        interpretations = detector.get_interpretation()
                        if interpretations:
                            st.subheader("Auditor Findings")
                            for interp in interpretations:
                                info_box(f"**{interp['title']}**<br>{interp['text']}", interp["type"])
                        
                        # Anomaly table
                        with st.expander("View Anomaly Details"):
                            display_cols = ["mr_date", "consumption", "daily_avg", "predicted", "residual_z", "anomaly_high", "anomaly_low"]
                            display_cols = [c for c in display_cols if c in anomaly_df.columns]
                            st.dataframe(anomaly_df[display_cols].reset_index(drop=True))
                    else:
                        st.warning("Insufficient data for weather-normalized analysis (need at least 6 billing periods).")
    
    # ─── ADVANCED ANALYSIS TAB ──────────────────────────────────────────────────
    if ami_data:
        with tabs[tab_index]:
            tab_index += 1
            
            st.header("Advanced Analysis: Complexity Patterns")
            
            st.markdown("""
            **What is Fractal Analysis?**
            
            This analysis uses Detrended Fluctuation Analysis (DFA) to measure the *complexity* of energy consumption patterns.
            The key metric is the **Hurst Exponent (H)**, which reveals whether usage patterns are:
            
            - **Anti-persistent (H < 0.5)**: Variable, unpredictable behavior. High periods tend to be followed by low periods.
            - **Random (H = 0.5)**: No clear pattern or memory in the data.
            - **Persistent (H > 0.5)**: Consistent, predictable patterns. Trends tend to continue.
            
            *Based on research by Knowles et al. (2017), University of Florida - Energy and Buildings*
            """)
            
            st.markdown("---")
            
            # Run fractal analysis
            analyzer = FractalAnalyzer(ami_data["df"])
            dfa_result = analyzer.detrended_fluctuation_analysis()
            dist_result = analyzer.distribution_analysis()
            
            fractal_data = {
                "dfa": dfa_result,
                "distribution": dist_result,
                "values": analyzer.values,
                "differenced": analyzer.differenced,
            }
            
            # Display results
            if "error" not in dfa_result:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Hurst Exponent", f"{dfa_result['hurst_exponent']:.3f}")
                with col2:
                    st.metric("R-Squared", f"{dfa_result['r_squared']:.3f}")
                with col3:
                    if dfa_result['hurst_exponent'] < 0.45:
                        pattern = "Anti-Persistent"
                    elif dfa_result['hurst_exponent'] > 0.55:
                        pattern = "Persistent"
                    else:
                        pattern = "Random/Mixed"
                    st.metric("Pattern Type", pattern)
            
            # DFA chart
            st.subheader("DFA Scaling Analysis")
            fig = create_dfa_chart(dfa_result)
            st.pyplot(fig)
            plt.close(fig)
            
            # Distribution chart
            st.subheader("Distribution Analysis")
            fig = create_distribution_chart(analyzer.values, analyzer.differenced)
            st.pyplot(fig)
            plt.close(fig)
            
            # Statistics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Raw Data Statistics:**")
                st.markdown(f"- Skewness: {dist_result['raw']['skewness']:.2f}")
                st.markdown(f"- Kurtosis: {dist_result['raw']['kurtosis']:.2f}")
                st.markdown(f"- Normal: {'Yes' if dist_result['normality_tests']['raw_is_normal'] else 'No'}")
            
            with col2:
                st.markdown("**Differenced Data Statistics:**")
                st.markdown(f"- Skewness: {dist_result['differenced']['skewness']:.2f}")
                st.markdown(f"- Kurtosis: {dist_result['differenced']['kurtosis']:.2f}")
                st.markdown(f"- Normal: {'Yes' if dist_result['normality_tests']['diff_is_normal'] else 'No'}")
            
            # Interpretations
            st.subheader("Auditor Findings")
            interpretations = analyzer.get_interpretation(dfa_result, dist_result)
            for interp in interpretations:
                info_box(f"**{interp['title']}**<br>{interp['text']}", interp["type"])
    
    # ─── EXPORT TAB ─────────────────────────────────────────────────────────────
    with tabs[tab_index]:
        st.header("Export Report")
        
        st.markdown("""
        Generate a PDF report to share with clients or include in audit documentation.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Standard Report")
            st.markdown("""
            **Includes:**
            - Customer information
            - Consumption history charts
            - Daily average trends
            - Temperature correlation analysis
            - Weather-normalized anomaly detection
            - Summary statistics
            
            **Best for:** Basic audit documentation and client summaries.
            """)
            
            if st.button("Generate Standard Report", type="primary"):
                with st.spinner("Generating report..."):
                    pdf_buffer = generate_pdf_report(
                        customer_info=customer_info,
                        meter_data=meter_data if meter_data else None,
                        ami_data=ami_data,
                        temp_data=temp_data,
                        fractal_data=None,
                        report_type="standard"
                    )
                    
                    customer_name = customer_info.get("customer_name", "Customer") if customer_info else "Customer"
                    filename = f"Energy_Audit_{customer_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
                    
                    st.download_button(
                        label="Download Standard Report",
                        data=pdf_buffer,
                        file_name=filename,
                        mime="application/pdf"
                    )
        
        with col2:
            st.subheader("Advanced Report")
            st.markdown("""
            **Includes everything in Standard, plus:**
            - Fractal complexity analysis
            - Hurst exponent calculations
            - Distribution analysis
            - Behavior pattern classification
            - Advanced auditor recommendations
            
            **Best for:** Detailed technical analysis and research-quality documentation.
            """)
            
            if ami_data:
                if st.button("Generate Advanced Report", type="secondary"):
                    with st.spinner("Generating report..."):
                        # Run fractal analysis if not already done
                        if fractal_data is None:
                            analyzer = FractalAnalyzer(ami_data["df"])
                            dfa_result = analyzer.detrended_fluctuation_analysis()
                            dist_result = analyzer.distribution_analysis()
                            fractal_data = {
                                "dfa": dfa_result,
                                "distribution": dist_result,
                                "values": analyzer.values,
                                "differenced": analyzer.differenced,
                            }
                        
                        pdf_buffer = generate_pdf_report(
                            customer_info=customer_info,
                            meter_data=meter_data if meter_data else None,
                            ami_data=ami_data,
                            temp_data=temp_data,
                            fractal_data=fractal_data,
                            report_type="advanced"
                        )
                        
                        customer_name = customer_info.get("customer_name", "Customer") if customer_info else "Customer"
                        filename = f"Energy_Audit_Advanced_{customer_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
                        
                        st.download_button(
                            label="Download Advanced Report",
                            data=pdf_buffer,
                            file_name=filename,
                            mime="application/pdf"
                        )
            else:
                st.info("Upload AMI data to enable advanced report generation.")


if __name__ == "__main__":
    main()
