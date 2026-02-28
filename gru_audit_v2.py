#!/usr/bin/env python3
"""
GRU Energy Audit Analyzer v2.0 — Pre-Survey Report Tool
========================================================

Enhanced command-line tool for energy auditors with:
- Weather-normalized anomaly detection (HDD/CDD)
- Temperature data caching
- Change-point detection (CUSUM)
- Load factor analysis for AMI
- HTML and PDF report generation
- Multi-utility support (Electric, Water, Gas)

Usage:
    python gru_audit_v2.py <customer_file.xlsx> [options]

Options:
    --ami, -a       Path to AMI interval data file
    --save, -s      Save charts to disk
    --html          Generate HTML report
    --pdf           Generate PDF report
    --verbose, -v   Enable verbose output
    --cache-dir     Directory for temperature cache (default: ./temp_cache)

Author: GRU Energy Audit Team
Version: 2.0
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import requests

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Centralized configuration."""
    
    # Location (Gainesville, FL)
    latitude: float = 29.6516
    longitude: float = -82.3248
    
    # Anomaly detection
    isolation_contamination: float = 0.05
    residual_z_threshold: float = 2.5
    min_history_periods: int = 8
    persistence_periods: int = 2
    
    # Temperature
    comfort_baseline: float = 65.0  # °F
    
    # Change-point detection
    cusum_threshold: float = 4.0
    
    # Display
    figure_dpi: int = 100
    default_figsize: Tuple[int, int] = (12, 4)
    
    # Caching
    cache_dir: str = "./temp_cache"
    
    # Verbosity
    verbose: bool = False
    
    # Colors
    colors: Dict[str, str] = field(default_factory=lambda: {
        "electric": "#2E86AB",
        "water": "#028090",
        "gas": "#F18F01",
        "anomaly_high": "#C73E1D",
        "anomaly_low": "#457B9D",
        "normal": "#A3B18A",
        "hot": "#E63946",
        "cold": "#457B9D",
        "mild": "#2A9D8F",
        "predicted": "#1D3557",
        "ci_band": "#A8DADC",
    })


# Global config instance
cfg = Config()


def log(msg: str, level: str = "info"):
    """Conditional logging based on verbosity."""
    if cfg.verbose or level == "error":
        prefix = {"info": "  ", "warn": "  ⚠ ", "error": "  ✘ ", "ok": "  ✔ "}
        print(f"{prefix.get(level, '  ')}{msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class TemperatureCache:
    """
    Cache temperature data to avoid repeated API calls.
    
    Stores data as CSV files with date range in filename.
    """
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_key(self, start: str, end: str) -> str:
        """Generate cache filename."""
        key = f"{cfg.latitude}_{cfg.longitude}_{start}_{end}"
        hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
        return f"temp_{hash_key}.csv"
    
    def get(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """Retrieve cached temperature data if available."""
        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        
        cache_file = self.cache_dir / self._cache_key(start, end)
        
        if cache_file.exists():
            log(f"Using cached temperature data: {cache_file.name}")
            df = pd.read_csv(cache_file, parse_dates=["date"], index_col="date")
            return df
        
        return None
    
    def save(self, df: pd.DataFrame, start_date, end_date):
        """Save temperature data to cache."""
        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        
        cache_file = self.cache_dir / self._cache_key(start, end)
        df.to_csv(cache_file)
        log(f"Cached temperature data: {cache_file.name}")
    
    def get_or_fetch(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """Get from cache or fetch from API."""
        # Check cache first
        cached = self.get(start_date, end_date)
        if cached is not None:
            return cached
        
        # Fetch from API
        df = self._fetch_from_api(start_date, end_date)
        
        if df is not None:
            self.save(df, start_date, end_date)
        
        return df
    
    def _fetch_from_api(self, start_date, end_date) -> Optional[pd.DataFrame]:
        """Fetch temperature data from Open-Meteo API."""
        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": cfg.latitude,
            "longitude": cfg.longitude,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_max,temperature_2m_min",
            "temperature_unit": "fahrenheit",
            "timezone": "America/New_York",
        }
        
        try:
            log(f"Fetching temperature data: {start} → {end}")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()["daily"]
            
            df = pd.DataFrame({
                "date": pd.to_datetime(data["time"]),
                "temp_max": data["temperature_2m_max"],
                "temp_min": data["temperature_2m_min"],
            })
            df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
            df = df.set_index("date")
            
            log(f"Loaded {len(df)} days of temperature data", "ok")
            return df
            
        except requests.exceptions.Timeout:
            log("Temperature API timeout - try again later", "error")
            return None
        except requests.exceptions.RequestException as e:
            log(f"Temperature API error: {e}", "error")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            log(f"Temperature data parse error: {e}", "error")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# METER DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class MeterLoader:
    """Load and clean GRU meter reading Excel files."""
    
    COLUMN_MAP = {
        "Division": "division",
        "Device": "device",
        "MR Reason": "mr_reason",
        "MR Type": "mr_type",
        "MR Date": "mr_date",
        "Days": "days",
        "MR Result": "mr_result",
        "MR Unit": "mr_unit",
        "Consumption": "consumption",
        "Avg.": "avg_daily",
        "Avg": "avg_daily",
    }
    
    NON_READ_REASONS = {3}  # Estimated reads
    VLINE_REASONS = {6, 21, 22}  # Move-in, meter change
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.has_mr_reason = False
    
    def _find_sheet(self, xl: pd.ExcelFile) -> str:
        """Find the consumption sheet with data."""
        for name in xl.sheet_names:
            if "consumption" in name.lower():
                df = pd.read_excel(xl, sheet_name=name, header=None, nrows=5)
                if not df.empty and len(df.columns) > 0:
                    return name
        
        raise ValueError(
            f"No valid consumption data found.\n"
            f"  Sheets available: {xl.sheet_names}\n"
            f"  Please ensure 'Consumption History' tab contains meter reading data\n"
            f"  with columns: Division, MR Date, Days, Consumption, etc."
        )
    
    def _detect_header_row(self, xl: pd.ExcelFile, sheet: str) -> int:
        """Find row containing 'Division' header."""
        for i in range(5):
            df = pd.read_excel(xl, sheet_name=sheet, header=i, nrows=1)
            cols = [str(c).strip() for c in df.columns]
            if "Division" in cols:
                return i
        return 0
    
    def _clean_numeric(self, series: pd.Series) -> pd.Series:
        """Clean numeric column."""
        if series.dtype == object:
            s = series.astype(str).str.replace(",", "", regex=False)
            return pd.to_numeric(s, errors="coerce")
        return pd.to_numeric(series, errors="coerce")
    
    def load(self) -> pd.DataFrame:
        """Load and clean meter reading data."""
        xl = pd.ExcelFile(self.filepath)
        sheet = self._find_sheet(xl)
        header_row = self._detect_header_row(xl, sheet)
        
        log(f"Reading sheet '{sheet}' with header at row {header_row}")
        
        df = pd.read_excel(xl, sheet_name=sheet, header=header_row)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns=self.COLUMN_MAP)
        
        self.has_mr_reason = "mr_reason" in df.columns
        
        # Parse and clean
        df["mr_date"] = pd.to_datetime(df["mr_date"], errors="coerce")
        df["consumption"] = self._clean_numeric(df["consumption"])
        
        for col in ["mr_result", "days", "avg_daily"]:
            if col in df.columns:
                df[col] = self._clean_numeric(df[col])
        
        df = df.dropna(subset=["mr_date"])
        
        if self.has_mr_reason:
            df["mr_reason"] = pd.to_numeric(df["mr_reason"], errors="coerce")
            df = df[~df["mr_reason"].isin(self.NON_READ_REASONS)]
            df = df[(df["consumption"] > 0) | (df["mr_reason"].isin(self.VLINE_REASONS))]
        else:
            df = df[df["consumption"] > 0]
        
        df = df[df["days"] > 0]
        df = df.sort_values(["division", "device", "mr_date"]).reset_index(drop=True)
        
        self.df = df
        log(f"Loaded {len(df)} meter readings", "ok")
        return df
    
    def get_division(self, name: str) -> pd.DataFrame:
        """Get data for a specific division."""
        if self.df is None:
            raise RuntimeError("Call load() first")
        sub = self.df[self.df["division"] == name].copy()
        if not sub.empty:
            sub = sub[sub["mr_date"] > sub["mr_date"].min()].reset_index(drop=True)
        return sub
    
    def get_customer_info(self) -> Dict[str, str]:
        """Extract customer info from Master Sheet."""
        try:
            ms = pd.read_excel(self.filepath, sheet_name="Master Sheet", header=None)
            
            cell_0_6 = str(ms.iloc[0, 6]).strip() if pd.notna(ms.iloc[0, 6]) else ""
            offset = 1 if cell_0_6 and not any(c.isdigit() for c in cell_0_6) else 0
            
            def safe_get(r, c):
                try:
                    val = ms.iloc[r + offset, c]
                    return str(val).strip() if pd.notna(val) else ""
                except:
                    return ""
            
            return {
                "account": safe_get(0, 6),
                "name": safe_get(1, 6),
                "own_rent": safe_get(2, 6),
                "community": safe_get(3, 6),
                "address": safe_get(4, 6),
                "city": safe_get(5, 6) or "Gainesville FL",
            }
        except Exception as e:
            log(f"Could not read Master Sheet: {e}", "warn")
            return {"account": "Unknown", "name": "Unknown"}


# ═══════════════════════════════════════════════════════════════════════════════
# METER FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class MeterFeatures:
    """Compute features and detect anomalies from meter data."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().sort_values("mr_date").reset_index(drop=True)
    
    def compute(self) -> Dict[str, Any]:
        """Compute all features."""
        df = self.df
        
        total = df["consumption"].sum()
        total_days = df["days"].sum()
        daily_avg = total / total_days if total_days > 0 else None
        peak = df["consumption"].max()
        base = df["consumption"].quantile(0.05)
        avg_interval = df["days"].mean()
        
        period_series = df.set_index("mr_date")["consumption"]
        rolling_avg = period_series.rolling(window=3).mean()
        
        # Isolation Forest anomaly detection
        iso_cols = [c for c in ["consumption", "days", "avg_daily"] if c in df.columns]
        iso_data = df[iso_cols].dropna()
        df["anomaly"] = False
        
        if len(iso_data) >= 5:
            model = IsolationForest(
                contamination=cfg.isolation_contamination,
                random_state=42
            )
            preds = model.fit_predict(iso_data)
            df.loc[iso_data.index, "anomaly"] = (preds == -1)
        
        n_anomalies = int(df["anomaly"].sum())
        unit = df["mr_unit"].iloc[0] if "mr_unit" in df.columns else ""
        quality = self._quality_score(df)
        
        return {
            "total": total,
            "daily_avg": daily_avg,
            "peak": peak,
            "base": base,
            "avg_interval": avg_interval,
            "n_reads": len(df),
            "n_anomalies": n_anomalies,
            "unit": unit,
            "quality_score": quality,
            "period_series": period_series,
            "rolling_avg": rolling_avg,
            "df": df,
        }
    
    def _quality_score(self, df: pd.DataFrame) -> int:
        """Compute data quality score (0-100)."""
        score = 100
        if df["consumption"].isna().any(): score -= 10
        if df["days"].std() > 10: score -= 5
        if len(df) < 12: score -= 15
        if (df["consumption"] == 0).sum() > 2: score -= 10
        if len(df) > 1:
            gaps = df["mr_date"].diff().dt.days
            if gaps.max() > 60: score -= 20
        return max(0, score)


# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER-NORMALIZED ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_degree_days(temp_avg: float, base: float = None) -> Tuple[float, float]:
    """Compute Heating and Cooling Degree Days."""
    base = base or cfg.comfort_baseline
    hdd = max(0, base - temp_avg)
    cdd = max(0, temp_avg - base)
    return hdd, cdd


class WeatherAnomalyDetector:
    """
    Weather-normalized anomaly detection using HDD/CDD regression.
    
    Supports Electric (HDD+CDD), Water (CDD-focused), and Gas (HDD-focused).
    """
    
    def __init__(self, df_temp: pd.DataFrame):
        self.df_temp = df_temp
    
    def analyze(self, df_division: pd.DataFrame, division: str = "Electricity") -> Dict[str, Any]:
        """
        Run weather-normalized anomaly detection.
        
        Returns dict with:
            - df: DataFrame with predictions, residuals, anomaly flags
            - summary: Dict with aggregate statistics
        """
        df = df_division.copy().sort_values("mr_date").reset_index(drop=True)
        
        if len(df) < cfg.min_history_periods:
            return {"df": pd.DataFrame(), "summary": {"error": "Insufficient data"}}
        
        # Build period-level features
        period_data = []
        for _, row in df.iterrows():
            end_date = row["mr_date"]
            start_date = end_date - pd.Timedelta(days=int(row["days"]))
            mask = (self.df_temp.index >= start_date) & (self.df_temp.index <= end_date)
            temp_slice = self.df_temp.loc[mask]
            
            if temp_slice.empty:
                continue
            
            temp_avg = temp_slice["temp_avg"].mean()
            hdd, cdd = compute_degree_days(temp_avg)
            
            period_data.append({
                "mr_date": end_date,
                "consumption": row["consumption"],
                "days": row["days"],
                "daily": row["consumption"] / row["days"],
                "temp_avg": temp_avg,
                "hdd": hdd,
                "cdd": cdd,
            })
        
        df_periods = pd.DataFrame(period_data)
        
        if len(df_periods) < cfg.min_history_periods:
            return {"df": pd.DataFrame(), "summary": {"error": "Insufficient temperature overlap"}}
        
        # Select features based on utility type
        if division == "Gas":
            feature_cols = ["hdd"]  # Gas correlates with heating
        elif division == "Water":
            feature_cols = ["cdd"]  # Water correlates with cooling (irrigation, etc.)
        else:
            feature_cols = ["hdd", "cdd"]  # Electric uses both
        
        # Rolling prediction
        results = []
        for i in range(cfg.min_history_periods, len(df_periods)):
            hist = df_periods.iloc[:i]
            current = df_periods.iloc[i]
            
            X_hist = hist[feature_cols].values
            y_hist = hist["daily"].values
            X_curr = current[feature_cols].values.reshape(1, -1)
            
            model = HuberRegressor(epsilon=1.35)
            model.fit(X_hist, y_hist)
            
            predicted = model.predict(X_curr)[0]
            residual = current["daily"] - predicted
            
            hist_residuals = y_hist - model.predict(X_hist)
            resid_std = np.std(hist_residuals) if np.std(hist_residuals) > 0 else 1
            resid_z = residual / resid_std
            
            n = len(hist)
            se_pred = resid_std * np.sqrt(1 + 1/n)
            
            results.append({
                "mr_date": current["mr_date"],
                "actual_daily": current["daily"],
                "predicted_daily": predicted,
                "residual": residual,
                "residual_z": resid_z,
                "ci_lower": predicted - 1.96 * se_pred,
                "ci_upper": predicted + 1.96 * se_pred,
                "temp_avg": current["temp_avg"],
                "hdd": current["hdd"],
                "cdd": current["cdd"],
            })
        
        df_result = pd.DataFrame(results)
        
        if df_result.empty:
            return {"df": df_result, "summary": {"error": "No predictions generated"}}
        
        # Flag anomalies
        df_result["anomaly_high"] = df_result["residual_z"] > cfg.residual_z_threshold
        df_result["anomaly_low"] = df_result["residual_z"] < -cfg.residual_z_threshold
        df_result["anomaly"] = df_result["anomaly_high"] | df_result["anomaly_low"]
        
        # Persistence
        df_result["persistent_high"] = (
            df_result["anomaly_high"].rolling(cfg.persistence_periods).sum()
            >= cfg.persistence_periods
        )
        df_result["persistent_low"] = (
            df_result["anomaly_low"].rolling(cfg.persistence_periods).sum()
            >= cfg.persistence_periods
        )
        
        summary = {
            "n_periods": len(df_result),
            "n_anomaly_high": int(df_result["anomaly_high"].sum()),
            "n_anomaly_low": int(df_result["anomaly_low"].sum()),
            "n_persistent_high": int(df_result["persistent_high"].sum()),
            "n_persistent_low": int(df_result["persistent_low"].sum()),
            "latest_z": df_result["residual_z"].iloc[-1] if len(df_result) > 0 else None,
        }
        
        return {"df": df_result, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE-POINT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_change_point(df_division: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Detect usage regime changes using CUSUM approach.
    
    Returns change point info or None if no change detected.
    """
    df = df_division.copy().sort_values("mr_date")
    
    if len(df) < 12:
        return None
    
    daily = df["consumption"] / df["days"]
    z = (daily - daily.mean()) / daily.std()
    
    # CUSUM
    cusum_pos = np.maximum.accumulate(np.cumsum(z.values) - np.arange(len(z)) * 0.5)
    cusum_neg = np.minimum.accumulate(np.cumsum(z.values) + np.arange(len(z)) * 0.5)
    
    change_idx = None
    direction = None
    
    if cusum_pos.max() > cfg.cusum_threshold:
        change_idx = int(np.argmax(cusum_pos > cfg.cusum_threshold))
        direction = "increase"
    elif cusum_neg.min() < -cfg.cusum_threshold:
        change_idx = int(np.argmax(cusum_neg < -cfg.cusum_threshold))
        direction = "decrease"
    
    if change_idx is not None and change_idx > 0 and change_idx < len(df) - 1:
        pre_avg = daily.iloc[:change_idx].mean()
        post_avg = daily.iloc[change_idx:].mean()
        
        return {
            "change_date": df.iloc[change_idx]["mr_date"],
            "change_idx": change_idx,
            "direction": direction,
            "pre_avg": pre_avg,
            "post_avg": post_avg,
            "pct_change": (post_avg / pre_avg - 1) * 100 if pre_avg > 0 else 0,
        }
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# AMI LOADER AND FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class AMILoader:
    """Load and clean AMI interval data."""
    
    SHEET_MAP = {
        "ELECTRIC": "Electric", "Electric": "Electric", "Sheet1": "Electric",
        "WATER": "Water", "Water": "Water",
        "GAS": "Gas", "Gas": "Gas",
    }
    UNITS = {"Electric": "kWh", "Water": "Gal", "Gas": "CCF"}
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.util_type = None
        self.unit = None
    
    def load(self) -> pd.DataFrame:
        """Load and clean AMI data."""
        xl = pd.ExcelFile(self.filepath)
        
        sheet = None
        for name in xl.sheet_names:
            if name in self.SHEET_MAP:
                sheet = name
                break
        sheet = sheet or xl.sheet_names[0]
        
        self.util_type = self.SHEET_MAP.get(sheet, "Electric")
        self.unit = self.UNITS[self.util_type]
        
        df = pd.read_excel(xl, sheet_name=sheet, header=None, skiprows=4)
        df = df[[0, 1]].copy()
        df.columns = ["timestamp", "raw_value"]
        
        df["timestamp"] = (df["timestamp"].astype(str)
                          .str.replace(r"\s+E[SD]T.*$", "", regex=True)
                          .str.strip())
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%b %d, %Y - %I:%M %p", errors="coerce"
        )
        
        df["value"] = (df["raw_value"].astype(str)
                      .str.replace(",", "", regex=False)
                      .str.extract(r"([\d.]+)")[0])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        if self.util_type == "Electric":
            df["value"] = df["value"] / 1000
        
        df["kwh"] = df["value"]
        df = df.dropna(subset=["timestamp", "value"])
        df = df[df["value"] > 0]
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        self.df = df
        log(f"Loaded {len(df)} AMI intervals ({self.util_type})", "ok")
        return df


class AMIFeatures:
    """Compute features from AMI data including load factor analysis."""
    
    def __init__(self, df: pd.DataFrame, interval_minutes: int = None):
        self.df = df.copy()
        self.interval_min = interval_minutes
    
    def compute(self) -> Dict[str, Any]:
        """Compute AMI features with load factor analysis."""
        df = self.df.sort_values("timestamp")
        
        # Auto-detect interval
        if self.interval_min is None:
            deltas = df["timestamp"].diff().dropna()
            interval = deltas.mode()[0]
            self.interval_min = int(interval.total_seconds() / 60)
        
        hours_per_interval = self.interval_min / 60
        
        # Base and peak
        base_kwh = df["kwh"].quantile(0.05)
        base_kw = base_kwh / hours_per_interval
        peak_kwh = df["kwh"].max()
        peak_kw = peak_kwh / hours_per_interval
        avg_kwh = df["kwh"].mean()
        avg_kw = avg_kwh / hours_per_interval
        
        # Load factor = Average Demand / Peak Demand
        load_factor = avg_kw / peak_kw if peak_kw > 0 else 0
        
        # Daily aggregates
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby("date")["kwh"].sum()
        daily_avg = daily.mean()
        peak_day = pd.Timestamp(daily.idxmax())
        
        # Daily peaks
        df["hour"] = df["timestamp"].dt.hour
        daily_peaks = df.groupby("date")["kwh"].max()
        avg_daily_peak_kw = (daily_peaks / hours_per_interval).mean()
        
        # Hourly profile
        hourly = df.groupby("hour")["kwh"].mean()
        peak_hour = hourly.idxmax()
        
        # Time-of-use analysis
        off_peak_hours = list(range(0, 7)) + list(range(22, 24))
        on_peak_hours = list(range(14, 20))
        
        off_peak_avg = df[df["hour"].isin(off_peak_hours)]["kwh"].mean()
        on_peak_avg = df[df["hour"].isin(on_peak_hours)]["kwh"].mean()
        tou_ratio = on_peak_avg / off_peak_avg if off_peak_avg > 0 else 1
        
        return {
            "interval_min": self.interval_min,
            "base_kwh": base_kwh,
            "base_kw": base_kw,
            "peak_kwh": peak_kwh,
            "peak_kw": peak_kw,
            "avg_kw": avg_kw,
            "load_factor": load_factor,
            "daily_avg": daily_avg,
            "daily_series": daily,
            "peak_day": peak_day,
            "avg_daily_peak_kw": avg_daily_peak_kw,
            "hourly_profile": hourly,
            "peak_hour": peak_hour,
            "off_peak_avg": off_peak_avg,
            "on_peak_avg": on_peak_avg,
            "tou_ratio": tou_ratio,
            "df": df,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class ChartGenerator:
    """Generate analysis charts."""
    
    def __init__(self, customer_info: Dict, output_dir: str = None):
        self.info = customer_info
        self.output_dir = Path(output_dir) if output_dir else None
        self.name = customer_info.get("name", "Unknown")
        self.account = customer_info.get("account", "Unknown")
        self.figures = []  # Store for PDF
    
    def _save_or_store(self, fig, name: str):
        """Save to disk or store for PDF."""
        self.figures.append((name, fig))
        if self.output_dir:
            path = self.output_dir / f"{self.account}_{name}.png"
            fig.savefig(path, dpi=cfg.figure_dpi, bbox_inches="tight")
            log(f"Saved chart: {path.name}")
    
    def plot_consumption_history(self, feats: Dict, division: str):
        """Plot consumption with anomalies."""
        df = feats["df"]
        unit = feats["unit"]
        
        fig, ax = plt.subplots(figsize=cfg.default_figsize)
        
        normal = df[~df["anomaly"]]
        anomaly = df[df["anomaly"]]
        
        ax.bar(normal["mr_date"], normal["consumption"], width=20,
               color=cfg.colors["normal"], alpha=0.8, label="Normal")
        ax.bar(anomaly["mr_date"], anomaly["consumption"], width=20,
               color=cfg.colors["anomaly_high"], alpha=0.9, label="Anomaly")
        
        ax.plot(feats["rolling_avg"].index, feats["rolling_avg"].values,
                color=cfg.colors["predicted"], linewidth=2, linestyle="--", 
                label="3-Period Avg")
        
        ax.set_title(f"{self.name} — {division} History", fontsize=12, fontweight="bold")
        ax.set_ylabel(unit)
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        plt.tight_layout()
        
        self._save_or_store(fig, f"{division.lower()}_history")
        return fig
    
    def plot_weather_anomaly(self, anomaly_result: Dict, division: str):
        """Plot weather-normalized anomaly detection."""
        df = anomaly_result["df"]
        
        if df.empty:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Top: Actual vs Predicted
        ax1.fill_between(df["mr_date"], df["ci_lower"], df["ci_upper"],
                        alpha=0.3, color=cfg.colors["ci_band"], label="95% CI")
        ax1.plot(df["mr_date"], df["predicted_daily"], 
                color=cfg.colors["predicted"], linewidth=2, linestyle="--", label="Predicted")
        
        normal = df[~df["anomaly"]]
        high = df[df["anomaly_high"]]
        low = df[df["anomaly_low"]]
        
        ax1.scatter(normal["mr_date"], normal["actual_daily"],
                   color=cfg.colors["normal"], s=50, zorder=5, label="Normal")
        ax1.scatter(high["mr_date"], high["actual_daily"],
                   color=cfg.colors["anomaly_high"], s=80, zorder=6, 
                   label="High", marker="^")
        ax1.scatter(low["mr_date"], low["actual_daily"],
                   color=cfg.colors["anomaly_low"], s=80, zorder=6,
                   label="Low", marker="v")
        
        ax1.set_title(f"{self.name} — {division} Weather-Normalized Analysis",
                     fontsize=12, fontweight="bold")
        ax1.set_ylabel("Daily Usage")
        ax1.legend(fontsize=8, loc="upper left")
        
        # Bottom: Z-scores
        colors = [cfg.colors["anomaly_high"] if z > cfg.residual_z_threshold
                 else cfg.colors["anomaly_low"] if z < -cfg.residual_z_threshold
                 else cfg.colors["normal"] for z in df["residual_z"]]
        
        ax2.bar(df["mr_date"], df["residual_z"], width=20, color=colors, alpha=0.7)
        ax2.axhline(cfg.residual_z_threshold, color=cfg.colors["anomaly_high"],
                   linestyle="--", linewidth=1.5)
        ax2.axhline(-cfg.residual_z_threshold, color=cfg.colors["anomaly_low"],
                   linestyle="--", linewidth=1.5)
        ax2.axhline(0, color="black", linewidth=1)
        ax2.set_ylabel("Z-Score")
        ax2.set_title("Residual Z-Score", fontsize=10)
        
        fig.autofmt_xdate()
        plt.tight_layout()
        
        self._save_or_store(fig, f"{division.lower()}_weather_anomaly")
        return fig
    
    def plot_ami_profile(self, feats: Dict, unit: str):
        """Plot AMI hourly profile with load factor."""
        hourly = feats["hourly_profile"]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        bars = ax.bar(hourly.index, hourly.values, color="#6A4C93", alpha=0.85, width=0.7)
        
        # Highlight peak hour
        peak_hour = feats["peak_hour"]
        bars[peak_hour].set_color(cfg.colors["anomaly_high"])
        
        ax.axhline(feats["base_kwh"], color=cfg.colors["mild"], linewidth=2,
                  linestyle="--", label=f"Base ({feats['base_kw']:.2f} kW)")
        
        ax.set_title(f"{self.name} — Hourly Profile (Load Factor: {feats['load_factor']:.1%})",
                    fontsize=12, fontweight="bold")
        ax.set_xlabel("Hour")
        ax.set_ylabel(f"{unit}/interval")
        ax.set_xticks(range(24))
        ax.legend(fontsize=8)
        plt.tight_layout()
        
        self._save_or_store(fig, "ami_profile")
        return fig
    
    def save_pdf(self, filepath: str):
        """Save all figures to PDF."""
        with PdfPages(filepath) as pdf:
            for name, fig in self.figures:
                pdf.savefig(fig, bbox_inches="tight")
        log(f"Saved PDF report: {filepath}", "ok")


class HTMLReportGenerator:
    """Generate HTML report."""
    
    def __init__(self, customer_info: Dict, output_path: str):
        self.info = customer_info
        self.output_path = output_path
        self.sections = []
    
    def add_section(self, title: str, content: str):
        """Add a section to the report."""
        self.sections.append({"title": title, "content": content})
    
    def add_stats_table(self, title: str, stats: Dict[str, Any]):
        """Add a statistics table."""
        rows = ""
        for key, value in stats.items():
            if value is not None:
                if isinstance(value, float):
                    val_str = f"{value:,.2f}"
                else:
                    val_str = str(value)
                rows += f"<tr><td>{key}</td><td>{val_str}</td></tr>\n"
        
        content = f"""
        <table class="stats-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """
        self.add_section(title, content)
    
    def add_chart(self, title: str, chart_path: str):
        """Add a chart image."""
        content = f'<img src="{chart_path}" alt="{title}" class="chart">'
        self.add_section(title, content)
    
    def add_checklist(self, title: str, items: List[str]):
        """Add a checklist."""
        items_html = "\n".join(f"<li>{item}</li>" for item in items)
        content = f"<ul class='checklist'>{items_html}</ul>"
        self.add_section(title, content)
    
    def generate(self):
        """Generate and save HTML report."""
        sections_html = ""
        for section in self.sections:
            sections_html += f"""
            <section>
                <h2>{section['title']}</h2>
                {section['content']}
            </section>
            """
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Audit Report - {self.info.get('name', 'Unknown')}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        header {{
            background: #2E86AB;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        header h1 {{ margin: 0 0 10px 0; }}
        header p {{ margin: 5px 0; opacity: 0.9; }}
        section {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #2E86AB;
            border-bottom: 2px solid #A8DADC;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .stats-table th, .stats-table td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .stats-table th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .chart {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .checklist {{
            list-style: none;
            padding: 0;
        }}
        .checklist li {{
            padding: 8px 0 8px 30px;
            position: relative;
        }}
        .checklist li:before {{
            content: "☐";
            position: absolute;
            left: 0;
            color: #2E86AB;
        }}
        .anomaly-high {{ color: #C73E1D; font-weight: bold; }}
        .anomaly-low {{ color: #457B9D; font-weight: bold; }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Energy Audit Report</h1>
        <p><strong>Customer:</strong> {self.info.get('name', 'N/A')}</p>
        <p><strong>Account:</strong> {self.info.get('account', 'N/A')}</p>
        <p><strong>Address:</strong> {self.info.get('address', 'N/A')}, {self.info.get('city', '')}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </header>
    
    {sections_html}
    
    <footer>
        GRU Energy Audit Analyzer v2.0 | Generated automatically
    </footer>
</body>
</html>"""
        
        with open(self.output_path, 'w') as f:
            f.write(html)
        
        log(f"Saved HTML report: {self.output_path}", "ok")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def print_header(text: str, char: str = "═"):
    """Print formatted header."""
    print(f"\n{char * 65}")
    print(f"  {text}")
    print(f"{char * 65}")


def print_section(text: str):
    """Print section header."""
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print(f"{'─' * 50}")


def print_stat(label: str, value, unit: str = ""):
    """Print formatted statistic."""
    if value is None:
        print(f"  {label:.<32} N/A")
    elif isinstance(value, float):
        print(f"  {label:.<32} {value:,.2f} {unit}")
    else:
        print(f"  {label:.<32} {value} {unit}")


def generate_report(
    meter_file: str,
    ami_file: Optional[str] = None,
    save_charts: bool = False,
    generate_html: bool = False,
    generate_pdf: bool = False,
):
    """Generate comprehensive pre-survey report."""
    
    print_header("GRU ENERGY AUDIT — PRE-SURVEY REPORT v2.0")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Meter File: {os.path.basename(meter_file)}")
    if ami_file:
        print(f"  AMI File: {os.path.basename(ami_file)}")
    
    output_dir = Path(meter_file).parent
    
    # Load meter data
    print_section("Loading Data")
    loader = MeterLoader(meter_file)
    df_all = loader.load()
    customer = loader.get_customer_info()
    
    print(f"  Divisions: {df_all['division'].unique().tolist()}")
    
    # Customer info
    print_header("CUSTOMER INFORMATION", "─")
    print(f"  Name    : {customer.get('name', 'N/A')}")
    print(f"  Account : {customer.get('account', 'N/A')}")
    print(f"  Address : {customer.get('address', 'N/A')}")
    print(f"            {customer.get('city', '')}")
    print(f"  Own/Rent: {customer.get('own_rent', 'N/A')}")
    
    # Setup generators
    chart_dir = output_dir if save_charts else None
    charts = ChartGenerator(customer, chart_dir)
    
    # Temperature data
    print_section("Weather Data")
    temp_cache = TemperatureCache()
    start = df_all["mr_date"].min() - pd.Timedelta(days=35)
    end = df_all["mr_date"].max()
    df_temp = temp_cache.get_or_fetch(start, end)
    
    weather_detector = WeatherAnomalyDetector(df_temp) if df_temp is not None else None
    
    # HTML report setup
    html_report = None
    if generate_html:
        html_path = output_dir / f"{customer.get('account', 'report')}_report.html"
        html_report = HTMLReportGenerator(customer, str(html_path))
    
    # Analyze each division
    divisions = ["Electricity", "Water", "Gas"]
    recommendations = []
    
    for div_name in divisions:
        df_div = loader.get_division(div_name)
        
        if df_div.empty:
            continue
        
        print_header(f"{div_name.upper()} ANALYSIS", "─")
        
        # Basic features
        feats = MeterFeatures(df_div).compute()
        
        print_stat("Total Reads", feats["n_reads"])
        print_stat("Total Consumption", feats["total"], feats["unit"])
        print_stat("Daily Average", feats["daily_avg"], f"{feats['unit']}/day")
        print_stat("Peak Period", feats["peak"], feats["unit"])
        print_stat("Base Load (P5)", feats["base"], feats["unit"])
        print_stat("Data Quality", feats["quality_score"], "/100")
        
        # Anomalies
        if feats["n_anomalies"] > 0:
            print(f"\n  ⚠ ISOLATION FOREST ANOMALIES: {feats['n_anomalies']}")
            recommendations.append(f"□ {div_name}: {feats['n_anomalies']} anomaly periods detected")
        
        charts.plot_consumption_history(feats, div_name)
        
        # Weather-normalized analysis
        if weather_detector:
            print(f"\n  Weather-Normalized Analysis:")
            anomaly_result = weather_detector.analyze(df_div, div_name)
            
            if not anomaly_result["df"].empty:
                summary = anomaly_result["summary"]
                print_stat("  High Anomalies", summary["n_anomaly_high"])
                print_stat("  Low Anomalies", summary["n_anomaly_low"])
                print_stat("  Persistent High", summary["n_persistent_high"])
                print_stat("  Persistent Low", summary["n_persistent_low"])
                print_stat("  Latest Z-Score", summary["latest_z"])
                
                if summary["n_persistent_high"] > 0:
                    recommendations.append(
                        f"□ {div_name}: Persistent high usage — check equipment efficiency"
                    )
                
                charts.plot_weather_anomaly(anomaly_result, div_name)
                
                if html_report:
                    html_report.add_stats_table(
                        f"{div_name} Weather-Normalized Analysis",
                        summary
                    )
        
        # Change-point detection
        change = detect_change_point(df_div)
        if change:
            print(f"\n  ⚠ USAGE REGIME CHANGE DETECTED")
            print(f"    Date: {change['change_date'].strftime('%Y-%m-%d')}")
            print(f"    Direction: {change['direction']}")
            print(f"    Change: {change['pct_change']:+.1f}%")
            recommendations.append(
                f"□ {div_name}: Usage changed {change['pct_change']:+.0f}% "
                f"around {change['change_date'].strftime('%Y-%m')}"
            )
        
        if html_report:
            html_report.add_stats_table(f"{div_name} Summary", {
                "Total Consumption": f"{feats['total']:,.0f} {feats['unit']}",
                "Daily Average": f"{feats['daily_avg']:.2f} {feats['unit']}/day",
                "Peak Period": f"{feats['peak']:,.0f} {feats['unit']}",
                "Anomalies": feats['n_anomalies'],
                "Data Quality": f"{feats['quality_score']}/100",
            })
    
    # AMI Analysis
    if ami_file and os.path.exists(ami_file):
        print_header("AMI INTERVAL ANALYSIS", "─")
        
        try:
            ami_loader = AMILoader(ami_file)
            df_ami = ami_loader.load()
            ami_feats = AMIFeatures(df_ami).compute()
            
            print_stat("Interval", ami_feats["interval_min"], "minutes")
            print_stat("Base Load", ami_feats["base_kw"], "kW")
            print_stat("Peak Demand", ami_feats["peak_kw"], "kW")
            print_stat("Average Demand", ami_feats["avg_kw"], "kW")
            print_stat("Load Factor", ami_feats["load_factor"] * 100, "%")
            print_stat("Daily Average", ami_feats["daily_avg"], f"{ami_loader.unit}/day")
            print_stat("Peak Hour", f"{ami_feats['peak_hour']}:00")
            print_stat("TOU Ratio (On/Off Peak)", ami_feats["tou_ratio"])
            
            if ami_feats["load_factor"] < 0.3:
                recommendations.append("□ Low load factor (<30%) — peaky demand pattern")
            if ami_feats["tou_ratio"] > 2:
                recommendations.append("□ High on-peak usage — consider TOU rate optimization")
            
            charts.plot_ami_profile(ami_feats, ami_loader.unit)
            
            if html_report:
                html_report.add_stats_table("AMI Analysis", {
                    "Base Load": f"{ami_feats['base_kw']:.2f} kW",
                    "Peak Demand": f"{ami_feats['peak_kw']:.2f} kW",
                    "Load Factor": f"{ami_feats['load_factor']:.1%}",
                    "Peak Hour": f"{ami_feats['peak_hour']}:00",
                    "TOU Ratio": f"{ami_feats['tou_ratio']:.2f}",
                })
            
        except Exception as e:
            log(f"AMI analysis error: {e}", "error")
    
    # Recommendations
    print_header("PRE-SURVEY CHECKLIST", "═")
    
    # Add standard items
    recommendations.extend([
        "□ Verify thermostat type and settings",
        "□ Check air filter condition",
        "□ Inspect windows and doors for air leaks",
        "□ Document appliance ages and conditions",
        "□ Review lighting (LED opportunities)",
    ])
    
    for rec in recommendations:
        print(f"  {rec}")
    
    # Generate reports
    if html_report:
        html_report.add_checklist("Pre-Survey Checklist", recommendations)
        html_report.generate()
    
    if generate_pdf:
        pdf_path = output_dir / f"{customer.get('account', 'report')}_report.pdf"
        charts.save_pdf(str(pdf_path))
    
    print_header("END OF REPORT", "═")
    
    if save_charts:
        print(f"\n  Charts saved to: {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GRU Energy Audit Pre-Survey Report Tool v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gru_audit_v2.py customer.xlsx
  python gru_audit_v2.py customer.xlsx --ami ami_data.xlsx
  python gru_audit_v2.py customer.xlsx --save --html --pdf
  python gru_audit_v2.py customer.xlsx -v --cache-dir ./my_cache

Features:
  - Weather-normalized anomaly detection (HDD/CDD)
  - Change-point detection (CUSUM)
  - Temperature data caching
  - Load factor analysis for AMI
  - HTML and PDF report generation
        """
    )
    
    parser.add_argument("meter_file", help="Customer meter reading Excel file")
    parser.add_argument("--ami", "-a", dest="ami_file", help="AMI interval data file")
    parser.add_argument("--save", "-s", action="store_true", help="Save charts to disk")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--cache-dir", default="./temp_cache", help="Temperature cache directory")
    
    args = parser.parse_args()
    
    # Update config
    cfg.verbose = args.verbose
    cfg.cache_dir = args.cache_dir
    
    # Validate files
    if not os.path.exists(args.meter_file):
        print(f"Error: File not found: {args.meter_file}")
        sys.exit(1)
    
    if args.ami_file and not os.path.exists(args.ami_file):
        print(f"Warning: AMI file not found: {args.ami_file}")
        args.ami_file = None
    
    # Run
    try:
        generate_report(
            meter_file=args.meter_file,
            ami_file=args.ami_file,
            save_charts=args.save,
            generate_html=args.html,
            generate_pdf=args.pdf,
        )
    except ValueError as e:
        print(f"\n  ⚠ Error: {e}")
        sys.exit(1)
    except Exception as e:
        if cfg.verbose:
            import traceback
            traceback.print_exc()
        print(f"\n  ⚠ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
