# Utility Consumption Analysis

A web app that analyzes utility billing and interval data to surface consumption patterns, flag anomalies, and generate actionable recommendations. Built for internal use by efficiency and account teams.

---

## What It Does

Upload a customer's billing file, AMI interval file, or both. The app automatically pulls local temperature data, overlays it on consumption charts, checks for cross-utility patterns, and writes out plain-language findings. A PDF report can be exported for the customer at the end.

It handles Electric, Water, and Gas in any combination. Tabs appear dynamically based on what data you upload -- if there is no gas data, there is no gas tab.

---

## How to Access


The app runs in a browser. No installation needed. Open the Streamlit link and upload your files from the sidebar.

---

## What to Upload

### Meter Billing File (Excel)

Needs a sheet with "Consumption" in its name. An optional "Master Sheet" tab will pull in customer name, account number, address, and survey date automatically.

Required columns in the Consumption sheet:

| Column | Description |
|---|---|
| Division | Electricity, Water, or Gas |
| MR Date | Meter read date |
| Days | Days in billing period |
| Consumption | Usage for period |
| Avg. | Daily average (optional but recommended) |
| MR Reason | Read reason code (optional) |
| MR Unit | Unit of measure |

Read reason codes the app handles:

- 3: Non-read, filtered out
- 6: Move-In, shown as a dashed line on charts
- 21, 22: Meter Change, shown as a shaded band

### AMI Interval File (Excel)

Four timestamp formats are supported. The app detects the format automatically. Timezone suffixes like EST and EDT are stripped. Multi-utility files can use separate sheets named ELECTRIC, WATER, or GAS.

---

## What You Get

### Overview Tab

This is the main summary. It shows:

- Temperature overlay charts for each utility (color-coded bars for hot, mild, and cold periods)
- Correlation scatter plots showing how closely usage tracks temperature
- Cross-utility correlation matrix and scatter plots
- Auto-generated recommendations based on all of the above

### Utility Tabs

Each utility gets its own tab.

Meter data shows a daily average chart, a billing period consumption chart, a rolling average trend, and anomaly detection.

AMI data additionally shows the full load shape, daily totals, an hourly usage profile, and load factor.

### Advanced Analysis Tab (AMI only)

Runs a fractal complexity analysis using the Hurst Exponent. Helps characterize whether a customer's usage is consistent and predictable or highly variable.

### Export Tab

Generates a customer-facing PDF with a cover page, consumption charts, and temperature overlays. One click to download.

---

## How Recommendations Work

The app generates recommendations in four areas:

**Temperature correlation** -- how strongly usage tracks outdoor temperature. A high correlation for electric or water points to HVAC dependence. A strong negative correlation for gas indicates furnace-driven heating.

**Cross-utility correlation** -- relationships between utilities. A high Water-Electric correlation may indicate an electric water heater issue or a hot water leak. A strong inverse Electric-Gas correlation is typical seasonal HVAC switching and is considered normal.

**Load factor** -- applies to AMI data. A low load factor means demand is peaky, which can point toward load-shifting opportunities.

**Base load** -- applies to AMI electric data. A high base load (above 1 kW) suggests significant always-on equipment.

---

## Troubleshooting

**"No consumption sheet found"** -- The meter file needs a sheet with "Consumption" in its name.

**Temperature data not loading** -- Requires internet access. Data is cached for one hour. Charts will still render without the overlay if the API is unavailable.

**Cross-utility correlation not showing** -- Needs at least two utilities with overlapping date ranges and five or more shared data points.

**Recent months not graphing** -- Known issue.


 The runtime is pinned to Python 3.11 in `runtime.txt`. The main file is `gru_audit_v2.py`.
