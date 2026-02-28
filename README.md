# ⚡ GRU Energy Audit Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful tool for energy auditors to analyze customer utility data before conducting on-site surveys. Available as both a **command-line tool** and a **web application**.

![App Screenshot](docs/screenshot.png)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Weather-Normalized Analysis** | HDD/CDD regression with Huber estimator |
| **Multi-Utility Support** | Electric, Water, Gas models |
| **Anomaly Detection** | IsolationForest + Z-score flagging |
| **Change-Point Detection** | CUSUM algorithm for regime shifts |
| **AMI Analysis** | Load factor, TOU ratios, peak demand |
| **Interactive Web App** | Streamlit-powered interface |
| **Report Generation** | HTML, PDF, and summary exports |

---

## 🚀 Quick Start

### Option 1: Web App (Streamlit Cloud)

**Try it now:** [https://your-app.streamlit.app](https://your-app.streamlit.app)

No installation required — just upload your files and analyze!

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/your-org/gru-energy-audit.git
cd gru-energy-audit

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run streamlit_app.py

# Or use the CLI
python gru_audit_v2.py customer.xlsx --save --html
```

---

## 📦 Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
requests>=2.25.0
openpyxl>=3.0.0
streamlit>=1.28.0
```

---

## 🖥️ Usage

### Web Application

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- Drag-and-drop file upload
- Interactive charts
- Adjustable analysis parameters
- Pre-survey checklist
- Downloadable reports

### Command Line Interface

```bash
# Basic analysis
python gru_audit_v2.py customer.xlsx

# With AMI data
python gru_audit_v2.py customer.xlsx --ami ami_data.xlsx

# Full report suite
python gru_audit_v2.py customer.xlsx --save --html --pdf -v

# Custom cache directory
python gru_audit_v2.py customer.xlsx --cache-dir ./shared_cache
```

#### CLI Options

| Option | Description |
|--------|-------------|
| `--ami`, `-a` | AMI interval data file |
| `--save`, `-s` | Save charts as PNG |
| `--html` | Generate HTML report |
| `--pdf` | Generate PDF report |
| `--verbose`, `-v` | Debug logging |
| `--cache-dir` | Temperature cache directory |

---

## ☁️ Deploy to Streamlit Cloud

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/your-username/gru-energy-audit.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repository
4. Set the main file path: `streamlit_app.py`
5. Click **"Deploy"**

Your app will be live at `https://your-app.streamlit.app` in minutes!

### Step 3: Custom Domain (Optional)

In Streamlit Cloud settings, add your custom domain under **"Custom subdomain"**.

---

## 📊 File Format Requirements

### Meter Reading File (Required)

GRU-format Excel file with:

| Sheet | Contents |
|-------|----------|
| **Master Sheet** | Customer info (account, name, address) |
| **Consumption History** | Meter reading data |

**Consumption History Columns:**

| Column | Description |
|--------|-------------|
| Division | Electricity, Water, or Gas |
| MR Date | Meter read date |
| Days | Billing period length |
| Consumption | Usage for period |
| MR Unit | kWh, kGal, CCF |

### AMI File (Optional)

15-minute interval data:
- Rows 1-4: Metadata (skipped)
- Column A: Timestamp (e.g., "Jan 15, 2025 - 2:30 PM EST")
- Column B: Value (Wh for electric)

---

## 🔧 Configuration

### Analysis Parameters

Edit in the web app sidebar or in `gru_audit_v2.py`:

```python
@dataclass
class Config:
    # Location (Gainesville, FL)
    latitude: float = 29.6516
    longitude: float = -82.3248
    
    # Anomaly Detection
    residual_z_threshold: float = 2.5   # Higher = fewer flags
    min_history_periods: int = 8        # Min billing periods
    persistence_periods: int = 2        # Consecutive for "persistent"
    
    # Temperature
    comfort_baseline: float = 65.0      # HDD/CDD base temp
    
    # Change-Point Detection
    cusum_threshold: float = 4.0
```

### Streamlit Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#2E86AB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

---

## 📁 Project Structure

```
gru-energy-audit/
├── streamlit_app.py      # Web application
├── gru_audit_v2.py       # Core library + CLI
├── requirements.txt      # Dependencies
├── test_core.py          # Unit tests
├── README.md             # Documentation
├── LICENSE               # MIT License
├── .gitignore            # Git ignore rules
└── .streamlit/
    └── config.toml       # Streamlit config
```

---

## 🧪 Testing

```bash
# Run all tests
python test_core.py

# With pytest (if installed)
pytest test_core.py -v
```

**Test Coverage:**
- Configuration defaults
- HDD/CDD calculations
- Meter feature computation
- Weather-normalized analysis
- Change-point detection
- AMI features
- Temperature caching

---

## 📈 Example Output

### Web App

The web app provides:
- **Tabbed interface** for each utility type
- **Interactive metrics** with color-coded anomalies
- **Zoomable charts** for consumption and anomaly analysis
- **Downloadable checklist** for on-site surveys

### CLI Report

```
═══════════════════════════════════════════════════════════════
  GRU ENERGY AUDIT — PRE-SURVEY REPORT v2.0
═══════════════════════════════════════════════════════════════
  Customer: JOHN SMITH (250007081590)
  Address:  1234 NW 45TH ST, Gainesville FL

─── ELECTRICITY ANALYSIS ───────────────────────────────────────
  Daily Average.................. 18.42 kWh/day
  Data Quality................... 95 /100

  Weather-Normalized Analysis:
    High Anomalies............... 3
    Persistent High.............. 1
    Latest Z-Score............... 1.24

  ⚠ USAGE REGIME CHANGE DETECTED
    Date: 2024-06-15 | Change: +28.3%

═══════════════════════════════════════════════════════════════
  PRE-SURVEY CHECKLIST
═══════════════════════════════════════════════════════════════
  □ Electricity: Persistent high — check equipment
  □ Verify thermostat settings
  □ Check air filter condition
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Write tests for new functionality
4. Commit changes: `git commit -am 'Add new feature'`
5. Push to branch: `git push origin feature/new-feature`
6. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Weather data: [Open-Meteo API](https://open-meteo.com/)
- UI framework: [Streamlit](https://streamlit.io/)
- Anomaly detection: [scikit-learn](https://scikit-learn.org/)

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/your-org/gru-energy-audit/issues)
- **Email:** energy-audit@gru.com

---

<p align="center">
  Made with ⚡ by the GRU Energy Audit Team
</p>
