import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.ensemble import IsolationForest
import requests
import io
import re
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

st.set_page_config(
page_title="Energy Audit Analyzer",
page_icon="E",
layout="wide",
initial_sidebar_state="expanded"
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def get_theme_colors():
    if st.session_state.dark_mode:
        return {
        "bg": "#1a1a2e",
        "card_bg": "#16213e",
        "text": "#eaeaea",
        "text_secondary": "#a0a0a0",
        "primary": "#4fc3f7",
        "secondary": "#81d4fa",
        "accent": "#ffb74d",
        "success": "#81c784",
        "warning": "#ffb74d",
        "danger": "#e57373",
        "chart_bg": "#1a1a2e",
        "grid": "#333355",
        }
    else:
        return {
        "bg": "#f8f9fa",
        "card_bg": "#ffffff",
        "text": "#1e3a5f",
        "text_secondary": "#6c757d",
        "primary": "#1e3a5f",
        "secondary": "#3498db",
        "accent": "#e67e22",
        "success": "#27ae60",
        "warning": "#f39c12",
        "danger": "#c0392b",
        "chart_bg": "#ffffff",
        "grid": "#e0e0e0",
        }

def apply_theme():
    colors = get_theme_colors()

    st.markdown(f"""
<style>
    .stApp {{
        background-color: {colors["bg"]};
    }}
    [data-testid="stSidebar"] {{
        background-color: {colors["card_bg"]};
    }}
    h1, h2, h3 {{
        color: {colors["text"]};
    }}
    p, span, label {{
        color: {colors["text"]};
    }}
    .info-box {{
        background-color: {"#1e3a5f22" if st.session_state.dark_mode else "#e8f4fd"};
        border-left: 4px solid {colors["secondary"]};
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        color: {colors["text"]};
    }}
</style>
""", unsafe_allow_html=True)