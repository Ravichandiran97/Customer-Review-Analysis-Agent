# pyrefly: ignore [missing-import]
import pandas as pd
import streamlit as st
import io

def validate_and_clean_csv(df):
    """
    Validates that the uploaded dataframe contains 'Rating' and 'Review' columns.
    Cleans blank reviews and normalizes columns.
    Returns (cleaned_df, error_message).
    """
    # Normalize columns to title case / strip whitespace
    df.columns = [str(col).strip().title() for col in df.columns]
    
    required = ["Rating", "Review"]
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        return None, f"Missing required column(s): {', '.join(missing)}. The file must contain both 'Rating' and 'Review'."
    
    # Drop rows where 'Review' or 'Rating' is completely null
    df = df.dropna(subset=["Review", "Rating"])
    
    # Clean blank reviews
    df = df[df["Review"].astype(str).str.strip() != ""]
    
    # Force Rating to numeric and clean
    try:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        # Drop invalid ratings
        df = df.dropna(subset=["Rating"])
        # Ensure ratings are in valid range (usually 1 to 5)
        df = df[(df["Rating"] >= 1) & (df["Rating"] <= 5)]
        df["Rating"] = df["Rating"].astype(int)
    except Exception as e:
        return None, f"Error processing 'Rating' column: {str(e)}"
        
    if len(df) == 0:
        return None, "The uploaded file does not contain any valid reviews after cleaning blank rows."
        
    return df, None

def get_sample_csv_bytes():
    """
    Returns a sample CSV file as bytes for downloading.
    """
    sample_data = """Rating,Review
5,The battery life on this laptop is absolutely incredible! Lasts over 12 hours on a single charge. Highly recommend it.
1,Terrible experience. The packaging was completely crushed when it arrived and the screen was cracked. Returning immediately.
4,Great product quality for the price. The delivery was a bit slow though taking almost a week to arrive.
2,Very disappointed with the sound quality. Also the battery dies in less than 2 hours. Not worth the money.
5,Outstanding customer service and super fast delivery. The package was secure and the item works perfectly.
3,The price is reasonable but the build quality feels a bit cheap and plasticky. It does the job but don't expect premium materials.
"""
    return sample_data.encode('utf-8')

def apply_custom_css():
    """
    Applies custom CSS for a premium, modern dashboard design
    utilizing an Emerald Green, Dark Green, Gold, Black, and White color theme.
    """
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
        
        /* Global Fade-In and Slide-Up Animation */
        @keyframes fadeInSlideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeEffect {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .stApp {
            background-color: #F8FAF8 !important; /* Premium light green-gray background */
            animation: fadeEffect 0.8s ease-out forwards;
        }
        
        /* Typography System */
        html, body, .stText, .stMarkdown, p, li, label, .stWidgetLabel {
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
            font-size: 15px !important; /* Premium body text size */
            color: #121212 !important; /* Premium Black */
        }
        
        /* Restore Streamlit icon font-family */
        [data-testid="collapsedControl"] span, 
        [data-testid="stSidebarCollapsedControl"] span,
        [class*="stSidebarCollapseButton"] span,
        [aria-label="Close sidebar"] span,
        [aria-label="Open sidebar"] span,
        .material-icons, 
        [class*="material-symbols"] {
            font-family: "Material Symbols Outlined", "Material Icons" !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif !important;
            font-weight: 700 !important;
            color: #121212 !important;
        }
        
        /* Sidebar container override */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #056839 0%, #022b17 100%) !important;
            border-right: 1px solid rgba(212, 175, 55, 0.2) !important;
            box-shadow: 4px 0 15px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Sidebar headers & navigation title */
        .sidebar-section-heading {
            color: #D4AF37 !important; /* Gold */
            font-family: 'Poppins', sans-serif !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.2px !important;
            margin-top: 24px !important;
            margin-bottom: 12px !important;
            padding-left: 6px !important;
            border-left: 3px solid #D4AF37 !important;
        }
        
        /* Sidebar Text Overrides */
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div {
            color: #FFFFFF !important;
        }
        
        /* Sidebar Navigation Button Styles */
        [data-testid="stSidebar"] button:not([data-testid="collapsedControl"]):not([data-testid="collapsedControl"] button):not([data-testid*="CollapsedControl"]):not([aria-label="Close"]):not([aria-label="Close sidebar"]):not([class*="stSidebarCollapseButton"]) {
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            border-radius: 12px !important; /* Rounded corners */
            padding: 10px 16px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            font-family: 'Inter', sans-serif !important;
            text-align: left !important;
            width: 100% !important;
            border: 1px solid transparent !important;
            margin-bottom: 6px !important;
            height: 44px !important;
            background-color: rgba(255, 255, 255, 0.04) !important;
            color: #E2E8F0 !important;
            transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            border-left: 4px solid transparent !important;
        }
        
        /* Inactive Navigation Hover Style */
        [data-testid="stSidebar"] button[kind="secondary"]:not([data-testid="collapsedControl"]):not([data-testid="collapsedControl"] button):not([data-testid*="CollapsedControl"]):not([aria-label="Close"]):not([aria-label="Close sidebar"]):not([class*="stSidebarCollapseButton"]):hover {
            background-color: rgba(255, 255, 255, 0.12) !important;
            color: #FFFFFF !important;
            border-left: 4px solid #0B8F4D !important; /* Emerald green left border on hover */
            transform: translateX(4px) !important; /* Shift transition */
        }
        
        /* Active Navigation Button Style */
        [data-testid="stSidebar"] button[kind="primary"]:not([data-testid="collapsedControl"]):not([data-testid="collapsedControl"] button):not([data-testid*="CollapsedControl"]):not([aria-label="Close"]):not([aria-label="Close sidebar"]):not([class*="stSidebarCollapseButton"]) {
            background: linear-gradient(135deg, #0B8F4D 0%, #056839 100%) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border-left: 4px solid #D4AF37 !important; /* Gold Left border */
            box-shadow: 0 4px 12px rgba(11, 143, 77, 0.25) !important;
            border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        [data-testid="stSidebar"] button[kind="primary"]:not([data-testid="collapsedControl"]):not([data-testid="collapsedControl"] button):not([data-testid*="CollapsedControl"]):not([aria-label="Close"]):not([aria-label="Close sidebar"]):not([class*="stSidebarCollapseButton"]):hover {
            background: linear-gradient(135deg, #0B8F4D 0%, #056839 100%) !important;
            box-shadow: 0 4px 14px rgba(212, 175, 55, 0.35) !important; /* Gold glow */
        }
        
        /* Sidebar Info Cards */
        .sidebar-info-card {
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(212, 175, 55, 0.3) !important; /* Gold accent */
            border-radius: 14px !important;
            padding: 14px !important;
            margin-top: 15px !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
        }
        
        .sidebar-info-card .info-title {
            color: #D4AF37 !important; /* Gold */
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            margin-bottom: 8px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
        }
        
        .sidebar-info-card .info-item {
            color: #E2E8F0 !important;
            font-size: 12px !important;
            margin-bottom: 6px !important;
            line-height: 1.4 !important;
        }
        
        .sidebar-info-card .info-item b {
            color: #D4AF37 !important;
        }
        
        /* Active Status Indicator */
        .status-indicator-green {
            width: 8px;
            height: 8px;
            background-color: #0B8F4D;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #0B8F4D;
            animation: pulse 1.8s infinite alternate;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(11, 143, 77, 0.6); }
            70% { transform: scale(1.1); box-shadow: 0 0 0 5px rgba(11, 143, 77, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(11, 143, 77, 0); }
        }
        
        /* Main Button Overrides (Content Area) */
        [data-testid="stAppViewContainer"] button[kind="primary"] {
            background: linear-gradient(135deg, #0B8F4D, #056839) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            padding: 10px 24px !important;
            box-shadow: 0 4px 12px rgba(11, 143, 77, 0.2) !important;
            transition: all 0.25s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            width: 100% !important;
            cursor: pointer;
        }
        
        [data-testid="stAppViewContainer"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #056839, #022b17) !important;
            box-shadow: 0 0 15px rgba(212, 175, 55, 0.45) !important; /* Gold hover glow */
            transform: scale(1.02) !important; /* Scale animation */
        }
        
        [data-testid="stAppViewContainer"] button[kind="secondary"] {
            background-color: #FFFFFF !important;
            color: #0B8F4D !important;
            font-weight: 600 !important;
            font-size: 15px !important;
            border-radius: 12px !important;
            border: 2px solid #0B8F4D !important;
            padding: 9px 24px !important;
            transition: all 0.25s ease !important;
            width: 100% !important;
        }
        
        [data-testid="stAppViewContainer"] button[kind="secondary"]:hover {
            background-color: #0B8F4D !important;
            color: #FFFFFF !important;
            transform: scale(1.01) !important;
        }
        
        /* Drag & Drop File Uploader Zone */
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF !important;
            border: 2px dashed #0B8F4D !important; /* Dashed green border */
            border-radius: 20px !important;
            padding: 24px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important;
            transition: all 0.25s ease !important;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: #D4AF37 !important; /* Gold on hover */
            background-color: #F8FAF8 !important;
            box-shadow: 0 6px 16px rgba(11, 143, 77, 0.08) !important;
        }
        
        [data-testid="stFileUploader"] button {
            background-color: #0B8F4D !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
        }
        
        /* Premium KPI Cards */
        .kpi-card {
            background-color: #FFFFFF !important;
            border: 1px solid rgba(212, 175, 55, 0.3) !important; /* Thin gold border */
            border-radius: 20px !important; /* 20px rounded corners */
            padding: 22px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04) !important; /* Soft shadow */
            display: flex !important;
            align-items: center !important;
            gap: 16px !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            margin-bottom: 20px !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .kpi-card::after {
            content: '' !important;
            position: absolute !important;
            top: 0 !important; left: 0 !important; width: 100% !important; height: 100% !important;
            background: linear-gradient(135deg, rgba(212, 175, 55, 0.03), transparent) !important;
            pointer-events: none !important;
        }
        
        .kpi-card:hover {
            transform: translateY(-5px) scale(1.02) !important; /* Slight zoom & hover lift */
            box-shadow: 0 10px 25px rgba(11, 143, 77, 0.1) !important; /* Glow on hover */
            border-color: #D4AF37 !important;
        }
        
        .kpi-icon-container {
            background-color: rgba(11, 143, 77, 0.08) !important;
            width: 50px !important;
            height: 50px !important;
            border-radius: 12px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border: 1px solid rgba(11, 143, 77, 0.15) !important;
        }
        
        .kpi-icon {
            font-size: 24px !important;
        }
        
        .kpi-content {
            display: flex !important;
            flex-direction: column !important;
        }
        
        .kpi-title {
            font-family: 'Inter', sans-serif !important;
            font-size: 12px !important;
            color: #666666 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
            font-weight: 600 !important;
            margin-bottom: 2px !important;
        }
        
        .kpi-value {
            font-family: 'Poppins', sans-serif !important;
            font-size: 26px !important;
            font-weight: 800 !important;
            color: #121212 !important;
            line-height: 1.2 !important;
        }
        
        .kpi-subtitle {
            font-size: 11px !important;
            color: #888888 !important;
            margin-top: 2px !important;
        }
        
        /* Grand Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #056839 0%, #022b17 100%) !important;
            border: 1px solid rgba(212, 175, 55, 0.35) !important;
            border-radius: 24px !important;
            padding: 45px 30px !important;
            text-align: center !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12) !important;
            position: relative !important;
            overflow: hidden !important;
            margin-bottom: 30px !important;
            animation: fadeInSlideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        
        .hero-banner::before {
            content: "" !important;
            position: absolute !important;
            top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(circle at 80% 20%, rgba(212, 175, 55, 0.08), transparent 50%),
                        radial-gradient(circle at 20% 80%, rgba(11, 143, 77, 0.15), transparent 50%) !important;
            pointer-events: none !important;
        }
        
        .hero-icon-container {
            margin-bottom: 18px !important;
            animation: float 4s ease-in-out infinite;
        }
        
        .hero-title {
            font-family: 'Poppins', sans-serif !important;
            font-size: 58px !important; /* Grand title: 58px */
            font-weight: 900 !important;
            color: #FFFFFF !important;
            margin: 0 0 12px 0 !important;
            line-height: 1.15 !important;
            letter-spacing: -1.2px !important;
            text-shadow: 0 2px 8px rgba(0,0,0,0.4) !important;
        }
        
        .hero-gold-line {
            width: 140px !important;
            height: 3.5px !important;
            background: linear-gradient(90deg, transparent, #D4AF37, transparent) !important;
            margin: 0 auto 16px auto !important;
            border-radius: 2px !important;
        }
        
        .hero-subtitle {
            font-family: 'Inter', sans-serif !important;
            font-size: 19px !important;
            font-weight: 500 !important;
            color: #D4AF37 !important; /* Gold */
            margin: 0 0 16px 0 !important;
            max-width: 800px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            line-height: 1.4 !important;
            letter-spacing: 0.2px !important;
        }
        
        .hero-welcome {
            font-family: 'Inter', sans-serif !important;
            font-size: 14.5px !important;
            color: #E2E8F0 !important;
            max-width: 720px !important;
            margin: 0 auto !important;
            line-height: 1.6 !important;
        }
        
        .hero-bg-shapes {
            position: absolute !important;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none !important;
        }
        
        .floating-shape {
            position: absolute !important;
            font-size: 24px !important;
            opacity: 0.12 !important;
            animation: float 6s ease-in-out infinite;
        }
        
        .shape1 { top: 15%; left: 8%; animation-delay: 0s; }
        .shape2 { top: 60%; right: 8%; animation-delay: 2s; }
        .shape3 { bottom: 15%; left: 18%; animation-delay: 4s; }
        
        @keyframes float {
            0% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-8px) rotate(4deg); }
            100% { transform: translateY(0px) rotate(0deg); }
        }
        
        /* Dashboard Card Container */
        .metric-card, .summary-box, .rec-card, .feature-card, .ai-analysis-container {
            background-color: #FFFFFF !important;
            border: 1px solid rgba(212, 175, 55, 0.25) !important; /* Thin Gold border */
            border-radius: 20px !important; /* 20px rounded corners */
            padding: 24px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            margin-bottom: 24px !important;
        }
        
        .metric-card:hover, .summary-box:hover, .rec-card:hover, .feature-card:hover, .ai-analysis-container:hover {
            transform: translateY(-5px) scale(1.01) !important; /* Slight zoom and lift */
            box-shadow: 0 12px 28px rgba(11, 143, 77, 0.09) !important; /* Glowing hover */
            border-color: #D4AF37 !important; /* Highlight gold border */
        }
        
        .summary-box h4, .ai-analysis-header {
            margin-top: 0 !important;
            margin-bottom: 12px !important;
            color: #121212 !important;
            font-family: 'Poppins', sans-serif !important;
            font-size: 20px !important;
            font-weight: 600 !important;
            border-bottom: 1.5px solid #F3F4F6 !important;
            padding-bottom: 6px !important;
        }
        
        /* Premium Styled Tables */
        table {
            border-collapse: separate !important;
            border-spacing: 0 !important;
            width: 100% !important;
            border-radius: 14px !important;
            overflow: hidden !important;
            font-family: 'Inter', sans-serif !important;
            border: 1px solid rgba(212, 175, 55, 0.25) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
        }
        
        th {
            background-color: #D4AF37 !important; /* Gold header */
            color: #121212 !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 14px 18px !important;
            font-size: 14.5px !important;
            position: sticky !important;
            top: 0 !important;
            z-index: 10 !important;
        }
        
        td {
            padding: 12px 18px !important;
            border-bottom: 1px solid #E5E7EB !important;
            font-size: 14px !important;
            background-color: #FFFFFF !important;
            color: #121212 !important;
        }
        
        tr:last-child td {
            border-bottom: none !important;
        }
        
        tr:nth-child(even) td {
            background-color: #EBF7EE !important; /* Light green alternating row */
        }
        
        tr:hover td {
            background-color: #F3FAF5 !important; /* Delicate green tint on hover */
            cursor: default;
        }
        
        /* Tabs Overrides */
        .stTabs [data-baseweb="tab-list"] {
            gap: 16px !important;
            border-bottom: 2px solid #E2E8F0 !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: #64748B !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            padding: 10px 14px !important;
            transition: all 0.2s ease !important;
            border-bottom: 2px solid transparent !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: #0B8F4D !important; /* Emerald green */
            border-bottom-color: #0B8F4D !important;
            font-weight: 700 !important;
        }
        
        /* Input & Controls */
        div[data-baseweb="input"], div[data-baseweb="select"] {
            border-radius: 10px !important;
            border: 1px solid #CBD5E1 !important;
            transition: all 0.25s ease !important;
        }
        
        div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
            border-color: #D4AF37 !important;
            box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.15) !important;
        }
        
        /* Footer divider */
        .footer-divider {
            margin-top: 40px !important;
            margin-bottom: 20px !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent, #D4AF37, transparent) !important;
        }
        
        .footer-text {
            text-align: center !important;
            font-size: 12.5px !important;
            color: #666666 !important;
            line-height: 1.6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_premium_kpi(icon, title, value, subtitle="", border_color="#D4AF37"):
    """
    Renders a premium, interactive KPI dashboard card with 20px rounded corners,
    gold/green border styles, soft shadows, hover transitions, and icons.
    """
    card_html = f"""
    <div class="kpi-card" style="border-left: 4px solid {border_color};">
        <div class="kpi-icon-container">
            <span class="kpi-icon">{icon}</span>
        </div>
        <div class="kpi-content">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            {f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_metric_card(title, value, metric_type="total"):
    """
    Backwards compatibility wrapper redirecting to render_premium_kpi.
    """
    icon_map = {
        "total": "📋",
        "pos": "🟢",
        "neu": "🟡",
        "neg": "🔴",
        "avg": "⭐"
    }
    color_map = {
        "total": "#121212",
        "pos": "#0B8F4D",
        "neu": "#D4AF37",
        "neg": "#cc3333",
        "avg": "#056839"
    }
    icon = icon_map.get(metric_type, "📊")
    color = color_map.get(metric_type, "#D4AF37")
    render_premium_kpi(icon, title, value, "", color)

def inject_page_theme(primary, secondary, bg_light, border):
    """
    Injects CSS variable overrides for page-specific theme accents.
    """
    theme_css = f"""
    <style>
    :root {{
        --theme-primary: {primary};
        --theme-secondary: {secondary};
        --theme-bg-light: {bg_light};
        --theme-border: {border};
        --theme-gradient: linear-gradient(135deg, {primary}, {secondary});
    }}
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

def render_section_header(page_title, subtitle=""):
    """
    Renders a premium section sub-header with custom SVG logos, consistent typography, and accents.
    """
    logo_svg = ""
    theme_color = "#0B8F4D" # Default Primary Green
    
    if "Home" in page_title or "Welcome" in page_title:
        logo_svg = (
            '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #0B8F4D;">'
            '<circle cx="9" cy="21" r="1" fill="#0B8F4D"></circle>'
            '<circle cx="20" cy="21" r="1" fill="#0B8F4D"></circle>'
            '<path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path>'
            '</svg>'
        )
    elif "Upload" in page_title or "Review" in page_title:
        theme_color = "#0B8F4D"
        logo_svg = (
            '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #0B8F4D;">'
            '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
            '<polyline points="14 2 14 8 20 8"></polyline>'
            '<line x1="12" y1="18" x2="12" y2="12" stroke="#D4AF37" stroke-width="2"></line>'
            '<polyline points="9 15 12 12 15 15" stroke="#D4AF37" stroke-width="2"></polyline>'
            '</svg>'
        )
    elif "Insights" in page_title or "Dashboard" in page_title or "Analysis" in page_title or "Charts" in page_title:
        logo_svg = (
            '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #0B8F4D;">'
            '<line x1="18" y1="20" x2="18" y2="10"></line>'
            '<line x1="12" y1="20" x2="12" y2="4" stroke="#D4AF37" stroke-width="2"></line>'
            '<line x1="6" y1="20" x2="6" y2="14"></line>'
            '<path d="M3 20h18"></path>'
            '</svg>'
        )
    elif "Search" in page_title or "Filter" in page_title or "Lookup" in page_title:
        theme_color = "#056839"
        logo_svg = (
            '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #0B8F4D;">'
            '<circle cx="11" cy="11" r="8"></circle>'
            '<line x1="21" y1="21" x2="16.65" y2="16.65"></line>'
            '</svg>'
        )
    elif "Recommendations" in page_title:
        theme_color = "#D4AF37"
        logo_svg = (
            '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #D4AF37;">'
            '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A5 5 0 0 0 8 8c0 1 .3 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"></path>'
            '<line x1="9" y1="18" x2="15" y2="18" stroke="#0B8F4D" stroke-width="2"></line>'
            '<line x1="10" y1="22" x2="14" y2="22"></line>'
            '</svg>'
        )
    elif "Export" in page_title or "Download" in page_title:
        logo_svg = (
            '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #0B8F4D;">'
            '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>'
            '<polyline points="7 10 12 15 17 10" stroke="#D4AF37" stroke-width="2"></polyline>'
            '<line x1="12" y1="15" x2="12" y2="3"></line>'
            '</svg>'
        )
    else:
        logo_svg = (
            '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: #0B8F4D;">'
            '<circle cx="12" cy="12" r="10"></circle>'
            '<line x1="12" y1="16" x2="12" y2="12"></line>'
            '<line x1="12" y1="8" x2="12.01" y2="8"></line>'
            '</svg>'
        )

    subtitle_html = f'<p style="margin: 4px 0 0 0; color: #666666; font-size: 14.5px; font-family: \'Inter\', sans-serif;">{subtitle}</p>' if subtitle else ''
    header_html = (
        f'<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; border-bottom: 1.5px solid rgba(212, 175, 55, 0.2); padding-bottom: 12px; margin-top: 15px;">'
        f'<div style="background-color: {theme_color}12; padding: 6px; border-radius: 6px; display: flex; align-items: center; justify-content: center; border: 1px solid {theme_color}25;">'
        f'{logo_svg}'
        f'</div>'
        f'<div>'
        f'<h3 style="margin: 0; color: #121212; font-family: \'Poppins\', sans-serif; font-size: 22px; font-weight: 600;">{page_title}</h3>'
        f'{subtitle_html}'
        f'</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

def render_footer():
    """
    Renders a premium footer at the bottom of the page.
    """
    st.markdown('<div class="footer-divider"></div>', unsafe_allow_html=True)
    footer_html = """
    <div class="footer-text">
        <b>Customer Review Analysis Agent</b> — AI-Powered Customer Intelligence & Sentiment Analytics<br>
        <span style="color:#D4AF37;">Version 2.0.0 (Enterprise)</span> | Developed using Python + Streamlit + NLP (NLTK/VADER)
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


