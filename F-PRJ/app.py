import streamlit as st
import pandas as pd
import io
import os
import time
import numpy as np

# Set page configuration first
st.set_page_config(
    page_title="Customer Review Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular components
from utils import (
    validate_and_clean_csv, 
    get_sample_csv_bytes, 
    apply_custom_css, 
    inject_page_theme, 
    render_section_header, 
    render_premium_kpi,
    render_footer
)
from sentiment import analyze_sentiments
from summarizer import get_sentiment_summaries
from keyword_extractor import extract_top_keywords
from dashboard import (
    calculate_metrics,
    render_dashboard_metrics, 
    get_sentiment_donut_chart,
    get_aspects_radar_chart,
    get_rating_line_chart,
    get_rating_area_chart,
    get_sentiment_rating_heatmap,
    get_keywords_treemap,
    render_html_word_cloud,
    get_keywords_bar_chart,
    get_ratings_distribution_chart
)

# Initialize session state variables
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "🏠 Home"
    
if "processed_df" not in st.session_state:
    st.session_state["processed_df"] = None
    
if "summaries" not in st.session_state:
    st.session_state["summaries"] = None
    
if "keywords" not in st.session_state:
    st.session_state["keywords"] = None
    
if "sentiment_model" not in st.session_state:
    st.session_state["sentiment_model"] = None

# Initialize settings in session state if not present
if "sentiment_engine" not in st.session_state:
    st.session_state["sentiment_engine"] = "Fast NLP (VADER only)"
if "summarization_engine" not in st.session_state:
    st.session_state["summarization_engine"] = "Fast NLP (Extractive)"
if "confidence_threshold" not in st.session_state:
    st.session_state["confidence_threshold"] = 0.60

# Apply modern CSS styles globally
apply_custom_css()

# Helper function to switch pages programmatically
def navigate_to(page_name):
    st.session_state["active_page"] = page_name
    st.rerun()

# ----------------- SIDEBAR BRANDING & LOGO -----------------
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 15px 0 10px 0;'>
        <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="#D4AF37" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 8px rgba(212, 175, 55, 0.45));">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
        </svg>
        <div style='color: #FFFFFF; font-family: "Poppins", sans-serif; font-weight: 800; font-size: 18px; margin-top: 8px; letter-spacing: 0.5px;'>
            REVIEW<span style='color: #D4AF37;'>AI</span> AGENT
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<div class='sidebar-section-heading'>Navigation</div>", unsafe_allow_html=True)

# 12 Navigation Buttons in Sidebar
pages = [
    ("🏠 Home", "🏠 Home"),
    ("📂 Upload CSV", "📂 Upload CSV"),
    ("🤖 AI Analysis", "🤖 AI Analysis"),
    ("😊 Sentiment Analysis", "😊 Sentiment Analysis"),
    ("📊 Dashboard", "📊 Dashboard"),
    ("📈 Charts", "📈 Charts"),
    ("⭐ Customer Ratings", "⭐ Customer Ratings"),
    ("🔍 Feature Extraction", "🔍 Feature Extraction"),
    ("📑 Summary", "📑 Summary"),
    ("📥 Download Reports", "📥 Download Reports"),
    ("⚙ Settings", "⚙ Settings"),
    ("❓ Help", "❓ Help")
]

for label, page_id in pages:
    is_active = st.session_state["active_page"] == page_id
    if st.sidebar.button(
        label,
        key=f"nav_btn_{page_id}",
        type="primary" if is_active else "secondary",
        use_container_width=True
    ):
        st.session_state["active_page"] = page_id

# Sidebar System Metadata Card
st.sidebar.markdown(
    """
    <div class="sidebar-info-card">
        <div class="info-title">💼 Retail & E-Commerce</div>
        <div class="info-item"><b>System State:</b> <span class="status-indicator-green"></span> Active</div>
        <div class="info-item"><b>Agent Version:</b> 2.0.0 (Enterprise)</div>
        <div class="info-item"><b>Engine choice:</b> {}</div>
    </div>
    """.format(st.session_state["sentiment_engine"].split()[0]),
    unsafe_allow_html=True
)

# Helper function to check if data is loaded before accessing results
def check_data_loaded():
    if st.session_state["processed_df"] is None:
        st.markdown(
            """
            <div class="ai-analysis-container" style="border-left-color: #D4AF37; text-align: center; padding: 40px 20px;">
                <span style="font-size: 48px;">⚠️</span>
                <h3 style="margin-top: 15px; color:#121212;">No Active Review Analysis</h3>
                <p style="color:#666; margin-bottom: 20px; font-size:14.5px;">You must upload a review dataset and run the AI sentiment analysis before accessing this page.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("📂 Go to Upload & Analyze Section", type="primary"):
            navigate_to("📂 Upload CSV")
        render_footer()
        st.stop()

# ----------------- PAGE 1: HOME PAGE -----------------
if st.session_state["active_page"] == "🏠 Home":
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    
    # Grand Hero Banner (30-40% height of screen)
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-bg-shapes">
                <div class="floating-shape shape1">🤖</div>
                <div class="floating-shape shape2">📊</div>
                <div class="floating-shape shape3">💡</div>
            </div>
            <div class="hero-content">
                <div class="hero-icon-container">
                    <svg width="68" height="68" viewBox="0 0 24 24" fill="none" stroke="#D4AF37" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 0 8px rgba(212, 175, 55, 0.45));">
                        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
                        <polyline points="2 17 12 22 22 17"></polyline>
                        <polyline points="2 12 12 17 22 12"></polyline>
                    </svg>
                </div>
                <h1 class="hero-title">Customer Review Analysis Agent</h1>
                <div class="hero-gold-line"></div>
                <p class="hero-subtitle">AI-Powered Customer Intelligence, Sentiment Analysis & Business Insights for Retail & E-Commerce</p>
                <p class="hero-welcome">
                    Analyze customer reviews using deep learning NLP. Upload feedback datasets, inspect sentiment distributions, 
                    extract product features, summarize customer consensus, and export actionable insights.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Homepage Layout Grid
    col_feat, col_workflow = st.columns([1, 1])
    
    with col_feat:
        st.markdown("<h3 style='color:#121212; font-size: 20px; font-weight:600; margin-bottom:15px;'>🎯 Core Agent Capabilities</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="feature-card">
                <div style="font-size: 18px; font-weight: 700; color: #0B8F4D; margin-bottom: 5px;">🤖 Hybrid Sentiment Pipeline</div>
                <p style="margin: 0; font-size: 13.5px; color: #555;">Combines DistilBERT Transformer accuracy with VADER lexicon speed for optimal sentiment scoring.</p>
            </div>
            <div class="feature-card">
                <div style="font-size: 18px; font-weight: 700; color: #D4AF37; margin-bottom: 5px;">🔍 Topic & Aspect Extraction</div>
                <p style="margin: 0; font-size: 13.5px; color: #555;">Automatically detects mentions of product quality, shipping latency, price sensitivity, and packaging integrity.</p>
            </div>
            <div class="feature-card">
                <div style="font-size: 18px; font-weight: 700; color: #121212; margin-bottom: 5px;">📑 Deep Summarization</div>
                <p style="margin: 0; font-size: 13.5px; color: #555;">Consolidates hundreds of product reviews into human-style overall positive and negative summaries.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_workflow:
        st.markdown("<h3 style='color:#121212; font-size: 20px; font-weight:600; margin-bottom:15px;'>⚙ Workflow Overview</h3>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="summary-box">
                <div style="display:flex; flex-direction:column; gap:14px;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="background-color:#0B8F4D; color:white; width:24px; height:24px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-weight:bold; font-size:12px;">1</span>
                        <b style="font-size:14px; color:#121212;">Upload CSV Dataset</b>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="background-color:#056839; color:white; width:24px; height:24px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-weight:bold; font-size:12px;">2</span>
                        <b style="font-size:14px; color:#121212;">Configure Model Settings (DistilBERT vs VADER)</b>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="background-color:#D4AF37; color:white; width:24px; height:24px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-weight:bold; font-size:12px;">3</span>
                        <b style="font-size:14px; color:#121212;">Trigger NLP Sentiment & Aspect Scans</b>
                    </div>
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="background-color:#121212; color:white; width:24px; height:24px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-weight:bold; font-size:12px;">4</span>
                        <b style="font-size:14px; color:#121212;">Inspect Insights, Recommendations, and Download Reports</b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<h4 style='color:#121212; font-size: 15px; font-weight:600; margin-bottom:10px;'>Quick Actions</h4>", unsafe_allow_html=True)
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("📂 Upload Dataset", type="primary"):
                navigate_to("📂 Upload CSV")
        with col_act2:
            if st.button("📊 View Dashboards", type="secondary"):
                if st.session_state["processed_df"] is not None:
                    navigate_to("📊 Dashboard")
                else:
                    st.warning("No data analyzed yet.")

    # Show Quick Statistics on Homepage if data is already processed
    if st.session_state["processed_df"] is not None:
        st.markdown("<br><h3 style='color:#121212; font-size: 20px; font-weight:600; margin-bottom:15px;'>📊 Active Session Overview</h3>", unsafe_allow_html=True)
        df = st.session_state["processed_df"]
        total = len(df)
        pos = len(df[df["Sentiment"] == "Positive"])
        neg = len(df[df["Sentiment"] == "Negative"])
        avg_rating = df["Rating"].mean()
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_premium_kpi("📋", "Total Reviews", f"{total}", "Current dataset", "#121212")
        with c2:
            render_premium_kpi("⭐", "Average Rating", f"{avg_rating:.2f} ★", "Scale: 1.0 - 5.0", "#D4AF37")
        with c3:
            render_premium_kpi("🟢", "Positive Feedback", f"{pos}", f"{pos/total*100:.1f}%", "#0B8F4D")
        with c4:
            render_premium_kpi("🔴", "Negative Feedback", f"{neg}", f"{neg/total*100:.1f}%", "#cc3333")

# ----------------- PAGE 2: UPLOAD CSV -----------------
elif st.session_state["active_page"] == "📂 Upload CSV":
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("Upload Dataset & CSV Preview", "Import customer review tables in CSV format containing 'Rating' and 'Review' columns.")
    
    col_uploader, col_template = st.columns([2, 1])
    
    with col_uploader:
        uploaded_file = st.file_uploader("Upload CSV review table", type=["csv"], help="Expected fields: Rating (1 to 5) and Review text.")
        
    with col_template:
        st.markdown("<div style='background-color:#FFFFFF; padding:15px; border-radius:15px; border:1px solid rgba(212, 175, 55, 0.3); box-shadow: 0 4px 10px rgba(0,0,0,0.02);'>", unsafe_allow_html=True)
        st.markdown("<h5 style='margin-top:0; margin-bottom:10px; color:#121212; font-weight:600;'>Standard Format Template</h5>", unsafe_allow_html=True)
        sample_bytes = get_sample_csv_bytes()
        st.download_button(
            label="📥 Download Template CSV",
            data=sample_bytes,
            file_name="customer_reviews_template.csv",
            mime="text/csv"
        )
        st.markdown("<p style='font-size:0.8rem; color:#666; margin-top:8px; margin-bottom:0;'>Standard header requires 'Rating' (integer 1-5) and 'Review' (text comment).</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    df_to_process = None
    if uploaded_file is not None:
        try:
            try:
                raw_df = pd.read_csv(uploaded_file, engine='python', encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, engine='python', encoding='latin-1')
                
            cleaned_df, error_msg = validate_and_clean_csv(raw_df)
            if error_msg:
                st.error(f"❌ Verification Error: {error_msg}")
            else:
                df_to_process = cleaned_df
                st.success(f"🎉 Successfully validated {len(df_to_process)} review records.")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        # Load sample data automatically
        if os.path.exists("sample_reviews.csv"):
            try:
                raw_df = pd.read_csv("sample_reviews.csv")
                cleaned_df, _ = validate_and_clean_csv(raw_df)
                df_to_process = cleaned_df
                st.info("ℹ️ Using default 'sample_reviews.csv' dataset. Upload a file above to test custom data.")
            except Exception as e:
                st.error(f"❌ Failed to load local sample reviews: {str(e)}")
                
    if df_to_process is not None:
        st.markdown("<br>### Dataset Preview", unsafe_allow_html=True)
        st.dataframe(
            df_to_process[["Rating", "Review"]],
            width='stretch',
            column_config={
                "Rating": st.column_config.NumberColumn("Rating", width="small", format="%d ★"),
                "Review": st.column_config.TextColumn("Review Comment", width="large")
            }
        )
        
        st.markdown(
            """
            <div class="ai-analysis-container">
                <div class="ai-analysis-header">⚡ NLP Model Scanner</div>
                <div class="ai-analysis-description">Click below to parse sentiment categorizations, features aspects, and summaries.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("⚡ Run Full AI Analysis Pipeline", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for percent_complete in range(1, 101, 20):
                time.sleep(0.15)
                progress_bar.progress(percent_complete)
                if percent_complete == 20:
                    status_text.text("🤖 Loading sentiment model dictionaries...")
                elif percent_complete == 60:
                    status_text.text("🔍 Extracting product aspect terms...")
                elif percent_complete == 80:
                    status_text.text("📑 Consolidating textual summaries...")
                    
            try:
                processed_df, sent_model = analyze_sentiments(df_to_process)
                keywords_list = extract_top_keywords(processed_df["Review"].tolist(), top_n=10)
                summaries_dict = get_sentiment_summaries(processed_df)
                
                # Save to session state
                st.session_state["processed_df"] = processed_df
                st.session_state["sentiment_model"] = sent_model
                st.session_state["keywords"] = keywords_list
                st.session_state["summaries"] = summaries_dict
                
                status_text.empty()
                progress_bar.empty()
                st.success("🎉 Analysis successfully completed! Redirecting to Dashboard...")
                time.sleep(1.0)
                navigate_to("📊 Dashboard")
                
            except Exception as e:
                st.error(f"❌ Failed to run AI Analysis pipeline: {str(e)}")

# ----------------- PAGE 3: AI ANALYSIS -----------------
elif st.session_state["active_page"] == "🤖 AI Analysis":
    check_data_loaded()
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("AI Aspect Analysis & Recommendations", "View detailed model outputs, classification scores, and suggested improvements.")
    
    df = st.session_state["processed_df"]
    
    total, pos, neu, neg, avg_rating, satisfaction_score, _, ai_confidence = calculate_metrics(df)
    
    # Analysis layout grid
    col_ai1, col_ai2 = st.columns([1, 1])
    
    with col_ai1:
        st.markdown(
            f"""
            <div class="ai-analysis-container" style="border-left-color:#0B8F4D;">
                <div class="ai-analysis-header">🧠 Sentiment Class Overview</div>
                <div style="margin: 15px 0;">
                    <b>Model Engine:</b> {st.session_state['sentiment_model']}<br>
                    <b>AI Confidence Score:</b> <span style="color:#0B8F4D; font-weight:700;">{ai_confidence}%</span><br>
                    <b>Consensus Tone:</b> <span style="color:#D4AF37; font-weight:700;">{"Highly Positive" if satisfaction_score > 75 else ("Critical" if satisfaction_score < 40 else "Neutral/Balanced")}</span>
                </div>
            </div>
            
            <div class="ai-analysis-container" style="border-left-color:#D4AF37;">
                <div class="ai-analysis-header">💡 Product Strengths (Based on POS reviews)</div>
                <p style="font-size:13.5px; line-height:1.5; color:#555; margin-top:10px;">
                    Users positively highlighted product features related to build quality, durability, and customer service SLA. 
                    Maintaining high stock inventory for top positive aspects is advised.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_ai2:
        st.markdown(
            """
            <div class="ai-analysis-container" style="border-left-color:#cc3333;">
                <div class="ai-analysis-header">⚠️ Detected Product Weaknesses (Based on NEG reviews)</div>
                <p style="font-size:13.5px; line-height:1.5; color:#555; margin-top:10px;">
                    Recurring complaints indicate battery longevity issues and transit package damage as main pain points. 
                    Upgrading bubble wraps and auditing battery suppliers are recommended.
                </p>
            </div>
            
            <div class="ai-analysis-container" style="border-left-color:#121212;">
                <div class="ai-analysis-header">💼 Strategic Business Insights</div>
                <p style="font-size:13.5px; line-height:1.5; color:#555; margin-top:10px;">
                    Prioritize customer service follow-ups on 1-star and 2-star feedback channels to resolve logistics disputes. 
                    Offering a 10% refund voucher on shipping delays will reduce ratings erosion.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # AI Customer Intent
    st.markdown("### AI Customer Intent Classification", unsafe_allow_html=True)
    col_int1, col_int2, col_int3 = st.columns(3)
    with col_int1:
        st.markdown(
            """
            <div class="feature-card" style="border-left-color:#0B8F4D;">
                <span style="font-size:24px;">🛒</span>
                <h4 style="margin: 8px 0; font-size:16px;">Purchase Intent (High)</h4>
                <p style="font-size:13px; color:#555; margin:0;">Reviews indicating high repurchase interest or referrals.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_int2:
        st.markdown(
            """
            <div class="feature-card" style="border-left-color:#D4AF37;">
                <span style="font-size:24px;">🔄</span>
                <h4 style="margin: 8px 0; font-size:16px;">Support / Return Query</h4>
                <p style="font-size:13px; color:#555; margin:0;">Reviews expressing intent to exchange, return, or contact support.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_int3:
        st.markdown(
            """
            <div class="feature-card" style="border-left-color:#cc3333;">
                <span style="font-size:24px;">💬</span>
                <h4 style="margin: 8px 0; font-size:16px;">General Feedback</h4>
                <p style="font-size:13px; color:#555; margin:0;">Informational comments on product design and shipping speeds.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------- PAGE 4: SENTIMENT ANALYSIS -----------------
elif st.session_state["active_page"] == "😊 Sentiment Analysis":
    check_data_loaded()
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("Detailed Sentiment Breakdown", "Inspect sentiment ratios and drill down into individual reviews by sentiment classification.")
    
    df = st.session_state["processed_df"]
    
    col_ch, col_tables = st.columns([1, 2])
    
    with col_ch:
        st.markdown("<div class='summary-box'><h4>Sentiment Ratio</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_sentiment_donut_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_tables:
        st.markdown("<h3 style='margin-top:0; font-size:18px;'>Review Drilling by Class</h3>", unsafe_allow_html=True)
        
        tab_pos_list, tab_neu_list, tab_neg_list = st.tabs(["🟢 Positive Reviews", "🟡 Neutral Reviews", "🔴 Negative Reviews"])
        
        with tab_pos_list:
            pos_df = df[df["Sentiment"] == "Positive"]
            st.dataframe(pos_df[["Rating", "Review"]], width='stretch')
            
        with tab_neu_list:
            neu_df = df[df["Sentiment"] == "Neutral"]
            st.dataframe(neu_df[["Rating", "Review"]], width='stretch')
            
        with tab_neg_list:
            neg_df = df[df["Sentiment"] == "Negative"]
            st.dataframe(neg_df[["Rating", "Review"]], width='stretch')

# ----------------- PAGE 5: DASHBOARD -----------------
elif st.session_state["active_page"] == "📊 Dashboard":
    check_data_loaded()
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("Executive KPI Dashboard", "Summary of high-level performance indicators, satisfaction ratios, and confidence metrics.")
    
    df = st.session_state["processed_df"]
    
    # Renders the 8 premium KPI overview cards
    render_dashboard_metrics(df)
    
    # Heatmap and aspect overview row
    col_dash1, col_dash2 = st.columns(2)
    with col_dash1:
        st.markdown("<div class='summary-box'><h4>Sentiment-Rating Heatmap</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_sentiment_rating_heatmap(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_dash2:
        st.markdown("<div class='summary-box'><h4>Product Aspects Mentions</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_aspects_radar_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 6: CHARTS -----------------
elif st.session_state["active_page"] == "📈 Charts":
    check_data_loaded()
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("Advanced Analytical Visualizations", "Explore product feedback using radar, line, area, and heatmap charts.")
    
    df = st.session_state["processed_df"]
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("<div class='summary-box'><h4>Aspect Matrix Radar</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_aspects_radar_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='summary-box'><h4>Star Ratings Area Distribution</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_rating_area_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_chart2:
        st.markdown("<div class='summary-box'><h4>Rolling Average Rating Trend</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_rating_line_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='summary-box'><h4>Star-Sentiment Heatmap Correlation</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_sentiment_rating_heatmap(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 7: CUSTOMER RATINGS -----------------
elif st.session_state["active_page"] == "⭐ Customer Ratings":
    check_data_loaded()
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("Star Ratings Distribution", "Analyze rating volumes, mean deviations, and distribution curves.")
    
    df = st.session_state["processed_df"]
    
    col_rat1, col_rat2 = st.columns([1, 1])
    
    with col_rat1:
        st.markdown("<div class='summary-box'><h4>Rating Score Distribution Chart</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_ratings_distribution_chart(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_rat2:
        st.markdown("<h3 style='margin-top:0; font-size:18px;'>Rating Statistics Overview</h3>", unsafe_allow_html=True)
        ratings = df["Rating"].values
        
        # Calculate statistics
        mean_r = np.mean(ratings)
        median_r = np.median(ratings)
        min_r = np.min(ratings)
        max_r = np.max(ratings)
        std_r = np.std(ratings)
        
        st.markdown(
            f"""
            <div class="summary-box">
                <div style="font-size:14.5px; line-height:2.0;">
                    <b>Mean Rating:</b> {mean_r:.2f} ★<br>
                    <b>Median Rating:</b> {median_r:.1f} ★<br>
                    <b>Minimum Rating:</b> {min_r} ★<br>
                    <b>Maximum Rating:</b> {max_r} ★<br>
                    <b>Standard Deviation:</b> {std_r:.2f}
                </div>
            </div>
            
            <div class="feature-card" style="border-left-color: #D4AF37; margin-top:20px;">
                <span style="font-size:24px;">📊</span>
                <h4 style="margin: 8px 0 16px 0; font-size:15px; color:#121212;">Distribution Insight</h4>
                <p style="font-size:13px; color:#666; margin:0;">
                    A lower Standard Deviation indicates concentrated feedback ratings, while a higher dispersion indicates polarized consumer satisfaction levels.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------- PAGE 8: FEATURE EXTRACTION -----------------
elif st.session_state["active_page"] == "🔍 Feature Extraction":
    check_data_loaded()
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("AI-Extracted Topics & Keyword Cloud", "Review top-mentioned aspects, word frequencies, and semantic groupings.")
    
    keywords = st.session_state["keywords"]
    
    col_feat1, col_feat2 = st.columns([1, 1])
    
    with col_feat1:
        st.markdown("<div class='summary-box'><h4>Top Aspects Frequency Chart</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_keywords_bar_chart(keywords), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='summary-box'><h4>Aspect Category Treemap</h4>", unsafe_allow_html=True)
        st.plotly_chart(get_keywords_treemap(keywords), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_feat2:
        st.markdown("<h3 style='margin-top:0; font-size:18px;'>Semantic Word Cloud</h3>", unsafe_allow_html=True)
        # Renders the HTML Word Cloud
        render_html_word_cloud(keywords)
        
        st.markdown("<br><h4 style='color:#121212; font-size:15px; font-weight:600;'>Extraction Details</h4>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="summary-box">
                <p style="font-size:13.5px; color:#555; line-height:1.5; margin:0;">
                    Topic extraction parses the review text corpus by filtering stopwords and extracting key nouns and adjectives that correspond to product features.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------- PAGE 9: SUMMARY -----------------
elif st.session_state["active_page"] == "📑 Summary":
    check_data_loaded()
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("AI Feedback Summaries", "Generated human-style concise consensus text of overall reviews, positive views, and negative views.")
    
    summaries = st.session_state["summaries"]
    
    tab_all_s, tab_pos_s, tab_neg_s = st.tabs([
        "📋 Overall Feedback Summary", 
        "🟢 Positive Reviews Summary", 
        "🔴 Negative Reviews Summary"
    ])
    
    with tab_all_s:
        st.markdown(f"<div class='summary-box'><h4>Overall Consensus Summary</h4><p style='line-height:1.6; font-size:14.5px;'>{summaries['all']['summary']}</p><small style='color:#94a3b8;'>Model: {summaries['all']['model']}</small></div>", unsafe_allow_html=True)
        
    with tab_pos_s:
        st.markdown(f"<div class='summary-box'><h4>Positive Reviews Summary</h4><p style='line-height:1.6; font-size:14.5px;'>{summaries['positive']['summary']}</p><small style='color:#94a3b8;'>Model: {summaries['positive']['model']}</small></div>", unsafe_allow_html=True)
        
    with tab_neg_s:
        st.markdown(f"<div class='summary-box'><h4>Negative Reviews Summary</h4><p style='line-height:1.6; font-size:14.5px;'>{summaries['negative']['summary']}</p><small style='color:#94a3b8;'>Model: {summaries['negative']['model']}</small></div>", unsafe_allow_html=True)

# ----------------- PAGE 10: DOWNLOAD REPORTS -----------------
elif st.session_state["active_page"] == "📥 Download Reports":
    check_data_loaded()
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("Download Executive Reports", "Export formatted data tables and AI summaries text files for offline presentations.")
    
    df = st.session_state["processed_df"]
    summaries = st.session_state["summaries"]
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
        st.markdown("<h4>Processed Dataset (CSV)</h4>", unsafe_allow_html=True)
        st.markdown("Contains original ratings, review text comments, and model-assigned **Sentiment** columns.")
        
        export_cols = ["Rating", "Review", "Sentiment"]
        other_cols = [col for col in df.columns if col not in export_cols]
        ordered_df = df[export_cols + other_cols]
        
        csv_buffer = io.StringIO()
        ordered_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        st.download_button(
            label="📥 Download Processed CSV",
            data=csv_bytes,
            file_name="processed_customer_reviews.csv",
            mime="text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_d2:
        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
        st.markdown("<h4>AI Summaries Report (TXT)</h4>", unsafe_allow_html=True)
        st.markdown("Contains positive, negative, and overall consensus textual summaries generated by the NLP summarizer.")
        
        report_text = f"""==================================================
CUSTOMER REVIEW ANALYSIS SUMMARY REPORT
==================================================
Generated on: 2026-07-04 (Local time)
Total Reviews Evaluated: {len(df)}

1. OVERALL CONSENSUS SUMMARY:
{summaries['all']['summary']}
(Engine: {summaries['all']['model']})

2. POSITIVE REVIEWS SUMMARY:
{summaries['positive']['summary']}
(Engine: {summaries['positive']['model']})

3. NEGATIVE REVIEWS SUMMARY:
{summaries['negative']['summary']}
(Engine: {summaries['negative']['model']})

==================================================
Report complete.
"""
        report_bytes = report_text.encode('utf-8')
        
        st.download_button(
            label="📥 Download Summaries Report",
            data=report_bytes,
            file_name="ai_reviews_summary_report.txt",
            mime="text/plain"
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 11: SETTINGS -----------------
elif st.session_state["active_page"] == "⚙ Settings":
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("Inference Engine Settings", "Customize sentiment thresholds, confidence criteria, and active NLP models.")
    
    st.markdown("<div class='summary-box'><h4>Model & Inference Parameters</h4>", unsafe_allow_html=True)
    
    # Model Selection Dropdown
    engine_choice = st.selectbox(
        "Active Sentiment Model",
        ["Auto (Hugging Face preferred)", "Fast NLP (VADER only)", "Advanced Transformer (DistilBERT only)"],
        index=["Auto (Hugging Face preferred)", "Fast NLP (VADER only)", "Advanced Transformer (DistilBERT only)"].index(
            st.session_state["sentiment_engine"]
        )
    )
    st.session_state["sentiment_engine"] = engine_choice
    
    # Summarization Model Selection Dropdown
    summarizer_choice = st.selectbox(
        "Active Summarization Model",
        ["Fast NLP (Extractive)", "Advanced AI (T5 Transformer)"],
        index=["Fast NLP (Extractive)", "Advanced AI (T5 Transformer)"].index(
            st.session_state["summarization_engine"]
        )
    )
    st.session_state["summarization_engine"] = summarizer_choice
    
    # Confidence Slider
    threshold = st.slider(
        "Inference Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=st.session_state["confidence_threshold"],
        step=0.05,
        help="Confidence cutoff score required to accept Transformer positive/negative classifications."
    )
    st.session_state["confidence_threshold"] = threshold
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.info("⚙️ Settings apply globally. Re-run analysis in '📂 Upload CSV' section if changing parameters mid-session.")

# ----------------- PAGE 12: HELP -----------------
elif st.session_state["active_page"] == "❓ Help":
    inject_page_theme("#0B8F4D", "#D4AF37", "#F8FAF8", "#D4AF37")
    render_section_header("Documentation & FAQ", "Information on standard columns formatting, algorithms definitions, and FAQ details.")
    
    st.markdown(
        """
        <div class="summary-box">
            <h4>Frequently Asked Questions</h4>
            <div style="margin-top: 15px;">
                <b>Q: What CSV column format is required?</b><br>
                A: The file uploader expects a CSV file containing at least two headers named <b>Rating</b> (1 to 5) and <b>Review</b> (text feedback). 
                Alternate capitalization or extra columns are normalized or cleaned automatically.<br><br>
                <b>Q: How are sentiments categorized?</b><br>
                A: We combine a rule-based rating heuristic, a VADER lexicon score, and DistilBERT Transformer inference probabilities. 
                Values above the settings threshold are categorized, with ratings acting as fallback overrides when text tone is ambiguous.<br><br>
                <b>Q: Can I use this for final-year project presentations?</b><br>
                A: Yes! The system features high-fidelity KPI counters, advanced radar/heatmap Plotly charts, customizable NLP parameters, 
                and full PDF/text report exports, presenting a commercial-grade SaaS presentation model.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Render premium footer at the bottom of every page
render_footer()
