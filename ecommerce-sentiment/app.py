import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import analysis_engine as engine

# 1. Page Config
st.set_page_config(page_title="Sentimates AI", layout="wide", page_icon="🛍️")

# 2. Theme State Management
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

# 3. CSS Logic
# We force specific colors for the card to ensure text never disappears
light_css = """
<style>
    /* Main Background */
    .stApp { background: linear-gradient(to bottom right, #ffffff, #f0f2f6); }
    
    /* Global Text */
    h1, h2, h3, h4, h5, p, span, div, label { color: #1a202c !important; }
    
    /* Card Container */
    .upload-container { background-color: #ffffff; border: 1px solid #e1e4e8; }
    
    /* DataFrame */
    .stDataFrame { color: #1a202c !important; }
</style>
"""

dark_css = """
<style>
    /* Main Background */
    .stApp { background: linear-gradient(to bottom right, #0f172a, #1e293b); }
    
    /* Global Text */
    h1, h2, h3, h4, h5, p, span, div, label { color: #f8fafc !important; }
    
    /* Card Container - Dark Background, White Text */
    .upload-container { background-color: #1e293b; border: 1px solid #334155; }
    
    /* DataFrame */
    .stDataFrame { color: #f8fafc !important; }
    text { fill: #f8fafc !important; }
</style>
"""

# Apply Theme
if st.session_state.theme == 'light':
    st.markdown(light_css, unsafe_allow_html=True)
else:
    st.markdown(dark_css, unsafe_allow_html=True)

# Common CSS (Layouts & Overrides)
st.markdown("""
<style>
    .upload-container {
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 0 auto;
        max-width: 700px;
        text-align: center;
    }
    
    /* FORCE Card Text Color based on container class to avoid "White on White" */
    .upload-container h3, .upload-container p {
        /* This color is set dynamically by the theme block above, 
           but we add this redundancy to be safe */
        opacity: 1 !important;
    }

    /* Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white !important;
        border: none;
        padding: 0.8rem 2.5rem;
        border-radius: 50px;
        font-size: 1.1rem;
        margin-top: 20px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Fix for the File Uploader Label Warning & Visibility */
    .stFileUploader label {
        display: none; /* We hide the actual label but provide it for accessibility */
    }
    
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HEADER WITH TOGGLE ---
c1, c2 = st.columns([8, 1])
with c1:
    st.title("🛍️ Sentimates")
with c2:
    btn_label = "🌙 Dark" if st.session_state.theme == 'light' else "☀️ Light"
    st.button(btn_label, on_click=toggle_theme)

st.markdown("<h3 style='text-align: center; font-weight: 300; margin-bottom: 40px;'>AI-Powered E-Commerce Sentiment Intelligence</h3>", unsafe_allow_html=True)

# --- CENTERED UPLOAD SECTION ---
col_spacer1, col_main, col_spacer2 = st.columns([1, 2, 1])

with col_main:
    with st.container():
        # Start Card Div
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        
        # Explicitly styled headers for safety
        st.markdown("<h3>🚀 Start Your Analysis</h3>", unsafe_allow_html=True)
        st.markdown("<p>Upload a CSV file or use the demo data to start.</p>", unsafe_allow_html=True)
        
        # Fixed File Uploader: Added a label to silence the warning
        uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        
        analyze_clicked = st.button("✨ Analyze Reviews")
        
        # End Card Div
        st.markdown('</div>', unsafe_allow_html=True)

# --- ANALYSIS SECTION ---
if analyze_clicked or 'data_loaded' in st.session_state:
    
    if analyze_clicked:
        st.session_state.data_loaded = True
        st.session_state.df = engine.load_data(uploaded_file)
    
    data_source = st.session_state.df

    if not data_source.empty:
        st.markdown("---")
        
        # KPI ROW
        st.subheader("📊 Executive Summary")
        k1, k2, k3, k4 = st.columns(4)
        
        if 'Sentiment_Score' not in data_source.columns:
             with st.spinner("Crunching numbers..."):
                data_source['Sentiment_Score'] = data_source['Review'].apply(engine.analyze_sentiment_vader)
                data_source['Sentiment_Label'] = data_source['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
        
        df = data_source
        avg_sent = df['Sentiment_Score'].mean()

        k1.metric("Average Sentiment", f"{avg_sent:.2f}")
        k2.metric("Total Reviews", len(df))
        k3.metric("Positive Reviews", len(df[df['Sentiment_Label']=='Positive']))
        k4.metric("Negative Reviews", len(df[df['Sentiment_Label']=='Negative']))

        # TABS
        tab1, tab2, tab3 = st.tabs(["📈 Visual Insights", "🔍 Word Cloud & Aspects", "🤖 Smart Assistant"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Sentiment Distribution")
                fig_dist, ax_dist = plt.subplots(facecolor='none')
                sns.countplot(x='Sentiment_Label', data=df, palette='viridis', ax=ax_dist)
                
                # Dynamic Theme Coloring
                text_color = '#1a202c' if st.session_state.theme == 'light' else '#f8fafc'
                ax_dist.tick_params(colors=text_color)
                ax_dist.xaxis.label.set_color(text_color)
                ax_dist.yaxis.label.set_color(text_color)
                ax_dist.spines['bottom'].set_color(text_color)
                ax_dist.spines['left'].set_color(text_color)
                
                sns.despine()
                st.pyplot(fig_dist)
            
            with c2:
                df_subset = df.head(50)
                if 'Emotion' not in df_subset.columns:
                     df_subset['Emotion'] = df_subset['Review'].apply(engine.detect_emotion)
                
                st.markdown("#### Emotional Tone")
                fig_emo, ax_emo = plt.subplots(facecolor='none')
                
                text_props = {'color': text_color}
                df_subset['Emotion'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax_emo, colors=sns.color_palette('pastel'), textprops=text_props)
                ax_emo.set_ylabel('')
                st.pyplot(fig_emo)

        with tab2:
            st.markdown("#### ☁️ What are people complaining about?")
            neg_text = " ".join(df[df['Sentiment_Label'] == 'Negative']['Review'].tolist())
            if neg_text:
                wc_bg = 'white' if st.session_state.theme == 'light' else '#1e293b'
                wordcloud = WordCloud(width=1000, height=400, background_color=wc_bg, colormap='Reds').generate(neg_text)
                st.image(wordcloud.to_array(), use_column_width=True)
            else:
                st.info("No negative reviews found to analyze!")

        with tab3:
            st.markdown("#### 📝 AI Response Drafter")
            df_display = df_subset[['Review', 'Sentiment_Label', 'Emotion']].copy()
            df_display['AI Drafted Reply'] = df_display.apply(lambda x: engine.generate_smart_reply(x['Review'], 0 if x['Sentiment_Label'] == 'Neutral' else (1 if x['Sentiment_Label']=='Positive' else -1)), axis=1)
            st.dataframe(df_display, use_container_width=True)