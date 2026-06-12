"""
app.py (flat structure version)
All files are in root directory - no src/ folder needed.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import re
import string
import nltk
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# ── Download NLTK data ───────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for r in ['punkt', 'stopwords', 'punkt_tab']:
        try:
            nltk.download(r, quiet=True)
        except:
            pass

download_nltk()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

def clean_text(text):
    import pandas as pd
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
    text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 { color: #ffffff; font-size: 2.4rem; font-weight: 700; margin: 0 0 0.5rem 0; }
.main-header p  { color: #a8b2d8; font-size: 1.05rem; margin: 0; }
.result-spam {
    background: linear-gradient(135deg, #ff416c, #ff4b2b); color: white;
    padding: 1.5rem 2rem; border-radius: 14px; text-align: center;
    box-shadow: 0 6px 24px rgba(255,65,108,0.35); margin: 1rem 0;
}
.result-ham {
    background: linear-gradient(135deg, #11998e, #38ef7d); color: white;
    padding: 1.5rem 2rem; border-radius: 14px; text-align: center;
    box-shadow: 0 6px 24px rgba(56,239,125,0.35); margin: 1rem 0;
}
.result-label { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-sub   { font-size: 1rem; opacity: 0.9; }
.section-header {
    font-size: 1.15rem; font-weight: 600; color: #2c3e50;
    border-left: 4px solid #3498db; padding-left: 0.7rem; margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SHOTS_DIR  = BASE_DIR   # screenshots are also in root

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    models = {}
    for name, fname in [('Naive Bayes', 'naive_bayes.pkl'), ('SVM', 'svm.pkl')]:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            models[name] = None
    return models

def predict(pipeline, text):
    cleaned  = clean_text(text)
    label_id = int(pipeline.predict([cleaned])[0])
    proba    = pipeline.predict_proba([cleaned])[0]
    return {
        'label':      'Spam' if label_id == 1 else 'Ham',
        'label_id':   label_id,
        'confidence': float(max(proba)),
        'spam_prob':  float(proba[1]),
        'ham_prob':   float(proba[0]),
        'clean_text': cleaned,
    }

def draw_gauge(spam_prob):
    fig, ax = plt.subplots(figsize=(4.5, 2.5), subplot_kw=dict(aspect='equal'))
    fig.patch.set_alpha(0)
    ax.set_axis_off()
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=18, color='#e9ecef', solid_capstyle='round', zorder=1)
    fill_theta = np.linspace(np.pi, np.pi - spam_prob * np.pi, 200)
    fill_color = '#E74C3C' if spam_prob > 0.5 else '#2ECC71'
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), lw=18, color=fill_color, solid_capstyle='round', zorder=2)
    angle = np.pi - spam_prob * np.pi
    ax.annotate('', xy=(0.7*np.cos(angle), 0.7*np.sin(angle)), xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5, mutation_scale=15))
    ax.text(0, -0.25, f"{spam_prob*100:.1f}%", ha='center', fontsize=18, fontweight='bold', color='#2c3e50')
    ax.text(0, -0.52, 'Spam Probability', ha='center', fontsize=9, color='#6c757d')
    ax.text(-1.1, -0.15, '0%',   ha='center', fontsize=8, color='#6c757d')
    ax.text( 1.1, -0.15, '100%', ha='center', fontsize=8, color='#6c757d')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.7, 1.15)
    return fig

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**⚙️ Configuration**")
    selected_model = st.selectbox("Select Model",
        ["SVM", "Naive Bayes", "Both (Compare)"], index=0)

    st.divider()
    st.markdown("**📋 Sample Emails**")
    samples = {
        "🚫 Spam – Prize Winner": "CONGRATULATIONS! You have been selected as a WINNER of our £1,000,000 lottery! Click here to claim your FREE prize NOW! Call 0800-FREE-WIN.",
        "🚫 Spam – Urgent Offer": "URGENT: Your account has been compromised. Click the link below to verify your details immediately. FREE gift inside!",
        "✅ Ham – Casual":        "Hey! Are you free this weekend? Was thinking we could catch up over coffee. Let me know what time suits you.",
        "✅ Ham – Work":          "Hi team, please find attached the quarterly report. The meeting is scheduled for Monday at 10am.",
    }
    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state['sample_text'] = text

    st.divider()
    st.markdown("""
    <div style='font-size:0.8rem;color:#6c757d;'>
    <b>Dataset:</b> SMS Spam Collection<br>
    <b>Samples:</b> 5,572 messages<br>
    <b>Algorithm:</b> TF-IDF + ML<br>
    <b>Accuracy:</b> ~98% (SVM)
    </div>""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ Spam Email Classifier</h1>
    <p>AI-powered spam detection using Naive Bayes & SVM · NLP · TF-IDF</p>
</div>""", unsafe_allow_html=True)

# Load models
with st.spinner("Loading models…"):
    models = load_models()

if not all(v is not None for v in models.values()):
    st.error("⚠️ Model files (naive_bayes.pkl / svm.pkl) not found in repository!")
    st.stop()

# ── Input ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📧 Enter Email Content</div>', unsafe_allow_html=True)
default_text = st.session_state.get('sample_text', '')
email_input  = st.text_area("", value=default_text, height=160,
    placeholder="Paste your email content here…\n\nExample: 'Congratulations! You've won a FREE prize!'",
    label_visibility="collapsed")

col1, col2, col3 = st.columns([1,1,4])
with col1:
    classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.pop('sample_text', None)
        st.rerun()

# ── Predict ──────────────────────────────────────────────────────────────────
if classify_btn and email_input.strip():
    st.divider()

    if selected_model == "Both (Compare)":
        st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
        col_nb, col_svm = st.columns(2)
        results = {}
        for col, mname in zip([col_nb, col_svm], ['Naive Bayes', 'SVM']):
            res = predict(models[mname], email_input)
            results[mname] = res
            with col:
                st.markdown(f"**{mname}**")
                card = 'result-spam' if res['label']=='Spam' else 'result-ham'
                icon = '🚫' if res['label']=='Spam' else '✅'
                st.markdown(f"""
                <div class="{card}">
                    <div class="result-label">{icon} {res['label']}</div>
                    <div class="result-sub">Confidence: {res['confidence']*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Spam Prob", f"{res['spam_prob']*100:.1f}%")
                with m2:
                    st.metric("Ham Prob",  f"{res['ham_prob']*100:.1f}%")
                st.pyplot(draw_gauge(res['spam_prob']), use_container_width=True)

        if results['Naive Bayes']['label'] == results['SVM']['label']:
            st.success(f"✅ Both models agree: **{results['SVM']['label']}**")
        else:
            st.warning("⚠️ Models disagree – SVM result is generally more reliable.")

    else:
        res  = predict(models[selected_model], email_input)
        st.markdown('<div class="section-header">🎯 Classification Result</div>', unsafe_allow_html=True)
        col_res, col_gauge = st.columns([3,2])

        with col_res:
            card = 'result-spam' if res['label']=='Spam' else 'result-ham'
            icon = '🚫' if res['label']=='Spam' else '✅'
            st.markdown(f"""
            <div class="{card}">
                <div class="result-label">{icon} {res['label'].upper()}</div>
                <div class="result-sub">
                    Model: {selected_model} · Confidence: {res['confidence']*100:.1f}%
                </div>
            </div>""", unsafe_allow_html=True)
            st.progress(res['spam_prob'], text=f"🚫 Spam: {res['spam_prob']*100:.1f}%")
            st.progress(res['ham_prob'],  text=f"✅ Ham:  {res['ham_prob']*100:.1f}%")

        with col_gauge:
            st.pyplot(draw_gauge(res['spam_prob']), use_container_width=True)

        with st.expander("🔬 View Preprocessed Text"):
            st.code(res['clean_text'] if res['clean_text'] else "Empty after preprocessing")

elif classify_btn:
    st.warning("⚠️ Please enter some email text first.")

# ── Visualisations ───────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">📈 Training Visualisations</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Class Distribution", "📝 Word Frequency",
    "🔲 Confusion Matrix",   "🏆 Model Comparison"
])

def show_img(fname, caption=''):
    path = os.path.join(BASE_DIR, fname)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info("📌 Image not found.")

with tab1: show_img('class_distribution.png',  'Ham vs Spam distribution')
with tab2: show_img('word_frequency.png',       'Top words in Ham and Spam')
with tab3:
    c1, c2 = st.columns(2)
    with c1: show_img('cm_naive_bayes.png', 'Naive Bayes – Confusion Matrix')
    with c2: show_img('cm_svm.png',         'SVM – Confusion Matrix')
with tab4: show_img('model_comparison.png', 'Naive Bayes vs SVM')

st.markdown(
    "<div style='text-align:center;color:#adb5bd;font-size:0.8rem;margin-top:2rem'>"
    "Built with ❤️ using Python, scikit-learn & Streamlit</div>",
    unsafe_allow_html=True
)
