"""
app.py
------
Streamlit web application for the Spam Email Classifier.

Run:
    streamlit run app.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.preprocess import clean_text
from src.utils import load_model

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #a8b2d8;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Result cards ── */
.result-spam {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(255,65,108,0.35);
    margin: 1rem 0;
}
.result-ham {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(56,239,125,0.35);
    margin: 1rem 0;
}
.result-label {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-sub {
    font-size: 1rem;
    opacity: 0.9;
}

/* ── Metric boxes ── */
.metric-box {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin: 0.3rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #2c3e50;
}
.metric-label {
    font-size: 0.78rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #2c3e50;
    border-left: 4px solid #3498db;
    padding-left: 0.7rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Info pills ── */
.pill {
    display: inline-block;
    background: #eaf2ff;
    color: #2980b9;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.15rem;
}

/* ── Sidebar ── */
.sidebar-title {
    font-size: 1rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SHOTS_DIR  = os.path.join(BASE_DIR, 'screenshots')

@st.cache_resource(show_spinner=False)
def load_models():
    """Load both trained pipelines once and cache them."""
    models = {}
    for name, fname in [('Naive Bayes', 'naive_bayes.pkl'),
                        ('SVM',         'svm.pkl')]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = load_model(path)
        else:
            models[name] = None
    return models


# ── Prediction helper ────────────────────────────────────────────────────────
def predict(pipeline, text: str) -> dict:
    """Run prediction through a loaded sklearn Pipeline."""
    cleaned   = clean_text(text)
    label_id  = int(pipeline.predict([cleaned])[0])
    proba     = pipeline.predict_proba([cleaned])[0]
    return {
        'label':      'Spam' if label_id == 1 else 'Ham',
        'label_id':   label_id,
        'confidence': float(max(proba)),
        'spam_prob':  float(proba[1]),
        'ham_prob':   float(proba[0]),
        'clean_text': cleaned,
    }


# ── Confidence gauge ─────────────────────────────────────────────────────────
def draw_gauge(spam_prob: float) -> plt.Figure:
    """Draw a semi-circular gauge showing spam probability."""
    fig, ax = plt.subplots(figsize=(4.5, 2.5),
                           subplot_kw=dict(aspect='equal'))
    fig.patch.set_alpha(0)
    ax.set_axis_off()

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=18, color='#e9ecef',
            solid_capstyle='round', zorder=1)

    # Coloured fill
    fill_theta = np.linspace(np.pi, np.pi - spam_prob * np.pi, 200)
    fill_color = '#E74C3C' if spam_prob > 0.5 else '#2ECC71'
    ax.plot(np.cos(fill_theta), np.sin(fill_theta),
            lw=18, color=fill_color, solid_capstyle='round', zorder=2)

    # Needle
    angle = np.pi - spam_prob * np.pi
    ax.annotate('', xy=(0.7 * np.cos(angle), 0.7 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50',
                                lw=2.5, mutation_scale=15))

    ax.text(0, -0.25, f"{spam_prob*100:.1f}%", ha='center',
            fontsize=18, fontweight='bold', color='#2c3e50')
    ax.text(0, -0.52, 'Spam Probability', ha='center',
            fontsize=9, color='#6c757d')
    ax.text(-1.1, -0.15, '0%',  ha='center', fontsize=8, color='#6c757d')
    ax.text( 1.1, -0.15, '100%', ha='center', fontsize=8, color='#6c757d')
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.7, 1.15)
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ Configuration</div>',
                unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Select Model",
        ["SVM", "Naive Bayes", "Both (Compare)"],
        index=0,
        help="SVM generally achieves higher precision; Naive Bayes is faster."
    )

    st.divider()
    st.markdown('<div class="sidebar-title">📋 Sample Emails</div>',
                unsafe_allow_html=True)

    samples = {
        "🚫 Spam – Prize Winner": (
            "CONGRATULATIONS! You have been selected as a WINNER of our "
            "£1,000,000 lottery! Click here to claim your FREE prize NOW! "
            "Call 0800-FREE-WIN. Limited time offer. Act immediately!"
        ),
        "🚫 Spam – Urgent Offer": (
            "URGENT: Your account has been compromised. Click the link below "
            "to verify your details immediately. Failure to act will result "
            "in permanent account suspension. FREE gift inside!"
        ),
        "✅ Ham – Casual": (
            "Hey! Are you free this weekend? Was thinking we could catch up "
            "over coffee. Let me know what time suits you best."
        ),
        "✅ Ham – Work": (
            "Hi team, please find attached the quarterly report. "
            "The meeting is scheduled for Monday at 10am. "
            "Let me know if you have any questions."
        ),
    }

    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state['sample_text'] = text

    st.divider()
    st.markdown("""
    <div style='font-size:0.8rem; color:#6c757d;'>
    <b>Dataset:</b> SMS Spam Collection<br>
    <b>Samples:</b> 5,572 messages<br>
    <b>Algorithm:</b> TF-IDF + ML<br>
    <b>Accuracy:</b> ~98% (SVM)
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🛡️ Spam Email Classifier</h1>
    <p>AI-powered spam detection using Naive Bayes & SVM · NLP · TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Loading models…"):
    models = load_models()

models_ok = all(v is not None for v in models.values())
if not models_ok:
    st.error(
        "⚠️ Trained models not found. Please run `python src/train.py` first "
        "to generate the model files in `/models/`."
    )
    st.stop()

# ── Input area ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📧 Enter Email Content</div>',
            unsafe_allow_html=True)

default_text = st.session_state.get('sample_text', '')
email_input  = st.text_area(
    label="",
    value=default_text,
    height=160,
    placeholder="Paste your email content here…\n\nExample: 'Congratulations! You've won a FREE prize!'",
    label_visibility="collapsed"
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
with col_btn1:
    classify_btn = st.button("🔍 Classify", type="primary",
                             use_container_width=True)
with col_btn2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop('sample_text', None)
    st.rerun()

# ── Prediction ───────────────────────────────────────────────────────────────
if classify_btn and email_input.strip():
    st.divider()

    if selected_model == "Both (Compare)":
        # ── Side-by-side comparison ──────────────────────────────────────
        st.markdown('<div class="section-header">📊 Model Comparison</div>',
                    unsafe_allow_html=True)

        col_nb, col_svm = st.columns(2)
        results = {}

        for col, mname in zip([col_nb, col_svm],
                               ['Naive Bayes', 'SVM']):
            res = predict(models[mname], email_input)
            results[mname] = res

            with col:
                st.markdown(f"**{mname}**")
                card_class = 'result-spam' if res['label'] == 'Spam' else 'result-ham'
                icon = '🚫' if res['label'] == 'Spam' else '✅'
                st.markdown(f"""
                <div class="{card_class}">
                    <div class="result-label">{icon} {res['label']}</div>
                    <div class="result-sub">Confidence: {res['confidence']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value" style="color:#E74C3C">{res['spam_prob']*100:.1f}%</div>
                        <div class="metric-label">Spam Prob</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-value" style="color:#2ECC71">{res['ham_prob']*100:.1f}%</div>
                        <div class="metric-label">Ham Prob</div>
                    </div>""", unsafe_allow_html=True)

                st.pyplot(draw_gauge(res['spam_prob']), use_container_width=True)

        # Agreement indicator
        if results['Naive Bayes']['label'] == results['SVM']['label']:
            st.success(f"✅ Both models agree: **{results['SVM']['label']}**")
        else:
            st.warning("⚠️ Models disagree – SVM result is generally more reliable.")

    else:
        # ── Single model ─────────────────────────────────────────────────
        model_key = selected_model
        res = predict(models[model_key], email_input)

        st.markdown('<div class="section-header">🎯 Classification Result</div>',
                    unsafe_allow_html=True)

        col_res, col_gauge = st.columns([3, 2])

        with col_res:
            card_class = 'result-spam' if res['label'] == 'Spam' else 'result-ham'
            icon = '🚫' if res['label'] == 'Spam' else '✅'
            st.markdown(f"""
            <div class="{card_class}">
                <div class="result-label">{icon} {res['label'].upper()}</div>
                <div class="result-sub">
                    This email is classified as <b>{res['label']}</b><br>
                    Model: {model_key} · Confidence: {res['confidence']*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown("**Probability Breakdown**")
            st.progress(res['spam_prob'],
                        text=f"🚫 Spam: {res['spam_prob']*100:.1f}%")
            st.progress(res['ham_prob'],
                        text=f"✅ Ham:  {res['ham_prob']*100:.1f}%")

            # Confidence pill
            level = ("High" if res['confidence'] > 0.90
                     else "Medium" if res['confidence'] > 0.70
                     else "Low")
            level_color = ("#2ECC71" if level == "High"
                           else "#F39C12" if level == "Medium"
                           else "#E74C3C")
            st.markdown(
                f"Confidence Level: "
                f"<span style='color:{level_color};font-weight:600'>{level}</span>",
                unsafe_allow_html=True
            )

        with col_gauge:
            st.pyplot(draw_gauge(res['spam_prob']), use_container_width=True)

        # Preprocessed text expander
        with st.expander("🔬 View Preprocessed Text"):
            st.markdown(f"""
            <div style='background:#f8f9fa;padding:1rem;border-radius:8px;
                        font-family:monospace;font-size:0.9rem;color:#495057;'>
            {res['clean_text'] if res['clean_text'] else '<em>Empty after preprocessing</em>'}
            </div>
            """, unsafe_allow_html=True)
            st.caption("Lowercased · URLs removed · Punctuation removed · "
                       "Stopwords removed · Porter Stemming applied")

elif classify_btn and not email_input.strip():
    st.warning("⚠️ Please enter some email text before classifying.")

# ── Visualisations tab ───────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">📈 Training Visualisations</div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Class Distribution",
    "📝 Word Frequency",
    "🔲 Confusion Matrix",
    "🏆 Model Comparison"
])

img_map = {
    'dist':    'class_distribution.png',
    'words':   'word_frequency.png',
    'cm_nb':   'cm_naive_bayes.png',
    'cm_svm':  'cm_svm.png',
    'compare': 'model_comparison.png',
}

def show_img(fname: str, caption: str = ''):
    path = os.path.join(SHOTS_DIR, fname)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"📌 Run `python src/train.py` to generate this visualisation.")

with tab1:
    show_img(img_map['dist'],
             'Ham vs Spam distribution in the SMS Spam Collection dataset')

with tab2:
    show_img(img_map['words'],
             'Most frequent stemmed words in Ham and Spam messages')

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        show_img(img_map['cm_nb'],  'Naive Bayes – Confusion Matrix')
    with c2:
        show_img(img_map['cm_svm'], 'SVM (LinearSVC) – Confusion Matrix')

with tab4:
    show_img(img_map['compare'],
             'Side-by-side metric comparison: Naive Bayes vs SVM')

# ── About section ────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About This Project"):
    st.markdown("""
    ### 🛡️ Spam Email Classifier

    **Dataset:** SMS Spam Collection (5,572 messages, UCI / Kaggle)

    **Pipeline:**
    1. Text cleaned → lowercased → URLs removed → punctuation stripped
    2. NLTK tokenisation → stopword removal → Porter Stemming
    3. TF-IDF vectorisation (5,000 features, unigrams + bigrams)
    4. Trained on 80% of data; evaluated on 20%

    **Models:**
    | Model | Notes |
    |-------|-------|
    | Naive Bayes (MultinomialNB) | Fast, probabilistic, great baseline |
    | SVM (LinearSVC + Calibration) | Higher precision, slower to train |

    **Why TF-IDF?**
    TF-IDF down-weights common words (stopwords slip through) and
    up-weights rare, discriminative words like *FREE*, *winner*, *prize*
    that are hallmarks of spam.

    **Tech Stack:** Python · scikit-learn · NLTK · Streamlit · Matplotlib · Seaborn
    """)

st.markdown(
    "<div style='text-align:center;color:#adb5bd;font-size:0.8rem;margin-top:2rem'>"
    "Built with ❤️ using Python, scikit-learn & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
