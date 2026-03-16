import streamlit as st
import re
import io
import os
import numpy as np
from pathlib import Path

import nltk

# NLTK data download — safe way for Streamlit Cloud
for corpus in ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']:
    nltk.download(corpus, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pdfplumber
import joblib

st.set_page_config(
    page_title="Resume IQ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:         #f8faff;
    --bg2:        #ffffff;
    --bg3:        #f1f5fe;
    --border:     #e3e8f7;
    --border2:    #c8d3f0;
    --text1:      #0a0e2e;
    --text2:      #3a4374;
    --text3:      #8b95c9;
    --blue:       #2563eb;
    --green:      #10b981;
    --purple:     #7c3aed;
    --grad:       linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
    --sh1:        0 1px 3px #2563eb0f, 0 4px 16px #2563eb08;
    --sh2:        0 4px 24px #2563eb14, 0 1px 4px #0a0e2e08;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
.main .block-container {
    background: var(--bg) !important;
    color: var(--text1) !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 3.5rem 6rem !important; max-width: 1240px !important; }

.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.4rem 0 1.2rem; margin-bottom: 3rem;
    border-bottom: 1px solid var(--border);
}
.nav-logo { display: flex; align-items: center; gap: 0.7rem; }
.nav-logo-mark {
    width: 36px; height: 36px; background: var(--grad);
    border-radius: 10px; display: grid; place-items: center;
    font-size: 1.05rem; box-shadow: 0 4px 12px #2563eb2a;
}
.nav-logo-text { font-size: 1.18rem; font-weight: 700; color: var(--text1); letter-spacing: -0.03em; }
.nav-logo-text span {
    background: var(--grad); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; background-clip: text;
}
.nav-right { display: flex; align-items: center; gap: 0.6rem; }
.nbadge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.14em; text-transform: uppercase;
    padding: 0.3rem 0.85rem; border-radius: 100px;
    border: 1px solid var(--border2); color: var(--text3); background: var(--bg2);
}
.nbadge-active { border-color: #2563eb30; color: var(--blue); background: #2563eb08; font-weight: 500; }

.hero { padding: 1rem 0 2.5rem; }
.hero-chip {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: #2563eb0a; border: 1px solid #2563eb1e;
    color: var(--blue); font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 0.32rem 1rem; border-radius: 100px; margin-bottom: 1.2rem;
}
.hero-chip-dot {
    width: 5px; height: 5px; background: var(--blue);
    border-radius: 50%; animation: pulse 2s infinite;
}
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.hero-h1 {
    font-size: clamp(2.2rem, 4vw, 3.2rem); font-weight: 800;
    letter-spacing: -0.04em; line-height: 1.08; color: var(--text1); margin-bottom: 1rem;
}
.hero-h1 em {
    font-style: normal; background: var(--grad);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-p { font-size: 0.95rem; color: var(--text3); line-height: 1.7; max-width: 460px; }

[data-testid="stMetric"] {
    background: var(--bg2) !important; border: 1px solid var(--border) !important;
    border-radius: 14px !important; padding: 1.4rem 1.5rem !important;
    box-shadow: var(--sh1) !important; position: relative !important;
    overflow: hidden !important; transition: box-shadow 0.2s, transform 0.2s !important;
}
[data-testid="stMetric"]:hover { box-shadow: var(--sh2) !important; transform: translateY(-2px) !important; }
[data-testid="stMetric"]::after {
    content: '' !important; position: absolute !important;
    top: 0; left: 0; right: 0; height: 2px !important;
    background: var(--grad) !important; border-radius: 14px 14px 0 0 !important;
}
[data-testid="stMetricValue"] > div {
    font-family: 'Inter', sans-serif !important; font-size: 2.1rem !important;
    font-weight: 800 !important; letter-spacing: -0.04em !important;
    background: var(--grad) !important; -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important; background-clip: text !important; line-height: 1 !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.6rem !important;
    color: var(--text3) !important; letter-spacing: 0.14em !important;
    text-transform: uppercase !important; margin-top: 0.35rem !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

.div-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border2) 30%, var(--border2) 70%, transparent);
    margin: 0 0 2.8rem;
}

.sec-head { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 1rem; }
.sec-bar { width: 3px; height: 16px; background: var(--grad); border-radius: 100px; }
.sec-lbl {
    font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; font-weight: 500;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--text3);
}

.inp-panel {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 18px; padding: 1.8rem; box-shadow: var(--sh2);
    position: relative; overflow: hidden;
}
.inp-panel::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--grad);
}

.stTextArea > label { display: none !important; }
.stTextArea textarea {
    background: var(--bg3) !important; border: 1.5px solid var(--border) !important;
    border-radius: 12px !important; color: var(--text1) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important;
    line-height: 1.8 !important; padding: 1.1rem 1.2rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--blue) !important; background: var(--bg2) !important;
    box-shadow: 0 0 0 4px #2563eb0e !important; outline: none !important;
}
.stTextArea textarea::placeholder { color: var(--text3) !important; font-style: italic; }

.stFileUploader > label { display: none !important; }
[data-testid="stFileUploader"] {
    background: var(--bg3) !important; border: 1.5px dashed var(--border2) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: var(--text3) !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important;
}

[data-testid="stButton"] button[kind="primary"] {
    background: var(--grad) !important; color: #fff !important; border: none !important;
    padding: 0.88rem 1.5rem !important; border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important;
    font-weight: 700 !important; width: 100% !important; margin-top: 1rem !important;
    box-shadow: 0 4px 20px #2563eb28 !important;
}

[data-baseweb="tab-list"] {
    background: var(--bg3) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; padding: 3px !important; gap: 2px !important;
    margin-bottom: 1.2rem !important;
}
[data-baseweb="tab"] {
    background: transparent !important; color: var(--text3) !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important;
    border-radius: 7px !important; padding: 0.44rem 1.1rem !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--bg2) !important; color: var(--blue) !important;
    box-shadow: var(--sh1) !important; font-weight: 500 !important;
}

[data-testid="stProgressBar"] > div {
    background: var(--bg3) !important; border-radius: 100px !important; height: 6px !important;
}
[data-testid="stProgressBar"] > div > div { border-radius: 100px !important; }

.res-panel-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.9rem 1.4rem; background: var(--bg3);
    border: 1px solid var(--border); border-bottom: none; border-radius: 14px 14px 0 0;
}
.res-panel-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--text3);
}
.res-panel-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.56rem; color: var(--blue);
    background: #2563eb0a; border: 1px solid #2563eb20; padding: 0.14rem 0.6rem; border-radius: 100px;
}

.rcard {
    background: var(--bg2); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.1rem 1.3rem; margin-bottom: 0.6rem;
    position: relative; overflow: hidden; box-shadow: var(--sh1); transition: box-shadow 0.15s;
}
.rcard:hover { box-shadow: var(--sh2); }
.rcard-best { background: linear-gradient(135deg, #eff6ff 0%, #f0feff 100%); border-color: #2563eb22; }
.rcard-best::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: var(--grad);
}
.rcard-top-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.45rem; }
.rcard-num {
    font-family: 'JetBrains Mono', monospace; font-size: 0.56rem;
    color: var(--text3); letter-spacing: 0.14em; text-transform: uppercase;
}
.rcard-pill {
    font-family: 'JetBrains Mono', monospace; font-size: 0.5rem; color: var(--green);
    background: #10b98110; border: 1px solid #10b98125; padding: 0.1rem 0.55rem; border-radius: 100px;
}
.rcard-name {
    font-family: 'Inter', sans-serif; font-size: 1.12rem; font-weight: 700;
    color: var(--text1); letter-spacing: -0.025em; margin-bottom: 0.1rem;
}
.rcard-conf-row { display: flex; justify-content: space-between; align-items: center; margin-top: 0.3rem; }
.rcard-conf-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.6rem; color: var(--text3); }
.rcard-conf-val { font-family: 'Inter', sans-serif; font-size: 0.88rem; font-weight: 700; }

.meta-strip {
    display: grid; grid-template-columns: repeat(4,1fr);
    border: 1px solid var(--border); border-radius: 0 0 14px 14px;
    overflow: hidden; background: var(--bg3);
}
.ms-cell { padding: 0.7rem 1rem; border-right: 1px solid var(--border); }
.ms-cell:last-child { border-right: none; }
.ms-k {
    font-family: 'JetBrains Mono', monospace; font-size: 0.52rem;
    color: var(--text3); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.22rem;
}
.ms-v { font-family: 'Inter', sans-serif; font-size: 0.88rem; font-weight: 600; color: var(--text2); }
.ms-ok { color: var(--green); }

.ib {
    border-radius: 10px; padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; line-height: 1.6; margin-top: 0.8rem;
}
.ib-ok  { background:#10b98108; border:1px solid #10b98120; color:#047857; }
.ib-err { background:#ef444408; border:1px solid #ef444420; color:#b91c1c; }

.empty {
    padding: 3.5rem 2rem; text-align: center;
    border: 1.5px dashed var(--border2); border-radius: 16px; background: var(--bg2);
}
.empty-ico { font-size: 2.5rem; display: block; margin-bottom: 0.9rem; opacity: 0.4; }
.empty-t1 { font-size: 0.92rem; font-weight: 600; color: var(--text3); margin-bottom: 0.4rem; }
.empty-t2 { font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:var(--text3); opacity:0.5; line-height:1.9; }
</style>
""", unsafe_allow_html=True)


# ── Backend ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_nltk_resources():
    return WordNetLemmatizer(), set(stopwords.words('english'))

lemmatizer, STOPS = load_nltk_resources()

def preprocess(text: str) -> str:
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(
        lemmatizer.lemmatize(w) for w in text.split()
        if w not in STOPS and len(w) > 2
    )

def extract_pdf_text(f) -> str:
    try:
        with pdfplumber.open(io.BytesIO(f.read())) as pdf:
            return '\n'.join(p.extract_text() or '' for p in pdf.pages)
    except Exception:
        return ''

@st.cache_resource
def load_model():
    mp = Path('model/resume_classifier.pkl')
    ep = Path('model/label_encoder.pkl')
    if not mp.exists() or not ep.exists():
        return None, None
    return joblib.load(mp), joblib.load(ep)

pipe, le = load_model()

def predict(text):
    cleaned = preprocess(text)
    probs   = pipe.predict_proba([cleaned])[0]
    top3    = probs.argsort()[::-1][:3]
    return [{'rank': i+1, 'category': le.classes_[idx],
             'confidence': round(float(probs[idx])*100, 1)}
            for i, idx in enumerate(top3)]

PROG_COLORS = {
    1: ('#2563eb',),
    2: ('#7c3aed',),
    3: ('#0ea5e9',),
}


# ══════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════
st.markdown("""
<div class="navbar">
  <div class="nav-logo">
    <div class="nav-logo-mark">🧠</div>
    <div class="nav-logo-text">Resume<span>IQ</span></div>
  </div>
  <div class="nav-right">
    <div class="nbadge nbadge-active">ML Powered</div>
    <div class="nbadge">Offline · No API</div>
  </div>
</div>
""", unsafe_allow_html=True)

if pipe is None:
    st.markdown("""
    <div class="ib ib-err">
        ⚠ <strong>Model not found.</strong>
        Run the notebook → saves <code>model/resume_classifier.pkl</code>
        and <code>model/label_encoder.pkl</code> → restart app.
    </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════
# HERO + STATS
# ══════════════════════════════════════
h_col, s_col = st.columns([2.2, 1], gap="large")

with h_col:
    st.markdown("""
    <div class="hero">
      <div class="hero-chip">
        <span class="hero-chip-dot"></span>
        Resume Intelligence Platform
      </div>
      <h1 class="hero-h1">
        Identify Job Profile<br>from <em>Any Resume</em>
      </h1>
      <p class="hero-p">
        Paste text or upload a PDF — our ML model predicts
        the top matching job categories in milliseconds.
        Works completely offline, no API needed.
      </p>
    </div>
    """, unsafe_allow_html=True)

with s_col:
    st.metric("Job Categories",  str(len(le.classes_)))
    st.metric("Resumes Trained", "2484")
    st.metric("F1 Score",        "99%")

st.markdown('<div class="div-line"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════
# INPUT | RESULTS
# ══════════════════════════════════════
c_left, c_right = st.columns([1.05, 0.95], gap="large")

with c_left:
    st.markdown("""
    <div class="sec-head">
      <div class="sec-bar"></div>
      <div class="sec-lbl">Resume Input</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="inp-panel">', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Paste Text", "⬆ Upload PDF"])
    resume_text = ""

    with tab1:
        resume_text = st.text_area(
            label="t", label_visibility="collapsed",
            placeholder="Paste resume content here...\n\nTip: Include skills, experience & education for best accuracy.",
            height=270, key="txt"
        )

    with tab2:
        up = st.file_uploader("u", type=["pdf"], label_visibility="collapsed", key="pdf_upload")
        if up:
            with st.spinner("Parsing PDF..."):
                pt = extract_pdf_text(up)
            if pt.strip():
                resume_text = pt
                st.markdown(f"""
                <div class="ib ib-ok">
                  ✓ <strong>Parsed</strong> — {len(pt.split()):,} words from <strong>{up.name}</strong>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="ib ib-err">✕ No text found. Use a text-based PDF.</div>
                """, unsafe_allow_html=True)

    btn = st.button("Analyze Resume →", use_container_width=True, type="primary", key="go")
    st.markdown('</div>', unsafe_allow_html=True)


with c_right:
    st.markdown("""
    <div class="sec-head">
      <div class="sec-bar"></div>
      <div class="sec-lbl">Prediction Results</div>
    </div>
    """, unsafe_allow_html=True)

    if btn and resume_text and resume_text.strip():
        with st.spinner("Running model..."):
            results  = predict(resume_text)
            wc       = len(resume_text.split())
            tc       = len(preprocess(resume_text).split())
            top_conf = results[0]['confidence']

        st.markdown("""
        <div class="res-panel-head">
          <div class="res-panel-title">Top Matches</div>
          <div class="res-panel-badge">3 Results</div>
        </div>
        """, unsafe_allow_html=True)

        colors = ['#2563eb', '#7c3aed', '#0ea5e9']
        for r in results:
            is_best   = r['rank'] == 1
            ccls      = "rcard rcard-best" if is_best else "rcard"
            badge     = '<span class="rcard-pill">✓ Best Match</span>' if is_best else ""
            cval      = colors[r['rank'] - 1]
            conf      = r['confidence']
            name_style = "background:linear-gradient(135deg,#2563eb,#0ea5e9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text" if is_best else ""

            st.markdown(f"""
            <div class="{ccls}">
              <div class="rcard-top-row">
                <span class="rcard-num">Match #{r['rank']}</span>
                {badge}
              </div>
              <div class="rcard-name" style="{name_style}">{r['category']}</div>
              <div class="rcard-conf-row">
                <span class="rcard-conf-lbl">Confidence</span>
                <span class="rcard-conf-val" style="color:{cval}">{conf:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(conf))

        st.markdown(f"""
        <div class="meta-strip">
          <div class="ms-cell"><div class="ms-k">Words</div><div class="ms-v">{wc:,}</div></div>
          <div class="ms-cell"><div class="ms-k">Tokens</div><div class="ms-v">~{tc:,}</div></div>
          <div class="ms-cell"><div class="ms-k">Top Score</div><div class="ms-v">{top_conf}%</div></div>
          <div class="ms-cell"><div class="ms-k">Status</div><div class="ms-v ms-ok">✓ Done</div></div>
        </div>
        """, unsafe_allow_html=True)

    elif btn:
        st.markdown("""
        <div class="ib ib-err">✕ Please paste resume text or upload a PDF.</div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty">
          <span class="empty-ico">🧠</span>
          <div class="empty-t1">No analysis yet</div>
          <div class="empty-t2">Paste or upload resume on the left<br>then click Analyze Resume</div>
        </div>
        """, unsafe_allow_html=True)