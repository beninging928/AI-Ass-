import streamlit as st
import gdown
import joblib
import os
import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from PIL import Image
import pandas as pd
import time
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pro Fruit AI Dashboard",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- GLOBAL ---------- */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio > label { font-size: 0.95rem; }
/* ---------- METRIC CARDS ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f3460 100%);
    border: 1px solid #2980b9;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 4px 15px rgba(41,128,185,0.15);
}
[data-testid="stMetricValue"]  { font-size: 1.6rem !important; font-weight: 700; color: #4fc3f7 !important; }
[data-testid="stMetricDelta"]  { font-size: 0.85rem !important; }
[data-testid="stMetricLabel"]  { font-size: 0.8rem !important;  color: #90caf9 !important; }
/* ---------- FRUIT CARDS ---------- */
.fruit-card {
    background: linear-gradient(135deg, #1e3a5f, #0f3460);
    border: 1px solid #2980b9;
    border-radius: 14px;
    padding: 18px 12px;
    text-align: center;
    margin: 6px 0;
    transition: transform .2s, box-shadow .2s;
    cursor: default;
}
.fruit-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(41,128,185,0.35);
}
.fruit-card .emoji  { font-size: 2.4rem; display: block; margin-bottom: 6px; }
.fruit-card .name   { font-size: 0.85rem; font-weight: 600; color: #e0e0e0; }
.fruit-card .cal    { font-size: 0.72rem; color: #90caf9; margin-top: 3px; }
/* ---------- ARCHITECTURE CARDS ---------- */
.arch-card {
    border-radius: 14px;
    padding: 20px;
    margin: 4px 0;
    color: #fff;
}
.arch-cnn  { background: linear-gradient(135deg, #1565c0, #0d47a1); border: 1px solid #42a5f5; }
.arch-svm  { background: linear-gradient(135deg, #6a1b9a, #4a148c); border: 1px solid #ce93d8; }
.arch-lr   { background: linear-gradient(135deg, #1b5e20, #2e7d32); border: 1px solid #81c784; }
/* ---------- SECTION HEADERS ---------- */
.section-header {
    font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(90deg, #4fc3f7, #81d4fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
/* ---------- VERDICT BANNER ---------- */
.verdict-banner {
    background: linear-gradient(90deg, #0d47a1, #1565c0);
    border: 1px solid #42a5f5;
    border-radius: 14px;
    padding: 18px 24px;
    margin: 10px 0;
}
.verdict-banner h2 { color: #fff; margin: 0; font-size: 1.6rem; }
.verdict-banner p  { color: #90caf9; margin: 4px 0 0; }
/* ---------- CONFIDENCE BAR ---------- */
.conf-bar-wrap { background: #1e2d45; border-radius: 20px; height: 10px; margin: 4px 0 12px; }
.conf-bar-fill { height: 10px; border-radius: 20px; background: linear-gradient(90deg, #4fc3f7, #0288d1); }
/* ---------- LIVE BADGE ---------- */
.live-badge {
    display: inline-block;
    background: #e53935; color: #fff;
    border-radius: 20px; padding: 2px 12px;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: .05em; animation: pulse 1.5s infinite;
    vertical-align: middle; margin-left: 8px;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
/* ---------- STAT PILL ---------- */
.stat-pill {
    display: inline-block;
    background: rgba(41,128,185,0.18);
    border: 1px solid #2980b9;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem; color: #90caf9;
    margin: 3px 3px;
}
/* ---------- HIDE STREAMLIT DEFAULT FOOTER ---------- */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS & LOOKUP TABLES
# ─────────────────────────────────────────────
IMG_SIZE = 64
fruit_labels = [
    "Apple", "Avocado", "Banana", "Broccoli", "Capsicum",
    "Cauliflower", "Cucumber", "Lemon", "Mango", "Watermelon"
]

model_metrics = {
    "CNN":                {"Accuracy": 0.8697, "F1": 0.87, "Precision": 0.88, "Recall": 0.87, "Note": "Deep Learning – Primary Model",  "color": "#4fc3f7"},
    "SVM":                {"Accuracy": 0.5403, "F1": 0.54, "Precision": 0.56, "Recall": 0.54, "Note": "Best at Watermelon",             "color": "#ce93d8"},
    "Logistic Regression":{"Accuracy": 0.4417, "F1": 0.44, "Precision": 0.45, "Recall": 0.44, "Note": "Baseline statistical model",    "color": "#81c784"},
}

fruit_info = {
    "Apple":       {"emoji": "🍎", "fact": "Apples are 25 % air — that's why they float!",          "calories": "52 kcal/100g",  "color": "#e53935"},
    "Avocado":     {"emoji": "🥑", "fact": "Avocados are technically large berries.",                "calories": "160 kcal/100g", "color": "#43a047"},
    "Banana":      {"emoji": "🍌", "fact": "Bananas are slightly radioactive!",                      "calories": "89 kcal/100g",  "color": "#fdd835"},
    "Broccoli":    {"emoji": "🥦", "fact": "More protein per calorie than steak.",                   "calories": "34 kcal/100g",  "color": "#2e7d32"},
    "Capsicum":    {"emoji": "🫑", "fact": "Red peppers are just fully-ripened green peppers.",      "calories": "20 kcal/100g",  "color": "#e53935"},
    "Cauliflower": {"emoji": "🥦", "fact": "Name means 'cabbage flower'.",                           "calories": "25 kcal/100g",  "color": "#eceff1"},
    "Cucumber":    {"emoji": "🥒", "fact": "Up to 96 % water content.",                              "calories": "15 kcal/100g",  "color": "#388e3c"},
    "Lemon":       {"emoji": "🍋", "fact": "Prevents other fruits from browning.",                   "calories": "29 kcal/100g",  "color": "#f9a825"},
    "Mango":       {"emoji": "🥭", "fact": "Most consumed fruit in the world!",                      "calories": "60 kcal/100g",  "color": "#fb8c00"},
    "Watermelon":  {"emoji": "🍉", "fact": "Every part — even the rind — is edible.",               "calories": "30 kcal/100g",  "color": "#e53935"},
}

# ─────────────────────────────────────────────
# SESSION STATE (for real-time analytics)
# ─────────────────────────────────────────────
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []   # list of dicts
if "total_scans"    not in st.session_state: st.session_state.total_scans    = 0
if "correct_high"   not in st.session_state: st.session_state.correct_high   = 0   # above threshold
if "model_wins"     not in st.session_state: st.session_state.model_wins     = {"CNN": 0, "SVM": 0, "LR": 0}
if "fruit_counter"  not in st.session_state: st.session_state.fruit_counter  = {f: 0 for f in fruit_labels}
if "conf_history"   not in st.session_state: st.session_state.conf_history   = []  # rolling confidence

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    model_configs = {
        "fruit_model_v2_wy.h5": "15cCVNTiTD3bmarY4UivdOt8vSSS3KjVs",
        "svm_best_v2.pkl":      "1DDBGQNAUZBu4VNX61NObYjso_6jDVBko",
        "lr_improved.pkl":      "1j632tPQnIkFzWcpOQNdqiOVgI3Tpl4qA",
    }
    for filename, file_id in model_configs.items():
        if not os.path.exists(filename):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", filename, quiet=False)
    return (
        tf.keras.models.load_model("fruit_model_v2_wy.h5", compile=False),
        joblib.load("svm_best_v2.pkl"),
        joblib.load("lr_improved.pkl"),
    )

# ─────────────────────────────────────────────
# FEATURE EXTRACTORS
# ─────────────────────────────────────────────
def extract_lr(img_bgr):
    img_res  = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_blur = cv2.GaussianBlur(img_res, (3, 3), 0)
    gray     = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, pixels_per_cell=(4,4), cells_per_block=(2,2), feature_vector=True)
    color_feat = cv2.calcHist([img_res],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    edge_feat  = cv2.Canny(gray, 100, 200).flatten()
    return np.hstack([hog_feat, color_feat, edge_feat])

def extract_svm(img_bgr):
    gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_res = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    hog_feat = hog(gray_res, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    img_res  = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    color_feat = cv2.calcHist([img_res],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    color_feat = cv2.normalize(color_feat, color_feat).flatten()
    return np.hstack([hog_feat, color_feat])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def conf_bar(value: float, color: str = "#4fc3f7") -> str:
    pct = int(value * 100)
    return f"""
    <div class="conf-bar-wrap">
        <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>
    <div style="text-align:right;font-size:.75rem;color:#90caf9;margin-top:-8px;">{pct}%</div>
    """

def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Converts a hex color string to an rgba string compatible with Plotly."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(79,195,247,{alpha})" # fallback blue

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
with st.spinner("🔄 Loading AI models…"):
    model_cnn, model_svm, model_lr = load_all_models()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 18px'>
        <span style='font-size:2.8rem'>🍎</span>
        <h2 style='margin:6px 0 2px;font-size:1.2rem;color:#4fc3f7'>Pro Fruit AI</h2>
        <p style='font-size:.75rem;color:#90caf9;margin:0'>Multi-Model Ensemble System</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    page = st.radio(
        "Navigate",
        ["🏠 System Overview", "📸 Real-Time Detection", "📊 Model Analytics"],
        label_visibility="collapsed",
    )
    
    st.divider()
    st.markdown("<p style='font-size:.78rem;color:#607d8b;text-align:center'>⚙️ Settings</p>", unsafe_allow_html=True)
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.70, step=0.05, format="%.2f")

    st.divider()
    st.markdown("<p style='font-size:.78rem;color:#607d8b;text-align:center'>Session Statistics</p>", unsafe_allow_html=True)
    
    # Placeholders for bottom-up execution
    stats_container = st.container()
    rate_placeholder = st.empty()
    
    st.divider()

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — SYSTEM OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "🏠 System Overview":
    st.markdown("""
    <div style='background:linear-gradient(90deg,#0d47a1,#1565c0);border-radius:16px;padding:28px 32px;margin-bottom:24px'>
        <h1 style='color:#fff;margin:0;font-size:2rem'>🏠 System Overview</h1>
        <p style='color:#90caf9;margin:8px 0 0;font-size:1rem'>
            Multi-model ensemble AI for real-time fruit &amp; vegetable classification
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Welcome to the **Pro Fruit AI Dashboard** — a production-grade ensemble system combining
    **Deep Learning** and **classical Machine Learning** to classify 10 fruit &amp; vegetable
    categories with high accuracy. Use the sidebar to navigate between pages.
    """)

    # ── KPI row ───────────────────────────────
    st.markdown("<div class='section-header'>⚡ System Performance at a Glance</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best Model Accuracy",  "86.97%", "CNN",                delta_color="normal")
    k2.metric("Classes Supported",    "10",      "Fruits & Vegetables", delta_color="off")
    k3.metric("Ensemble Models",      "3",       "CNN + SVM + LR",      delta_color="off")
    k4.metric("Confidence Threshold", f"{confidence_threshold*100:.0f}%", "Auto-reject below",   delta_color="off")
    
    st.divider()

    # ── Fruit grid ────────────────────────────
    st.markdown("<div class='section-header'>✅ Supported Fruits & Vegetables</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    for i, label in enumerate(fruit_labels):
        info = fruit_info.get(label, {})
        emoji = info.get("emoji", "▫️")
        cal   = info.get("calories", "")
        with cols[i % 5]:
            st.markdown(f"""
            <div class="fruit-card">
                <span class="emoji">{emoji}</span>
                <div class="name">{label}</div>
                <div class="cal">{cal}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.divider()

    # ── Architecture cards ────────────────────
    st.markdown("<div class='section-header'>🧠 System Architecture</div>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""
        <div class="arch-card arch-cnn">
            <h3 style="margin:0 0 8px">🤖 CNN</h3>
            <p style="margin:0;font-size:.88rem;opacity:.9">
                Convolutional Neural Network — extracts complex spatial features directly from
                pixel data using 128×128 RGB input. Primary model of the ensemble.
            </p>
            <br>
            <span class="stat-pill">Accuracy 86.97%</span>
            <span class="stat-pill">F1 0.87</span>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="arch-card arch-svm">
            <h3 style="margin:0 0 8px">📐 SVM</h3>
            <p style="margin:0;font-size:.88rem;opacity:.9">
                Support Vector Machine on HOG texture features + 8-bin color histograms.
                Finds optimal decision hyperplanes in high-dimensional feature space.
            </p>
            <br>
            <span class="stat-pill">Accuracy 54.03%</span>
            <span class="stat-pill">F1 0.54</span>
        </div>
        """, unsafe_allow_html=True)
    with a3:
        st.markdown("""
        <div class="arch-card arch-lr">
            <h3 style="margin:0 0 8px">📊 Logistic Regression</h3>
            <p style="margin:0;font-size:.88rem;opacity:.9">
                Combines HOG, Canny edge map, and color histograms. Provides a fast,
                interpretable probabilistic baseline for the ensemble.
            </p>
            <br>
            <span class="stat-pill">Accuracy 44.17%</span>
            <span class="stat-pill">F1 0.44</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — REAL-TIME DETECTION
# ══════════════════════════════════════════════════════════════════════
elif page == "📸 Real-Time Detection":
    st.markdown("""
    <div style='background:linear-gradient(90deg,#1b5e20,#2e7d32);border-radius:16px;padding:28px 32px;margin-bottom:24px'>
        <h1 style='color:#fff;margin:0;font-size:2rem'>📸 Real-Time Fruit Detection</h1>
        <p style='color:#c8e6c9;margin:8px 0 0;font-size:1rem'>
            Upload or capture a fruit/vegetable image — the ensemble will classify it instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Input area ────────────────────────────
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_cam, tab_up = st.tabs(["📷 Camera Snapshot", "📁 Upload Image"])
        with tab_cam: picture = st.camera_input("Point camera at a fruit")
        with tab_up:  upload  = st.file_uploader("Drag & drop or browse", type=["jpg", "jpeg", "png"])
        
    input_img = picture if picture else upload

    if not input_img:
        st.markdown("""
        <div style='text-align:center;padding:40px 0;opacity:.5'>
            <span style='font-size:4rem'>🍎🥑🍌</span>
            <p style='color:#90caf9;margin-top:10px'>Waiting for an image…</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        img_raw = Image.open(input_img)
        img_cv  = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

        with st.spinner("🔍 Running ensemble inference…"):
            # CNN
            img_rgb  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            cnn_in   = cv2.resize(img_rgb, (128, 128)) / 255.0
            cnn_probs = model_cnn.predict(np.expand_dims(cnn_in, axis=0), verbose=0)[0]
            
            # SVM
            svm_feat = extract_svm(img_cv)
            if hasattr(model_svm, "predict_proba"):
                svm_probs = model_svm.predict_proba([svm_feat])[0]
            else:
                scores    = model_svm.decision_function([svm_feat])[0]
                e_s       = np.exp(scores - np.max(scores))
                svm_probs = e_s / e_s.sum()
                
            # LR
            lr_feat  = extract_lr(img_cv)
            lr_probs = model_lr.predict_proba([lr_feat])[0]

        # Weighted ensemble
        weighted_probs = cnn_probs * 0.7 + svm_probs * 0.2 + lr_probs * 0.1
        best_conf  = float(np.max(weighted_probs))
        final_idx  = int(np.argmax(weighted_probs))
        final_fruit = fruit_labels[final_idx]

        # Individual Model Predictions
        cnn_pred = fruit_labels[np.argmax(cnn_probs)]
        svm_pred = fruit_labels[np.argmax(svm_probs)]
        lr_pred  = fruit_labels[np.argmax(lr_probs)]

        # ── Update session state ───────────────
        st.session_state.total_scans += 1
        if best_conf >= confidence_threshold:
            st.session_state.correct_high += 1
        st.session_state.fruit_counter[final_fruit] += 1
        st.session_state.conf_history.append(round(best_conf, 4))
        
        if len(st.session_state.conf_history) > 50:
            st.session_state.conf_history = st.session_state.conf_history[-50:]
            
        if cnn_pred == final_fruit: st.session_state.model_wins["CNN"] += 1
        if svm_pred == final_fruit: st.session_state.model_wins["SVM"] += 1
        if lr_pred  == final_fruit: st.session_state.model_wins["LR"]  += 1

        # Check individual thresholds for the history log
        st.session_state.prediction_history.append({
            "Scan #":      st.session_state.total_scans,
            "Prediction":  final_fruit,
            "Confidence":  f"{best_conf*100:.1f}%",
            "CNN":         cnn_pred if np.max(cnn_probs) >= confidence_threshold else f"⚠️ {cnn_pred}",
            "SVM":         svm_pred if np.max(svm_probs) >= confidence_threshold else f"⚠️ {svm_pred}",
            "LR":          lr_pred  if np.max(lr_probs)  >= confidence_threshold else f"⚠️ {lr_pred}",
            "Accepted":    "✅" if best_conf >= confidence_threshold else "⚠️",
        })
        if len(st.session_state.prediction_history) > 20:
            st.session_state.prediction_history = st.session_state.prediction_history[-20:]

        st.divider()

        # ── Low confidence path ───────────────
        if best_conf < confidence_threshold:
            st.warning(f"⚠️ No supported fruit detected with confidence >= {confidence_threshold*100:.0f}%. Try a clearer image.")
            c_img, c_det = st.columns([1, 2])
            with c_img:
                st.image(img_raw, use_container_width=True)
            with c_det:
                st.markdown("**Top ensemble guesses:**")
                top3 = weighted_probs.argsort()[-3:][::-1]
                for i in top3:
                    st.markdown(conf_bar(weighted_probs[i], "#ef9a9a"), unsafe_allow_html=True)
                    st.caption(f"{fruit_labels[i]}  —  {weighted_probs[i]*100:.1f}%")
                    
        # ── High confidence path ──────────────
        else:
            info = fruit_info.get(final_fruit, {"emoji": "❓", "fact": "N/A", "calories": "N/A", "color": "#4fc3f7"})
            accent = info.get("color", "#4fc3f7")
            
            # Verdict banner
            st.markdown(f"""
            <div class="verdict-banner" style="border-color:{accent}">
                <h2>{info['emoji']} Ensemble Verdict: <span style="color:{accent}">{final_fruit}</span></h2>
                <p>Weighted confidence: <b style="color:{accent}">{best_conf*100:.1f}%</b>
                   &nbsp;·&nbsp; Scan #{st.session_state.total_scans}</p>
            </div>
            """, unsafe_allow_html=True)

            # Per-model results
            mc1, mc2, mc3 = st.columns(3)
            model_data = [
                ("🤖 CNN",                 cnn_probs, "#4fc3f7", mc1),
                ("📐 SVM",                 svm_probs, "#ce93d8", mc2),
                ("📊 Logistic Regression", lr_probs,  "#81c784", mc3),
            ]
            
            for name, probs, col_hex, col in model_data:
                with col:
                    idx   = int(np.argmax(probs))
                    label = fruit_labels[idx]
                    conf  = float(probs[idx])
                    
                    # -------------------------------------------------------------
                    # APPLYING THE THRESHOLD RULE TO INDIVIDUAL MODELS
                    # -------------------------------------------------------------
                    meets_threshold = conf >= confidence_threshold
                    display_label = label if meets_threshold else f"{label} (Low Conf)"
                    label_color = col_hex if meets_threshold else "#ef9a9a" # Turns red if below threshold
                    
                    st.markdown(f"""
                    <div style='background:#1e2d45;border:1px solid {col_hex};border-radius:12px;padding:14px 16px;margin-bottom:8px'>
                        <p style='color:#90caf9;font-size:.78rem;margin:0'>{name}</p>
                        <h4 style='color:{label_color};margin:4px 0'>{"✅ " if meets_threshold else "⚠️ "}{display_label}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(conf_bar(conf, label_color), unsafe_allow_html=True)
                    
                    # Mini Plotly bar chart — top 3
                    top3 = probs.argsort()[-3:][::-1]
                    fig  = go.Figure(go.Bar(
                        x=[fruit_labels[i] for i in top3],
                        y=[probs[i] * 100 for i in top3],
                        marker_color=[col_hex] + ["#37474f", "#37474f"],
                        text=[f"{probs[i]*100:.1f}%" for i in top3],
                        textposition="outside",
                    ))
                    
                    # Add Horizontal Threshold Line to individual charts
                    fig.add_hline(
                        y=confidence_threshold*100, 
                        line_dash="dot", 
                        line_color="#e53935", 
                        line_width=2,
                        opacity=0.8
                    )
                    
                    fig.update_layout(
                        height=180, margin=dict(l=0,r=0,t=12,b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 110]), # Fixed Y range to fit text
                        xaxis=dict(tickfont=dict(size=10, color="#90caf9")),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            st.divider()

            # Detail row
            img_col, info_col = st.columns([1, 2])
            with img_col:
                st.image(img_raw, use_container_width=True, caption=f"Uploaded image — {final_fruit}")
            with info_col:
                st.markdown(f"### {info['emoji']} About {final_fruit}")
                st.success(f"**Confidence:** {best_conf*100:.1f}%")
                st.info(f"💡 **Did you know?** {info['fact']}")
                st.markdown(f"🔥 **Estimated energy:** `{info['calories']}`")

                # Radar chart — ensemble breakdown
                st.markdown("**Ensemble probability breakdown (all classes)**")
                fig_radar = go.Figure(go.Scatterpolar(
                    r=[weighted_probs[i] * 100 for i in range(len(fruit_labels))],
                    theta=fruit_labels,
                    fill="toself",
                    line_color=accent,
                    fillcolor=hex_to_rgba(accent, 0.15),
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8, color="#607d8b")),
                        angularaxis=dict(tickfont=dict(size=9, color="#90caf9")),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    height=280, margin=dict(l=30,r=30,t=10,b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.divider()
            
            # Recent history table
            if st.session_state.prediction_history:
                st.markdown("**📋 Recent Scan History**")
                df_hist = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(df_hist, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL ANALYTICS  (dynamic / real-time data)
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 Model Analytics":
    # Live badge + auto-refresh toggle
    col_title, col_refresh = st.columns([4, 1])
    with col_title:
        st.markdown("""
        <div style='background:linear-gradient(90deg,#4a148c,#6a1b9a);border-radius:16px;padding:28px 32px;margin-bottom:24px'>
            <h1 style='color:#fff;margin:0;font-size:2rem'>
                📊 Model Analytics
                <span class="live-badge">LIVE</span>
            </h1>
            <p style='color:#e1bee7;margin:8px 0 0;font-size:1rem'>
                Real-time session metrics + historical training performance
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_refresh:
        st.write("")
        st.write("")
        auto_refresh = st.toggle("🔄 Auto-refresh (5 s)", value=False)
        
    if auto_refresh:
        time.sleep(5)
        st.rerun()

    # ── SECTION A: Live session stats ─────────
    st.markdown("<div class='section-header'>📡 Live Session Statistics</div>", unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Scans",        st.session_state.total_scans)
    a2.metric("High-Confidence",    st.session_state.correct_high,
              f"{(st.session_state.correct_high/max(st.session_state.total_scans,1)*100):.0f}% rate")
    a3.metric("Most Detected",
              max(st.session_state.fruit_counter, key=st.session_state.fruit_counter.get)
              if st.session_state.total_scans > 0 else "—",
              f"{max(st.session_state.fruit_counter.values())} times" if st.session_state.total_scans > 0 else "")
    a4.metric("Avg Confidence",
              f"{np.mean(st.session_state.conf_history)*100:.1f}%" if st.session_state.conf_history else "—",
              f"Last: {st.session_state.conf_history[-1]*100:.1f}%" if st.session_state.conf_history else "")

    # ── Row 1: Confidence trend + Model win rate
    r1a, r1b = st.columns([3, 2])
    with r1a:
        st.markdown("**📈 Confidence Trend (last 50 scans)**")
        if st.session_state.conf_history:
            fig_trend = go.Figure()
            x_vals = list(range(1, len(st.session_state.conf_history) + 1))
            fig_trend.add_trace(go.Scatter(
                x=x_vals, y=[v * 100 for v in st.session_state.conf_history],
                mode="lines+markers", line=dict(color="#4fc3f7", width=2),
                marker=dict(size=5), fill="tozeroy",
                fillcolor='rgba(253,216,53,0.15)', name="Confidence",
            ))
            fig_trend.add_hline(y=confidence_threshold*100, line_dash="dash", line_color="#e53935",
                                annotation_text=f"Threshold {confidence_threshold*100:.0f}%",
                                annotation_font_color="#e53935")
            fig_trend.update_layout(
                height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10,r=10,t=10,b=10),
                yaxis=dict(range=[0,105], gridcolor="#1e2d45", ticksuffix="%",
                           tickfont=dict(color="#607d8b")),
                xaxis=dict(gridcolor="#1e2d45", tickfont=dict(color="#607d8b")),
                showlegend=False,
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Run some detections to populate this chart.")
            
    with r1b:
        st.markdown("**🏆 Model Agreement Rate (this session)**")
        wins = st.session_state.model_wins
        total = max(st.session_state.total_scans, 1)
        fig_wins = go.Figure(go.Bar(
            x=["CNN", "SVM", "LR"],
            y=[wins["CNN"]/total*100, wins["SVM"]/total*100, wins["LR"]/total*100],
            marker_color=["#4fc3f7", "#ce93d8", "#81c784"],
            text=[f"{wins['CNN']}", f"{wins['SVM']}", f"{wins['LR']}"],
            textposition="outside",
        ))
        fig_wins.update_layout(
            height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0),
            yaxis=dict(range=[0,120], ticksuffix="%", gridcolor="#1e2d45",
                       tickfont=dict(color="#607d8b")),
            xaxis=dict(tickfont=dict(color="#90caf9")),
            showlegend=False,
        )
        st.plotly_chart(fig_wins, use_container_width=True)

    # ── Row 2: Fruit detection frequency
    st.markdown("**🍎 Fruit Detection Frequency (this session)**")
    fc_vals = [st.session_state.fruit_counter[f] for f in fruit_labels]
    emojis  = [fruit_info[f]["emoji"] for f in fruit_labels]
    colors  = [fruit_info[f]["color"] for f in fruit_labels]
    
    fig_freq = go.Figure(go.Bar(
        x=[f"{e} {l}" for e, l in zip(emojis, fruit_labels)],
        y=fc_vals,
        marker_color=colors,
        text=fc_vals, textposition="outside",
    ))
    fig_freq.update_layout(
        height=220, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=10,b=0),
        yaxis=dict(gridcolor="#1e2d45", tickfont=dict(color="#607d8b")),
        xaxis=dict(tickfont=dict(size=11, color="#90caf9")),
        showlegend=False,
    )
    st.plotly_chart(fig_freq, use_container_width=True)

    if st.session_state.prediction_history:
        with st.expander("📋 Full Prediction Log", expanded=False):
            df_log = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(df_log, hide_index=True, use_container_width=True)
            
    if st.button("🗑️ Reset Session Data", type="secondary"):
        st.session_state.prediction_history = []
        st.session_state.total_scans  = 0
        st.session_state.correct_high = 0
        st.session_state.model_wins   = {"CNN": 0, "SVM": 0, "LR": 0}
        st.session_state.fruit_counter = {f: 0 for f in fruit_labels}
        st.session_state.conf_history = []
        st.rerun()

    st.divider()

    # ── SECTION B: Historical training performance ──
    st.markdown("<div class='section-header'>📚 Historical Training Performance</div>", unsafe_allow_html=True)
    st.markdown("Baseline training accuracy, F1, Precision, and Recall across all three models — this dictates ensemble weighting.")
    
    df_metrics = pd.DataFrame(model_metrics).T.reset_index().rename(columns={"index": "Model"})
    b1, b2 = st.columns([3, 2])
    with b1:
        # Grouped bar chart
        fig_bar = go.Figure()
        metrics_to_show = ["Accuracy", "F1", "Precision", "Recall"]
        metric_colors   = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1"]
        for metric, mc in zip(metrics_to_show, metric_colors):
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=df_metrics["Model"],
                y=df_metrics[metric],
                marker_color=mc,
                text=[f"{v:.2f}" for v in df_metrics[metric]],
                textposition="outside",
            ))
        fig_bar.update_layout(
            barmode="group", height=340,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0,r=0,t=10,b=0),
            legend=dict(font=dict(color="#90caf9"), bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(range=[0, 1.15], tickformat=".0%", gridcolor="#1e2d45",
                       tickfont=dict(color="#607d8b")),
            xaxis=dict(tickfont=dict(color="#90caf9", size=12)),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with b2:
        # Styled dataframe
        display_df = df_metrics[["Model", "Accuracy", "F1", "Precision", "Recall"]].copy()
        display_df[["Accuracy","F1","Precision","Recall"]] = display_df[["Accuracy","F1","Precision","Recall"]].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in df_metrics.iterrows():
            color = row["color"]
            st.markdown(f"""
            <div style='background:#1e2d45;border-left:3px solid {color};padding:8px 12px;border-radius:6px;margin-bottom:6px'>
                <b style='color:{color}'>{row['Model']}</b>
                <span style='color:#90caf9;font-size:.82rem;margin-left:8px'>{row['Note']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
# ══════════════════════════════════════════════════════════════════════
# RENDER SIDEBAR METRICS (Done at the end so it reflects state updates)
# ══════════════════════════════════════════════════════════════════════
with stats_container:
    sb1, sb2 = st.columns(2)
    sb1.metric("Scans", st.session_state.total_scans)
    sb2.metric("High-Conf", st.session_state.correct_high)

if st.session_state.total_scans > 0:
    rate = st.session_state.correct_high / st.session_state.total_scans * 100
    rate_placeholder.markdown(f"<p style='text-align:center;font-size:.8rem;color:#4fc3f7'>High-Confidence Rate: <b>{rate:.0f}%</b></p>", unsafe_allow_html=True)
