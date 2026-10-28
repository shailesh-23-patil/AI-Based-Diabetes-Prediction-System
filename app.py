from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Based Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "diabetes_preprocessor.pkl"


# --------------------------------------------------
# 2. Custom styling (CSS)
# --------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, button, input, select, textarea {
    font-family: 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif !important;
}

/* ---------- App background & layout ---------- */
.stApp {
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(99,102,241,.10), transparent 60%),
        radial-gradient(900px 500px at -10% 10%, rgba(13,148,136,.10), transparent 60%),
        #f3f7fb;
}
.block-container { max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }

@keyframes fadeUp { from {opacity:0; transform: translateY(14px);} to {opacity:1; transform:none;} }
@keyframes fillgauge { from { stroke-dasharray: 0 251.33; } }
@keyframes floaty { 0%,100% {transform: translateY(0);} 50% {transform: translateY(-8px);} }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden; color: #fff;
    padding: 2.6rem 2.4rem; border-radius: 28px;
    background: linear-gradient(125deg, #0f766e 0%, #0d9488 35%, #4f46e5 100%);
    box-shadow: 0 24px 50px -18px rgba(79,70,229,.55);
    animation: fadeUp .6s ease both;
}
.hero::before, .hero::after {
    content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,.10);
}
.hero::before { width: 320px; height: 320px; right: -90px; top: -120px; }
.hero::after  { width: 200px; height: 200px; right: 160px; bottom: -110px; background: rgba(255,255,255,.07); }
.hero .emoji { position:absolute; right: 3rem; top: 50%; margin-top:-48px; font-size: 5.5rem;
    animation: floaty 4s ease-in-out infinite; filter: drop-shadow(0 12px 18px rgba(0,0,0,.25)); }
.hero h1 { margin: 0 0 .5rem 0; font-size: 2.35rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.15; color:#fff; }
.hero p  { margin: 0; max-width: 640px; font-size: 1.02rem; line-height: 1.65; color: rgba(255,255,255,.88); }
.badges  { margin-top: 1.2rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.badge   { padding: .38rem .85rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
           background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); backdrop-filter: blur(6px); }

/* ---------- Stat cards ---------- */
.stats { display:grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.3rem 0 1.4rem; }
.stat  { background:#fff; border-radius: 20px; padding: 1.1rem 1.2rem; border: 1px solid #e8eef5;
         box-shadow: 0 8px 24px -14px rgba(15,23,42,.18); animation: fadeUp .6s ease both;
         transition: transform .2s ease, box-shadow .2s ease; }
.stat:hover { transform: translateY(-3px); box-shadow: 0 16px 30px -16px rgba(15,23,42,.28); }
.stat .ico  { width: 40px; height: 40px; border-radius: 12px; display:flex; align-items:center;
              justify-content:center; font-size: 1.2rem; margin-bottom:.6rem; }
.stat .val  { font-size: 1.35rem; font-weight: 800; color:#0f172a; line-height:1.1; }
.stat .lbl  { font-size: .8rem; color:#64748b; margin-top:.2rem; font-weight:500; }
.ic-teal   { background:#ccfbf1; } .ic-indigo { background:#e0e7ff; }
.ic-amber  { background:#fef3c7; } .ic-rose   { background:#ffe4e6; }

/* ---------- Disclaimer ---------- */
.notice { display:flex; gap:.8rem; align-items:flex-start; padding: .95rem 1.2rem; border-radius: 16px;
          background: #fffbeb; border: 1px solid #fde68a; color:#92400e; font-size:.9rem; line-height:1.55;
          margin-bottom: 1.4rem; }
.notice b { color:#78350f; }

/* ---------- Section headings ---------- */
.sec-title { font-size: 1.45rem; font-weight: 800; color:#0f172a; margin: .2rem 0 .15rem; letter-spacing:-.01em; }
.sec-sub   { color:#64748b; font-size:.93rem; margin-bottom: 1rem; }

/* ---------- Form card ---------- */
[data-testid="stForm"] {
    background:#fff; border: 1px solid #e8eef5; border-radius: 26px; padding: 1.6rem 1.7rem 1.4rem;
    box-shadow: 0 18px 40px -24px rgba(15,23,42,.25);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; background:#f1f5f9; padding:.35rem; border-radius: 16px; }
.stTabs [data-baseweb="tab"] { height: 2.7rem; border-radius: 12px; padding: 0 1.05rem; font-weight: 600; color:#475569; background: transparent; }
.stTabs [aria-selected="true"] { background:#fff !important; color:#0d9488 !important; box-shadow: 0 4px 12px -4px rgba(15,23,42,.2); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

.tab-head { display:flex; align-items:center; gap:.7rem; margin: 1.1rem 0 .9rem; }
.tab-head .chip { width:42px; height:42px; border-radius:13px; display:flex; align-items:center; justify-content:center; font-size:1.3rem; }
.tab-head .t { font-weight:700; font-size:1.08rem; color:#0f172a; line-height:1.2; }
.tab-head .s { font-size:.84rem; color:#64748b; }

/* Inputs */
label, .stMarkdown p { color:#334155; }
[data-testid="stWidgetLabel"] p { font-weight:600; font-size:.88rem; color:#334155; }
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
    border-radius: 12px !important; border-color:#e2e8f0 !important; background:#f8fafc !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color:#0d9488 !important; box-shadow: 0 0 0 3px rgba(13,148,136,.15) !important; background:#fff !important;
}

.next-hint { margin-top: 1.1rem; padding: .7rem 1rem; border-radius: 12px; background:#f0fdfa; color:#0f766e;
              font-size:.86rem; font-weight:600; border:1px dashed #99f6e4; text-align:right; }
.final-note { padding:.8rem 1rem; border-radius:12px; background:#eef2ff; color:#3730a3; font-size:.9rem;
              border:1px solid #c7d2fe; margin-bottom:.9rem; }

/* Submit button */
.stFormSubmitButton button {
    height: 3.3rem; border: 0; border-radius: 16px; font-size: 1.02rem; font-weight: 700; color:#fff;
    background: linear-gradient(120deg, #0d9488, #4f46e5); box-shadow: 0 14px 26px -12px rgba(79,70,229,.7);
    transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
.stFormSubmitButton button:hover { transform: translateY(-2px); filter: brightness(1.06); box-shadow: 0 20px 30px -14px rgba(79,70,229,.8); color:#fff; border:0; }
.stFormSubmitButton button:active { transform: translateY(0); }

/* ---------- Result card ---------- */
.result { display:flex; flex-wrap:wrap; align-items:center; gap: 2rem; background:#fff; border-radius: 28px;
          padding: 1.8rem 2rem; border: 1px solid #e8eef5; position:relative; overflow:hidden;
          box-shadow: 0 24px 50px -26px rgba(15,23,42,.35); animation: fadeUp .5s ease both; }
.result::before { content:""; position:absolute; left:0; top:0; bottom:0; width:9px; }
.result.pos::before { background: linear-gradient(180deg,#f43f5e,#ef4444); }
.result.neg::before { background: linear-gradient(180deg,#10b981,#14b8a6); }
.result .left  { flex: 0 0 280px; max-width:100%; }
.result .right { flex: 1 1 280px; }
.gauge { width:100%; height:auto; }
.gauge-fill { animation: fillgauge 1.2s ease-out both; }
.gauge-num { font-size: 30px; font-weight: 800; fill:#0f172a; }
.gauge-cap { font-size: 9.5px; fill:#64748b; font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
.pill { display:inline-block; padding:.4rem .95rem; border-radius:999px; font-size:.8rem; font-weight:700; letter-spacing:.03em; margin-bottom:.8rem; }
.pill.pos { background:#ffe4e6; color:#be123c; } .pill.neg { background:#d1fae5; color:#047857; }
.result h2 { margin:0 0 .5rem; font-size:1.7rem; font-weight:800; color:#0f172a; letter-spacing:-.015em; }
.result p  { margin:0 0 .9rem; color:#475569; font-size:.95rem; line-height:1.6; }
.band { display:inline-flex; gap:.5rem; align-items:center; font-size:.88rem; color:#334155; background:#f1f5f9;
        padding:.45rem .9rem; border-radius:12px; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }

/* ---------- Snapshot tiles ---------- */
.tiles { display:grid; grid-template-columns: repeat(6, 1fr); gap:.8rem; margin: 1.1rem 0 .4rem; }
.tile  { background:#fff; border:1px solid #e8eef5; border-radius:18px; padding:.9rem 1rem; text-align:center;
         box-shadow: 0 8px 20px -16px rgba(15,23,42,.3); animation: fadeUp .6s ease both; }
.tile .k { font-size:.72rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:.05em; }
.tile .v { font-size:1.25rem; font-weight:800; color:#0f172a; margin-top:.15rem; }
.tile .u { font-size:.7rem; color:#94a3b8; }

/* ---------- Submitted details ---------- */
.dgrid { display:grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 1rem; padding: .4rem 0 .2rem; }
.dcard { background:#fff; border:1px solid #e8eef5; border-radius:18px; padding: 1rem 1.2rem;
         box-shadow: 0 8px 20px -16px rgba(15,23,42,.3); }
.dtitle { font-weight:800; font-size:.98rem; color:#0f172a; margin-bottom:.55rem; padding-bottom:.55rem; border-bottom:2px solid #f1f5f9; }
.drow { display:flex; justify-content:space-between; gap:1rem; padding:.42rem 0; border-bottom:1px dashed #e2e8f0; font-size:.88rem; }
.drow:last-child { border-bottom:0; }
.dk { color:#64748b; font-weight:500; }
.dv { color:#0f172a; font-weight:700; text-align:right; }
[data-testid="stExpander"] { border-radius:18px !important; border:1px solid #e8eef5 !important; background:#f8fafc; }
[data-testid="stExpander"] summary { font-weight:700; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: linear-gradient(185deg, #0f172a 0%, #1e1b4b 100%); }
[data-testid="stSidebar"] * { color:#cbd5e1; }
[data-testid="stSidebar"] .side-title { color:#fff; font-weight:800; font-size:1.25rem; margin:.2rem 0 .1rem; }
[data-testid="stSidebar"] .side-sub { font-size:.82rem; color:#94a3b8; margin-bottom:1.2rem; }
.step { display:flex; gap:.75rem; margin-bottom: .95rem; align-items:flex-start; }
.step .n { flex:0 0 28px; height:28px; border-radius:9px; display:flex; align-items:center; justify-content:center;
           font-weight:800; font-size:.82rem; background: linear-gradient(135deg,#14b8a6,#6366f1); color:#fff !important; }
.step .tx { font-size:.86rem; line-height:1.5; }
.step .tx b { color:#fff; }
.status { display:inline-flex; align-items:center; gap:.5rem; padding:.45rem .8rem; border-radius:999px; font-size:.78rem; font-weight:600;
          background: rgba(16,185,129,.14); border:1px solid rgba(16,185,129,.35); color:#6ee7b7 !important; margin: .4rem 0 1.2rem; }
.side-card { background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1); border-radius:16px; padding: .9rem 1rem;
             font-size:.8rem; line-height:1.55; margin-top:1rem; }

/* ---------- Footer ---------- */
.footer { text-align:center; color:#94a3b8; font-size:.82rem; padding: 2rem 0 .5rem; }
.footer b { color:#64748b; }

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .tiles { grid-template-columns: repeat(3, 1fr); }
    .hero .emoji { display:none; }
    .hero h1 { font-size: 1.8rem; }
    .result .left { flex-basis: 100%; }
}
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)


def tab_header(icon, bg, title, subtitle):
    st.markdown(
        f'<div class="tab-head"><div class="chip" style="background:{bg}">{icon}</div>'
        f'<div><div class="t">{title}</div><div class="s">{subtitle}</div></div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 3. Load the trained model and preprocessing pipeline
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


# --------------------------------------------------
# 4. Hero + overview
# --------------------------------------------------

st.markdown(
    '<div class="hero"><div class="emoji">🩺</div>'
    '<h1>AI-Based Diabetes<br>Prediction System</h1>'
    '<p>An academic machine learning project that analyses health indicators and '
    'predicts the diabetes label using a trained Random Forest classifier.</p>'
    '<div class="badges"><span class="badge">🌲 Random Forest</span>'
    '<span class="badge">⚙️ scikit-learn</span><span class="badge">⚡ Real-time prediction</span>'
    '<span class="badge">🎓 Academic project</span></div></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="stats">'
    '<div class="stat"><div class="ico ic-teal">🧬</div><div class="val">28</div><div class="lbl">Health indicators</div></div>'
    '<div class="stat"><div class="ico ic-indigo">🗂️</div><div class="val">5</div><div class="lbl">Input categories</div></div>'
    '<div class="stat"><div class="ico ic-amber">🌲</div><div class="val">Random Forest</div><div class="lbl">Classification model</div></div>'
    '<div class="stat"><div class="ico ic-rose">⚡</div><div class="val">Instant</div><div class="lbl">Prediction in seconds</div></div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice"><div style="font-size:1.3rem">⚠️</div><div><b>Educational use only.</b> '
    'This model is not a medical diagnostic tool. Do not use its predictions to make '
    'healthcare decisions.</div></div>',
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
    st.error(
        "Model files not found. Please ensure that both files "
        "are inside the project's models folder."
    )
    st.stop()

try:
    model, preprocessor = load_model()
except Exception as error:
    st.error(f"Unable to load the model files: {error}")
    st.stop()


# --------------------------------------------------
# 5. Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="side-title">🩺 DiaPredict</div>'
        '<div class="side-sub">AI-Based Diabetes Prediction System</div>'
        '<div class="status">● Model loaded successfully</div>'
        '<div class="step"><div class="n">1</div><div class="tx"><b>Enter details</b><br>Go through the five tabs and fill in the patient information.</div></div>'
        '<div class="step"><div class="n">2</div><div class="tx"><b>Click Predict</b><br>On the last tab, press the button to process all 28 indicators.</div></div>'
        '<div class="step"><div class="n">3</div><div class="tx"><b>Read the result</b><br>See the predicted label and the model-estimated probability.</div></div>'
        '<div class="side-card">📚 <b style="color:#fff">About</b><br>Supervised classification with a '
        'Random Forest trained on patient health records. Built for learning and demonstration.</div>'
        '<div class="side-card">⚠️ Not a substitute for professional medical advice.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# 6. Patient health information form
# --------------------------------------------------

st.markdown(
    '<div class="sec-title">Enter Health Information</div>'
    '<div class="sec-sub">Go through all five tabs in order. The Predict button appears on the last tab, after every section is complete.</div>',
    unsafe_allow_html=True,
)

with st.form("diabetes_prediction_form"):

    tab_personal, tab_life, tab_hist, tab_body, tab_blood = st.tabs(
        ["👤 Personal", "🏃 Lifestyle", "❤️ Medical History", "📏 Body", "🧪 Cholesterol & Glucose"]
    )

    # ---------- Personal ----------
    with tab_personal:
        tab_header("👤", "#e0e7ff", "Personal Information", "Basic demographic details")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=90, value=35)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])

        with col2:
            ethnicity = st.selectbox("Ethnicity", ["Asian", "Black", "Hispanic", "Other", "White"])
            education_level = st.selectbox(
                "Education Level", ["Graduate", "Highschool", "No formal", "Postgraduate"]
            )

        with col3:
            income_level = st.selectbox(
                "Income Level", ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"]
            )
            employment_status = st.selectbox(
                "Employment Status", ["Employed", "Retired", "Student", "Unemployed"]
            )

        smoking_status = st.selectbox("Smoking Status", ["Current", "Former", "Never"])

        st.markdown('<div class="next-hint">Step 1 of 5 · continue to the <b>Lifestyle</b> tab →</div>', unsafe_allow_html=True)

    # ---------- Lifestyle ----------
    with tab_life:
        tab_header("🏃", "#ccfbf1", "Lifestyle Information", "Daily habits and activity")
        col1, col2, col3 = st.columns(3)

        with col1:
            alcohol_consumption_per_week = st.number_input(
                "Alcohol Consumption (units/week)", min_value=0, max_value=10, value=0
            )
            physical_activity_minutes_per_week = st.number_input(
                "Physical Activity (minutes/week)", min_value=0, max_value=833, value=100
            )

        with col2:
            diet_score = st.number_input(
                "Diet Score", min_value=0.0, max_value=10.0, value=6.0, step=0.5
            )
            sleep_hours_per_day = st.number_input(
                "Sleep Hours per Day", min_value=3.0, max_value=10.0, value=7.0, step=0.5
            )

        with col3:
            screen_time_hours_per_day = st.number_input(
                "Screen Time (hours/day)", min_value=0.5, max_value=16.8, value=6.0, step=0.5
            )

        st.markdown('<div class="next-hint">Step 2 of 5 · continue to the <b>Medical History</b> tab →</div>', unsafe_allow_html=True)

    # ---------- Medical history ----------
    with tab_hist:
        tab_header("❤️", "#ffe4e6", "Medical History", "Family and personal health background")
        col1, col2, col3 = st.columns(3)

        with col1:
            family_history_diabetes = st.selectbox(
                "Family History of Diabetes", options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
        with col2:
            hypertension_history = st.selectbox(
                "History of Hypertension", options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )
        with col3:
            cardiovascular_history = st.selectbox(
                "History of Cardiovascular Disease", options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        st.markdown('<div class="next-hint">Step 3 of 5 · continue to the <b>Body</b> tab →</div>', unsafe_allow_html=True)

    # ---------- Body measurements ----------
    with tab_body:
        tab_header("📏", "#fef3c7", "Body Measurements", "Physical measurements and vitals")
        col1, col2, col3 = st.columns(3)

        with col1:
            bmi = st.number_input("BMI", min_value=15.0, max_value=39.2, value=25.6, step=0.1)
            waist_to_hip_ratio = st.number_input(
                "Waist-to-Hip Ratio", min_value=0.67, max_value=1.06, value=0.86, step=0.01
            )

        with col2:
            systolic_bp = st.number_input(
                "Systolic Blood Pressure (mmHg)", min_value=90, max_value=179, value=116
            )
            diastolic_bp = st.number_input(
                "Diastolic Blood Pressure (mmHg)", min_value=50, max_value=110, value=75
            )

        with col3:
            heart_rate = st.number_input(
                "Heart Rate (beats/minute)", min_value=40, max_value=105, value=70
            )

        st.markdown('<div class="next-hint">Step 4 of 5 · continue to the <b>Cholesterol &amp; Glucose</b> tab to finish →</div>', unsafe_allow_html=True)

    # ---------- Cholesterol & glucose ----------
    with tab_blood:
        tab_header("🧪", "#dbeafe", "Cholesterol and Glucose Measurements", "Laboratory values")
        col1, col2, col3 = st.columns(3)

        with col1:
            cholesterol_total = st.number_input(
                "Total Cholesterol", min_value=100, max_value=318, value=186
            )
            hdl_cholesterol = st.number_input(
                "HDL Cholesterol", min_value=20, max_value=98, value=54
            )

        with col2:
            ldl_cholesterol = st.number_input(
                "LDL Cholesterol", min_value=50, max_value=263, value=102
            )
            triglycerides = st.number_input(
                "Triglycerides", min_value=30, max_value=344, value=121
            )

        with col3:
            glucose_fasting = st.number_input(
                "Fasting Glucose", min_value=60, max_value=172, value=111
            )
            glucose_postprandial = st.number_input(
                "Postprandial Glucose", min_value=70, max_value=287, value=160
            )

        col4, col5, _ = st.columns(3)
        with col4:
            insulin_level = st.number_input(
                "Insulin Level", min_value=2.0, max_value=32.22, value=8.79, step=0.1
            )
        with col5:
            hba1c = st.number_input(
                "HbA1c (%)", min_value=4.0, max_value=9.8, value=6.52, step=0.1
            )

        st.divider()
        st.markdown(
            '<div class="final-note">✅ <b>Final step:</b> you have reached the last section. '
            'Make sure the previous tabs are filled in, then press Predict.</div>',
            unsafe_allow_html=True,
        )
        try:
            submitted = st.form_submit_button(
                "🔍  Predict Diabetes Label", type="primary", width="stretch"
            )
        except TypeError:  # older Streamlit versions
            submitted = st.form_submit_button(
                "🔍  Predict Diabetes Label", type="primary", use_container_width=True
            )



# --------------------------------------------------
# 7. Prepare the input and make a prediction
# --------------------------------------------------

def gauge_svg(p, color):
    arc = 251.33
    return (
        '<svg viewBox="0 0 200 125" class="gauge">'
        '<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#e5e7eb" stroke-width="16" stroke-linecap="round"/>'
        f'<path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" '
        f'stroke-linecap="round" stroke-dasharray="{arc * p:.2f} {arc}" class="gauge-fill"/>'
        f'<text x="100" y="88" text-anchor="middle" class="gauge-num">{p * 100:.1f}%</text>'
        '<text x="100" y="112" text-anchor="middle" class="gauge-cap">model probability</text>'
        '</svg>'
    )


if submitted:

    # Keep these names and spellings identical to training.
    patient_data = {
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "education_level": education_level,
        "income_level": income_level,
        "employment_status": employment_status,
        "smoking_status": smoking_status,
        "alcohol_consumption_per_week": alcohol_consumption_per_week,
        "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
        "diet_score": diet_score,
        "sleep_hours_per_day": sleep_hours_per_day,
        "screen_time_hours_per_day": screen_time_hours_per_day,
        "family_history_diabetes": family_history_diabetes,
        "hypertension_history": hypertension_history,
        "cardiovascular_history": cardiovascular_history,
        "bmi": bmi,
        "waist_to_hip_ratio": waist_to_hip_ratio,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "cholesterol_total": cholesterol_total,
        "hdl_cholesterol": hdl_cholesterol,
        "ldl_cholesterol": ldl_cholesterol,
        "triglycerides": triglycerides,
        "glucose_fasting": glucose_fasting,
        "glucose_postprandial": glucose_postprandial,
        "insulin_level": insulin_level,
        "hba1c": hba1c,
    }

    input_df = pd.DataFrame([patient_data])

    try:
        # Apply the exact preprocessing used during training.
        processed_input = preprocessor.transform(input_df)

        # Predict the dataset label.
        prediction = int(model.predict(processed_input)[0])

        # Get the model's estimated probability for class 1.
        class_index = list(model.classes_).index(1)
        probability = float(model.predict_proba(processed_input)[0][class_index])

        # ---- Presentation ----
        if probability < 0.30:
            band, band_color = "Low", "#10b981"
        elif probability < 0.60:
            band, band_color = "Moderate", "#f59e0b"
        else:
            band, band_color = "High", "#ef4444"

        if prediction == 1:
            cls, pill, headline, icon = "pos", "● DIABETIC", "Predicted label: Diabetic (1)", "🔴"
        else:
            cls, pill, headline, icon = "neg", "● NON-DIABETIC", "Predicted label: Non-Diabetic (0)", "🟢"

        st.markdown('<div class="sec-title" style="margin-top:1.8rem">Prediction Result</div>',
                    unsafe_allow_html=True)

        st.markdown(
            f'<div class="result {cls}">'
            f'<div class="left">{gauge_svg(probability, band_color)}</div>'
            f'<div class="right"><span class="pill {cls}">{pill}</span>'
            f'<h2>{headline}</h2>'
            '<p>The percentage shows how strongly the Random Forest model associates this '
            'profile with label 1 (diabetic) in the training data.</p>'
            f'<span class="band"><span class="dot" style="background:{band_color}"></span>'
            f'Probability band: <b>{band}</b></span></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="tiles">'
            f'<div class="tile"><div class="k">Age</div><div class="v">{age}</div><div class="u">years</div></div>'
            f'<div class="tile"><div class="k">BMI</div><div class="v">{bmi:.1f}</div><div class="u">kg/m²</div></div>'
            f'<div class="tile"><div class="k">Blood Pressure</div><div class="v">{systolic_bp}/{diastolic_bp}</div><div class="u">mmHg</div></div>'
            f'<div class="tile"><div class="k">Fasting Glucose</div><div class="v">{glucose_fasting}</div><div class="u">mg/dL</div></div>'
            f'<div class="tile"><div class="k">HbA1c</div><div class="v">{hba1c:.1f}</div><div class="u">%</div></div>'
            f'<div class="tile"><div class="k">Cholesterol</div><div class="v">{cholesterol_total}</div><div class="u">mg/dL</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "This percentage is the model's estimated probability "
            "for the dataset label, not a clinically validated "
            "estimate of an individual's medical risk."
        )

        with st.expander("View submitted input details", expanded=True):
            yn = lambda v: "Yes" if v == 1 else "No"
            groups = [
                ("👤", "Personal Information", [
                    ("Age", f"{age} years"), ("Gender", gender), ("Ethnicity", ethnicity),
                    ("Education Level", education_level), ("Income Level", income_level),
                    ("Employment Status", employment_status), ("Smoking Status", smoking_status)]),
                ("🏃", "Lifestyle", [
                    ("Alcohol Consumption", f"{alcohol_consumption_per_week} units/week"),
                    ("Physical Activity", f"{physical_activity_minutes_per_week} min/week"),
                    ("Diet Score", f"{diet_score}"), ("Sleep", f"{sleep_hours_per_day} hrs/day"),
                    ("Screen Time", f"{screen_time_hours_per_day} hrs/day")]),
                ("❤️", "Medical History", [
                    ("Family History of Diabetes", yn(family_history_diabetes)),
                    ("Hypertension", yn(hypertension_history)),
                    ("Cardiovascular Disease", yn(cardiovascular_history))]),
                ("📏", "Body Measurements", [
                    ("BMI", f"{bmi:.1f}"), ("Waist-to-Hip Ratio", f"{waist_to_hip_ratio:.2f}"),
                    ("Systolic BP", f"{systolic_bp} mmHg"), ("Diastolic BP", f"{diastolic_bp} mmHg"),
                    ("Heart Rate", f"{heart_rate} bpm")]),
                ("🧪", "Cholesterol & Glucose", [
                    ("Total Cholesterol", f"{cholesterol_total}"), ("HDL Cholesterol", f"{hdl_cholesterol}"),
                    ("LDL Cholesterol", f"{ldl_cholesterol}"), ("Triglycerides", f"{triglycerides}"),
                    ("Fasting Glucose", f"{glucose_fasting}"), ("Postprandial Glucose", f"{glucose_postprandial}"),
                    ("Insulin Level", f"{insulin_level:.2f}"), ("HbA1c", f"{hba1c:.2f} %")]),
            ]
            cards = ""
            for icon, title, rows in groups:
                body = "".join(
                    f'<div class="drow"><span class="dk">{k}</span><span class="dv">{v}</span></div>'
                    for k, v in rows
                )
                cards += f'<div class="dcard"><div class="dtitle">{icon} {title}</div>{body}</div>'
            st.markdown(f'<div class="dgrid">{cards}</div>', unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the model, preprocessing pipeline and "
            "input feature names match those used during training."
        )


# --------------------------------------------------
# 8. Footer
# --------------------------------------------------

st.markdown(
    '<div class="footer"><b>AI-Based Diabetes Prediction System</b> &nbsp;|&nbsp; Academic project '
    '&nbsp;|&nbsp; Machine Learning with Random Forest</div>',
    unsafe_allow_html=True,
)