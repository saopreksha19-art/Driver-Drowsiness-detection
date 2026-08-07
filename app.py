import time
import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc import WebRtcMode

from webrtc_processor import VideoProcessor
from detection import DrowsinessDetector

# =====================================================
# HTML RENDER HELPER
# =====================================================
def render_html(markup: str):
    """st.markdown() wrapper that strips leading indentation so
    Markdown doesn't mistake indented HTML for a code block."""
    dedented = "\n".join(line.lstrip() for line in markup.strip().split("\n"))
    st.markdown(dedented, unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================
if "running" not in st.session_state:
    st.session_state.running = False

if "detector" not in st.session_state:
    st.session_state.detector = None

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon=None,
    layout="wide",
)

# =====================================================
# DESIGN TOKENS + GLOBAL STYLE
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
    --bg:#0A0C10;
    --surface:#14181F;
    --surface-raised:#1B2029;
    --ink:#F0F3F7;
    --muted:#7C8798;
    --line:#262C36;
    --teal:#2FD9C4;
    --teal-soft:rgba(47,217,196,.10);
    --amber:#FFB020;
    --amber-soft:rgba(255,176,32,.10);
    --red:#FF4538;
    --red-soft:rgba(255,69,56,.12);
    --neutral-soft:#1E232C;
}

/* =====================================================
   PAGE
===================================================== */
.stApp{
    background:
        radial-gradient(circle at top right, rgba(47,217,196,.08), transparent 35%),
        radial-gradient(circle at bottom left, rgba(255,69,56,.05), transparent 35%),
        linear-gradient(180deg,#090B0F,#11161D);
}

.main .block-container{
    max-width:1400px;
    padding-top:2.5rem;
    padding-bottom:3rem;
    padding-left:2rem;
    padding-right:2rem;
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
}

h1, h2, h3{
    letter-spacing:-0.5px;
}

/* =====================================================
   HIDE STREAMLIT CHROME
===================================================== */
#MainMenu{ visibility:hidden; }
header{ visibility:hidden; }
footer{ visibility:hidden; }
[data-testid="stDecoration"]{ display:none; }
[data-testid="stStatusWidget"]{ display:none; }
[data-testid="stToolbar"]{ display:none; }

/* =====================================================
   SCROLLBAR
===================================================== */
::-webkit-scrollbar{ width:8px; }
::-webkit-scrollbar-thumb{ background:var(--line); border-radius:20px; }
::-webkit-scrollbar-track{ background:var(--bg); }

/* =====================================================
   TYPOGRAPHY
===================================================== */
h1, h2, h3{
    font-family:'Rajdhani', sans-serif;
    font-weight:700;
    text-transform:uppercase;
    color:var(--ink) !important;
}

p, label, li{
    color:var(--muted) !important;
    font-size:16px;
}

/* =====================================================
   SIGNATURE: BLINKING EYE GLYPH
===================================================== */
.eye-glyph{
    position:relative;
    width:20px;
    height:12px;
    flex-shrink:0;
}

.eye-glyph .lid{
    position:absolute;
    inset:0;
    border:1.6px solid currentColor;
    border-radius:50%;
    overflow:hidden;
}

.eye-glyph .lid::after{
    content:"";
    position:absolute;
    left:50%;
    top:50%;
    width:6px;
    height:6px;
    border-radius:50%;
    background:currentColor;
    transform:translate(-50%,-50%);
    animation:blink 4.5s ease-in-out infinite;
}

.eye-glyph.shut .lid::after{
    animation:none;
    transform:translate(-50%,-50%) scaleY(0);
}

.eye-glyph.shut .lid{ border-radius:2px; height:2px; top:5px; }

@keyframes blink{
    0%, 92%, 100% { transform:translate(-50%,-50%) scaleY(1); }
    96%           { transform:translate(-50%,-50%) scaleY(0.05); }
}

/* =====================================================
   EYEBROW / LIVE BADGE
===================================================== */
.eyebrow{
    display:inline-flex;
    align-items:center;
    gap:9px;
    font-family:'IBM Plex Mono', monospace;
    font-size:12px;
    font-weight:600;
    letter-spacing:1.8px;
    text-transform:uppercase;
    padding:7px 14px 7px 12px;
    border-radius:100px;
    border:1px solid var(--line);
    background:var(--surface);
    color:var(--muted);
}

.eyebrow.live{
    color:var(--teal);
    border-color:rgba(47,217,196,.35);
    background:var(--teal-soft);
}

.eyebrow.live .eye-glyph{ color:var(--teal); }
.eyebrow:not(.live) .eye-glyph{ color:var(--muted); }

/* =====================================================
   GAUGE CARDS (status / score / alarm)
===================================================== */
.gauge-row{
    display:grid;
    grid-template-columns:repeat(3, 1fr);
    gap:14px;
    margin-bottom:24px;
}

.gauge-card{
    position:relative;
    background:#171C24;
    border:1px solid rgba(255,255,255,.05);
    border-radius:18px;
    padding:26px;
    overflow:hidden;
    transition:.3s ease;
    box-shadow:0 10px 30px rgba(0,0,0,.35);
}

.gauge-card:hover{
    transform:translateY(-3px);
    border-color:rgba(47,217,196,.3);
}

.gauge-card::before{
    content:"";
    position:absolute;
    top:0; left:0; right:0;
    height:3px;
    background:repeating-linear-gradient(
        90deg,
        var(--neutral-soft) 0px, var(--neutral-soft) 6px,
        transparent 6px, transparent 9px
    );
}

.gauge-card.state-good::before{
    background:repeating-linear-gradient(90deg, var(--teal) 0px, var(--teal) 6px, transparent 6px, transparent 9px);
}
.gauge-card.state-warn::before{
    background:repeating-linear-gradient(90deg, var(--amber) 0px, var(--amber) 6px, transparent 6px, transparent 9px);
}
.gauge-card.state-bad::before{
    background:repeating-linear-gradient(90deg, var(--red) 0px, var(--red) 6px, transparent 6px, transparent 9px);
    animation:tick-flash 1s steps(1) infinite;
}

@keyframes tick-flash{ 50%{ opacity:.35; } }

.gauge-label{
    font-family:'IBM Plex Mono', monospace;
    font-size:11.5px;
    font-weight:600;
    letter-spacing:1.6px;
    text-transform:uppercase;
    color:var(--muted);
    margin:0 0 10px 0;
}

.gauge-value{
    font-family:'IBM Plex Mono', monospace;
    font-size:32px;
    font-weight:600;
    color:var(--ink);
    display:flex;
    align-items:center;
    gap:11px;
    font-variant-numeric:tabular-nums;
}

.gauge-value .indicator{
    width:9px;
    height:9px;
    border-radius:50%;
    background:var(--muted);
    flex-shrink:0;
}

.state-good .gauge-value .indicator{ background:var(--teal); box-shadow:0 0 0 4px var(--teal-soft); }
.state-warn .gauge-value .indicator{ background:var(--amber); box-shadow:0 0 0 4px var(--amber-soft); }
.state-bad  .gauge-value .indicator{
    background:var(--red);
    box-shadow:0 0 0 4px var(--red-soft);
}

.state-good .gauge-value .eye-glyph{ color:var(--teal); }
.state-warn .gauge-value .eye-glyph{ color:var(--amber); }
.state-bad  .gauge-value .eye-glyph{ color:var(--red); }

/* =====================================================
   CARDS
===================================================== */
.panel{
    background:var(--surface);
    border:1px solid var(--line);
    border-radius:12px;
    padding:24px 26px;
    margin-bottom:20px;
    transition:.25s ease;
}

.panel:hover{
    border-color:rgba(47,217,196,.3);
    transform:translateY(-2px);
}

.panel h3{
    margin:0 0 6px 0;
    font-size:19px;
    font-weight:700;
}

.panel p{
    margin:0;
    font-size:15px;
}

.panel-head{
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin-bottom:6px;
}

/* =====================================================
   CAMERA FRAME
===================================================== */
.camera-frame{
    position:relative;
    border-radius:18px;
    padding:20px;
    background:#11161D;
    border:1px solid rgba(255,255,255,.05);
    box-shadow:0 12px 40px rgba(0,0,0,.4);
}
.camera-frame.active{
    background-color:#04060A;
}

.camera-frame::before,
.camera-frame::after{
    content:"";
    position:absolute;
    width:18px;
    height:18px;
    border-color:var(--line);
    border-style:solid;
    border-width:0;
}

.camera-frame::before{ top:8px; left:8px; border-top-width:2px; border-left-width:2px; }
.camera-frame::after{ bottom:8px; right:8px; border-bottom-width:2px; border-right-width:2px; }

.camera-frame.active::before,
.camera-frame.active::after{ border-color:var(--teal); }

/* =====================================================
   BUTTONS
===================================================== */
.stButton > button{
    width:100%;
    height:56px;
    border-radius:14px;
    font-size:15px;
    font-weight:700;
    letter-spacing:1px;
    transition:.25s;
}

div[data-testid="column"]:nth-of-type(1) .stButton > button{
    background:var(--teal);
    color:#04140F;
    border:1px solid var(--teal);
}

div[data-testid="column"]:nth-of-type(1) .stButton > button:hover{
    background:#27C1AE;
    box-shadow:0 0 0 4px var(--teal-soft);
}

div[data-testid="column"]:nth-of-type(2) .stButton > button{
    background:transparent;
    color:var(--red);
    border:1px solid rgba(255,69,56,.4);
}

div[data-testid="column"]:nth-of-type(2) .stButton > button:hover{
    background:var(--red-soft);
    box-shadow:0 0 0 4px var(--red-soft);
}

/* =====================================================
   ALERT
===================================================== */
div[data-testid="stAlert"]{
    background:var(--surface);
    border:1px solid var(--line);
    border-radius:10px;
}

div[data-testid="stAlert"] p{ color:var(--muted) !important; }

/* =====================================================
   SIDEBAR
===================================================== */
section[data-testid="stSidebar"]{
    background:var(--surface);
    border-right:1px solid var(--line);
}

section[data-testid="stSidebar"] h2{
    font-size:19px;
    font-family:'Rajdhani', sans-serif;
    text-transform:uppercase;
}

section[data-testid="stSidebar"] strong{
    font-family:'IBM Plex Mono', monospace;
    font-size:12px;
    letter-spacing:1px;
    text-transform:uppercase;
    color:var(--teal) !important;
}

/* =====================================================
   DIVIDER / MISC
===================================================== */
hr{
    border-color:var(--line);
}

img{
    border-radius:14px;
}

.gauge-value{
    font-size:38px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# APPLICATION HEADER
# =====================================================
live = st.session_state.running

status_color = "#2FD9C4" if live else "#7C8798"
status_text = "● LIVE MONITORING" if live else "● SYSTEM STANDBY"

render_html(f"""
<div style="
    background:linear-gradient(135deg,#14181F,#1B2029);
    border:1px solid #262C36;
    border-radius:18px;
    padding:32px 36px;
    margin-bottom:28px;
    box-shadow:0 8px 30px rgba(0,0,0,.35);
">
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        flex-wrap:wrap;
        gap:20px;
    ">
        <div>
            <div style="
                color:{status_color};
                font-family:'IBM Plex Mono',monospace;
                font-size:13px;
                letter-spacing:1.5px;
                font-weight:600;
                margin-bottom:10px;
            ">
                {status_text}
            </div>

            <h1 style="
                margin:0;
                font-size:48px;
                font-weight:700;
                color:white;
                line-height:1.1;
            ">
                Driver Drowsiness Detection
            </h1>

            <p style="
                margin-top:12px;
                color:#A8B3C2;
                font-size:18px;
                line-height:1.6;
                max-width:720px;
            ">
                AI-powered real-time driver monitoring using
                <strong style="color:#2FD9C4;">OpenCV</strong>,
                <strong style="color:#2FD9C4;">TensorFlow</strong>,
                and a
                <strong style="color:#2FD9C4;">Convolutional Neural Network</strong>
                to detect prolonged eye closure and issue fatigue alerts.
            </p>
        </div>

        <div style="
            background:#0A0C10;
            border:1px solid #2FD9C4;
            border-radius:16px;
            padding:22px 28px;
            text-align:center;
            min-width:170px;
        ">
            <div style="
                font-size:14px;
                color:#7C8798;
                letter-spacing:1px;
                text-transform:uppercase;
            ">
                System Status
            </div>

            <div style="
                margin-top:10px;
                font-size:26px;
                font-weight:700;
                color:{status_color};
            ">
                {"ACTIVE" if live else "IDLE"}
            </div>
        </div>
    </div>
</div>
""")

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("Project Overview")

    st.write("""
**Model**
- Convolutional Neural Network (CNN)

**Framework**
- TensorFlow / Keras

**Computer Vision**
- OpenCV
- Haar Cascade Classifiers

**Frontend**
- Streamlit

**Dataset**
- MRL Eye Dataset

**Author**
- Preksha Sao
""")

    st.divider()

    st.subheader("Instructions")

    st.write("""
1. Click **Start Detection**
2. Allow webcam access
3. Keep your face centered
4. Click **Stop Detection** to exit
""")

# =====================================================
# SYSTEM STATUS DASHBOARD
# =====================================================
dashboard = st.empty()

def render_dashboard(status="Inactive", score=0, alarm=False):
    if alarm:
        state = "state-bad"
    elif status == "Drowsy":
        state = "state-warn"
    elif status == "Monitoring":
        state = "state-good"
    else:
        state = ""

    score_state = "state-bad" if alarm else ("state-warn" if int(score) >= 5 else state)
    eyes_shut = " shut" if status == "Drowsy" else ""

    dashboard_html = f"""
    <div class="gauge-row">
    <div class="gauge-card {state}">
    <p class="gauge-label">Status</p>
    <div class="gauge-value"><span class="eye-glyph{eyes_shut}"><span class="lid"></span></span>{status}</div>
    </div>
    <div class="gauge-card {score_state}">
    <p class="gauge-label">Drowsiness Score</p>
    <div class="gauge-value"><span class="indicator"></span>{score}</div>
    </div>
    <div class="gauge-card {"state-bad" if alarm else ""}">
    <p class="gauge-label">Alarm</p>
    <div class="gauge-value"><span class="indicator"></span>{"ON" if alarm else "OFF"}</div>
    </div>
    </div>
    """
    dashboard.markdown(
        "\n".join(line.lstrip() for line in dashboard_html.strip().split("\n")),
        unsafe_allow_html=True,
    )

render_dashboard()

# =====================================================
# LIVE CAMERA FEED
# =====================================================
st.markdown("<br>", unsafe_allow_html=True)

feed_badge = (
    '<span class="eyebrow live"><span class="eye-glyph"><span class="lid"></span></span>LIVE</span>'
    if live
    else '<span class="eyebrow"><span class="eye-glyph shut"><span class="lid"></span></span>OFFLINE</span>'
)

render_html(f"""
<div class="panel">
<div class="panel-head">
<h3>Live Camera Feed</h3>
{feed_badge}
</div>
<p>The webcam continuously monitors the driver's eye movements and detects prolonged eye closure in real time.</p>
</div>
""")

# =====================================================
# CAMERA PREVIEW
# =====================================================

frame_class = "camera-frame active" if live else "camera-frame"

render_html(f'<div class="{frame_class}">')

if st.session_state.running:
    webrtc_ctx = webrtc_streamer(
        key="driver-monitor",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

render_html("</div>")
# =====================================================
# DETECTION CONTROLS
# =====================================================
left, right = st.columns(2)

with left:
    start = st.button(
        "⏻  Start Detection",
        use_container_width=True,
        key="start_btn",
    )

with right:
    stop = st.button(
        "◼  Stop Detection",
        use_container_width=True,
        key="stop_btn",
    )

if start and not st.session_state.running:
    st.session_state.running = True
    st.rerun()

if stop and st.session_state.running:
    st.session_state.running = False
    st.rerun()

# =====================================================
# REAL-TIME DETECTION (WebRTC)
# =====================================================

if st.session_state.running:

    if (
        webrtc_ctx
        and webrtc_ctx.state.playing
        and webrtc_ctx.video_processor
    ):

        processor = webrtc_ctx.video_processor

        render_dashboard(
            status=processor.status,
            score=processor.score,
            alarm=processor.alarm,
        )

else:
    st.info("Camera is inactive. Click **Start Detection** to begin monitoring.")
# =====================================================
# ABOUT THE PROJECT
# =====================================================
st.markdown("<br>", unsafe_allow_html=True)

render_html("""
<div class="panel">
<h3>About the Project</h3>
<p>
This application performs real-time driver drowsiness detection
using a Convolutional Neural Network (CNN), OpenCV, TensorFlow,
and Haar Cascade classifiers. The system continuously monitors
eye movements and raises an alert when prolonged eye closure is
detected, helping reduce fatigue-related road accidents.
</p>
</div>
""")

# =====================================================
# FOOTER
# =====================================================
st.markdown("<br><br>", unsafe_allow_html=True)

render_html("""
<hr>
<div style="text-align:center;color:var(--muted);font-size:14px;padding-bottom:10px;font-family:'JetBrains Mono', monospace;letter-spacing:.5px;">
DRIVER DROWSINESS DETECTION
</div>
""")