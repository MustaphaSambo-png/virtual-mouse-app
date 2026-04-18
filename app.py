"""
Precision Virtual Mouse — Streamlit Edition v3
Merged Eye Tracking and Hand Tracking via MediaPipe Tasks API.
Python 3.13+ Compatible.
"""

import streamlit as st
import cv2
import numpy as np
import pyautogui
import time
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Precision Virtual Mouse",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Eye constants
LEFT_IRIS = [469, 470, 471, 472]     # Left physical eye (due to mirror)
RIGHT_IRIS = [474, 475, 476, 477]    # Right physical eye
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380] 
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144] 

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0
SCREEN_W, SCREEN_H = pyautogui.size()

# ─── Helper Functions ───────────────────────────────────────────────────────────

def calculate_ear(landmarks, indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h  = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 1.0

def iris_centroid(landmarks, indices):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    return pts.mean(axis=0)

def get_distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def is_finger_extended(tip, mcp, wrist):
    return get_distance(tip, wrist) > get_distance(mcp, wrist) * 1.1

def draw_calibration_overlay(frame, step, cam_w, cam_h):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (cam_w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if step == 1:
        msg = ">> EYE: LOOK TOP-LEFT then click [Record Top-Left] <<"
        color = (0, 200, 255)
        cv2.circle(frame, (40, 40), 20, color, -1)
        cv2.circle(frame, (40, 40), 25, (255, 255, 255), 2)
    elif step == 2:
        msg = ">> EYE: LOOK BOTTOM-RIGHT then click [Record Bottom-Right] <<"
        color = (0, 255, 100)
        cv2.circle(frame, (cam_w - 40, cam_h - 40), 20, color, -1)
        cv2.circle(frame, (cam_w - 40, cam_h - 40), 25, (255, 255, 255), 2)
    else:
        return frame

    cv2.putText(frame, msg, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return frame

# ─── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 700; margin-bottom: 0;
    }
    .sub-title { color: #888; font-size: 1rem; margin-top: -8px; margin-bottom: 24px; }
    .status-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px; padding: 16px 20px; margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .status-card h4 {
        margin: 0 0 4px 0; color: #667eea !important;
        font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;
    }
    .status-card p { margin: 0; font-size: 1.3rem; font-weight: 600; color: #fff !important; }
    .calibrating { border-color: #f5af19 !important; }
    .calibrating h4 { color: #f5af19 !important; }
    .active { border-color: #00e676 !important; }
    .active h4 { color: #00e676 !important; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
defaults = {
    "running": False,
    "mode": "Hand ✋",          # "Eye 👁️" or "Hand ✋"
    
    # Shared / Cross Tracking
    "prev_x": SCREEN_W / 2,
    "prev_y": SCREEN_H / 2,

    # Eye State
    "eye_cal_phase": 0,         # 0=not started, 1=wait TL, 2=wait BR, 3=done
    "cal_tl": None,
    "cal_br": None,
    "is_blinking": False,
    "record_tl": False,
    "record_br": False,
    
    # Hand State
    "hand_cal_phase": 0,        # 0=not started, 1=calibrating 3s, 2=done
    "hand_cal_start_time": 0.0,
    "hand_cal_samples": [],
    "rest_wrist_x": 0.5,
    "rest_wrist_y": 0.5,
    "rest_depth": 0.1,
    "is_dragging": False,
    "prev_scroll_y": 0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🕹️ Mode Selection")
    mode = st.radio("Select tracking method:", ["Hand ✋", "Eye 👁️"], 
                    index=0 if st.session_state.mode == "Hand ✋" else 1)
    
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        st.session_state.running = False
        st.rerun()

    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    
    if mode == "Eye 👁️":
        alpha = st.slider("Smoothing (α)", 0.05, 1.0, 0.2, 0.05, help="Lower = smoother but laggier.")
        ear_thresh = st.slider("EAR Threshold", 0.10, 0.35, 0.21, 0.01)
        
        st.markdown("---")
        st.markdown("## 🎯 Eye Calibration")
        if st.button("📍 Record Top-Left", width='stretch', disabled=(st.session_state.eye_cal_phase != 1)):
            st.session_state.record_tl = True
        if st.button("📍 Record Bottom-Right", width='stretch', disabled=(st.session_state.eye_cal_phase != 2)):
            st.session_state.record_br = True
        if st.button("🔄 Recalibrate", width='stretch'):
            st.session_state.eye_cal_phase = 1
            st.session_state.cal_tl = None
            st.session_state.cal_br = None
            st.session_state.record_tl = False
            st.session_state.record_br = False

    elif mode == "Hand ✋":
        alpha = st.slider("Smoothing (α)", 0.05, 1.0, 0.15, 0.05, help="Hand shakes more, Keep low (0.15) for EMA.")
        pinch_thresh = st.slider("Pinch Threshold", 0.02, 0.15, 0.05, 0.01)
        scroll_sens = st.slider("Scroll Sensitivity", 10, 100, 50, 10)
        
        st.markdown("---")
        st.markdown("## 🎯 Hand Calibration")
        st.caption("Hold your hand comfortably in front of you. Click below to recalibrate resting position.")
        if st.button("🔄 Recalibrate", width='stretch'):
            st.session_state.hand_cal_phase = 1
            st.session_state.hand_cal_start_time = time.time()
            st.session_state.hand_cal_samples = []

    st.markdown("---")
    st.markdown("## 📖 Guide")
    if mode == "Hand ✋":
        st.markdown("""
        1. **Pinch** thumb and index to Click.
        2. Raise **Index + Middle** together & move up/down to **Scroll**.
        3. Hand distance affects tracking box scaling (moves further = higher sensitivity).
        """)
    else:
        st.markdown("""
        1. Look at corners & record bounds.
        2. **Blink** = Left Click
        """)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎯 Precision Virtual Mouse</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">Running in {st.session_state.mode} Mode · Powered by MediaPipe</p>', unsafe_allow_html=True)

col1, col2, _ = st.columns([1, 1, 4])
with col1:
    if st.button("▶️ Start", width='stretch', type="primary"):
        st.session_state.running = True
        if mode == "Eye 👁️" and st.session_state.eye_cal_phase == 0:
            st.session_state.eye_cal_phase = 1
        elif mode == "Hand ✋" and st.session_state.hand_cal_phase == 0:
            st.session_state.hand_cal_phase = 1
            st.session_state.hand_cal_start_time = time.time()
with col2:
    if st.button("⏹️ Stop", width='stretch'):
        st.session_state.running = False

# Status Cards
s1, s2, s3, s4 = st.columns(4)

if mode == "Eye 👁️":
    phase = st.session_state.eye_cal_phase
    phase_labels = {0: "Not Started", 1: "Look Top-Left ↖", 2: "Look Bottom-Right ↘", 3: "✅ Active"}
    val3 = f"{ear_thresh:.2f}"
    val4 = "-"
else:
    phase = st.session_state.hand_cal_phase
    phase_labels = {0: "Not Started", 1: "⏳ Calibrating (Hold still)", 2: "✅ Active"}
    val3 = f"{pinch_thresh:.2f}"
    val4 = str(scroll_sens)

phase_cls = "calibrating" if phase in (1, 2) and (mode=="Eye 👁️" or phase==1) else "active" if phase > 1 and mode=="Eye 👁️" or phase == 2 and mode=="Hand ✋" else ""

with s1: st.markdown(f'<div class="status-card {phase_cls}"><h4>Status</h4><p>{phase_labels[phase]}</p></div>', unsafe_allow_html=True)
with s2: st.markdown(f'<div class="status-card"><h4>Smoothing α</h4><p>{alpha}</p></div>', unsafe_allow_html=True)
with s3: st.markdown(f'<div class="status-card"><h4>{"EAR Thresh" if mode=="Eye 👁️" else "Pinch Thresh"}</h4><p>{val3}</p></div>', unsafe_allow_html=True)
with s4: st.markdown(f'<div class="status-card"><h4>{"Blink Ms" if mode=="Eye 👁️" else "Scroll Sens"}</h4><p>{val4}</p></div>', unsafe_allow_html=True)

video_placeholder = st.empty()
info_placeholder = st.empty()

# ─── Main Loop ──────────────────────────────────────────────────────────────────
if st.session_state.running:

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        st.session_state.running = False
        st.stop()

    try:
        if mode == "Eye 👁️":
            options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
                num_faces=1,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
        else:
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                num_hands=1,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)

        while st.session_state.running:
            ok, frame = cap.read()
            if not ok: continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            # ═══════════════════════════════════════════════════════════════════════
            # EYE MODE LOGIC
            # ═══════════════════════════════════════════════════════════════════════
            if mode == "Eye 👁️":
                cal = st.session_state.eye_cal_phase
                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    iris_xy = iris_centroid(lm, LEFT_IRIS)
                    raw_x, raw_y = float(iris_xy[0]), float(iris_xy[1])
                    
                    tx, ty = int(raw_x * CAM_WIDTH), int(raw_y * CAM_HEIGHT)
                    cv2.circle(frame, (tx, ty), 6, (0, 255, 0), -1)
                    cv2.circle(frame, (tx, ty), 10, (0, 255, 0), 2)

                    if st.session_state.record_tl and cal == 1:
                        st.session_state.cal_tl = (raw_x, raw_y)
                        st.session_state.eye_cal_phase = 2
                        st.session_state.record_tl = False
                        st.rerun()

                    if st.session_state.record_br and cal == 2:
                        st.session_state.cal_br = (raw_x, raw_y)
                        st.session_state.eye_cal_phase = 3
                        st.session_state.record_br = False
                        st.rerun()

                    if cal in (1, 2):
                        frame = draw_calibration_overlay(frame, cal, CAM_WIDTH, CAM_HEIGHT)
                    elif cal == 3:
                        tl, br = st.session_state.cal_tl, st.session_state.cal_br
                        pad_x, pad_y = abs(br[0] - tl[0]) * 0.10, abs(br[1] - tl[1]) * 0.10
                        min_x, max_x = min(tl[0], br[0]) - pad_x, max(tl[0], br[0]) + pad_x
                        min_y, max_y = min(tl[1], br[1]) - pad_y, max(tl[1], br[1]) + pad_y

                        sx = float(np.interp(raw_x, [min_x, max_x], [0, SCREEN_W]))
                        sy = float(np.interp(raw_y, [min_y, max_y], [0, SCREEN_H]))
                        curr_x, curr_y = alpha * sx + (1 - alpha) * st.session_state.prev_x, alpha * sy + (1 - alpha) * st.session_state.prev_y
                        st.session_state.prev_x, st.session_state.prev_y = curr_x, curr_y
                        pyautogui.moveTo(int(curr_x), int(curr_y))

                        cv2.putText(frame, f"Eye Cursor: ({int(curr_x)}, {int(curr_y)})", (10, CAM_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    ear_l, ear_r = calculate_ear(lm, LEFT_EYE_EAR), calculate_ear(lm, RIGHT_EYE_EAR)
                    
                    currently_blinking = (ear_l < ear_thresh) and (ear_r < ear_thresh)
                    if currently_blinking:
                        if not st.session_state.is_blinking:
                            st.session_state.is_blinking = True
                            if cal == 3: pyautogui.click(button='left')
                    else:
                        st.session_state.is_blinking = False

                    cl, cr = (0, 255, 0) if ear_l >= ear_thresh else (0, 0, 255), (0, 255, 0) if ear_r >= ear_thresh else (0, 0, 255)
                    cv2.putText(frame, f"L-EAR: {ear_l:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cl, 2)
                    cv2.putText(frame, f"R-EAR: {ear_r:.2f}", (CAM_WIDTH - 160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cr, 2)
                else:
                    cv2.putText(frame, "No face detected", (CAM_WIDTH // 2 - 100, CAM_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # ═══════════════════════════════════════════════════════════════════════
            # HAND MODE LOGIC
            # ═══════════════════════════════════════════════════════════════════════
            elif mode == "Hand ✋":
                cal = st.session_state.hand_cal_phase
                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    wrist, thumb_tip, index_mcp, index_tip = lm[0], lm[4], lm[5], lm[8]
                    middle_mcp, middle_tip = lm[9], lm[12]

                    for mark in [0, 4, 8, 12, 16, 20]:
                        mx, my = int(lm[mark].x * CAM_WIDTH), int(lm[mark].y * CAM_HEIGHT)
                        cv2.circle(frame, (mx, my), 4, (255, 100, 100), -1)

                    depth_dist = get_distance(wrist, middle_mcp)
                    raw_x, raw_y = index_tip.x, index_tip.y
                    
                    if cal == 1:
                        elapsed = time.time() - st.session_state.hand_cal_start_time
                        if elapsed < 3.0:
                            cv2.putText(frame, f"Calibrating Box... HOLD STILL ({int(3-elapsed)}s)", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                            st.session_state.hand_cal_samples.append((wrist.x, wrist.y, depth_dist))
                        else:
                            if len(st.session_state.hand_cal_samples) > 10:
                                arr = np.array(st.session_state.hand_cal_samples)
                                st.session_state.rest_wrist_x = np.mean(arr[:, 0])
                                st.session_state.rest_wrist_y = np.mean(arr[:, 1])
                                st.session_state.rest_depth = max(0.01, np.mean(arr[:, 2]))
                                st.session_state.hand_cal_phase = 2
                                st.rerun()
                            else:
                                st.session_state.hand_cal_start_time = time.time()
                                st.session_state.hand_cal_samples = []
                    elif cal == 2:
                        scale = st.session_state.rest_depth / max(0.01, depth_dist)
                        box_width, box_height = 0.4 * scale, 0.4 * scale
                        min_x, max_x = st.session_state.rest_wrist_x - box_width/2, st.session_state.rest_wrist_x + box_width/2
                        min_y, max_y = st.session_state.rest_wrist_y - box_height/2, st.session_state.rest_wrist_y + box_height/2

                        bx1, by1 = int(min_x * CAM_WIDTH), int(min_y * CAM_HEIGHT)
                        bx2, by2 = int(max_x * CAM_WIDTH), int(max_y * CAM_HEIGHT)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (100, 100, 100), 1)

                        sx = float(np.interp(raw_x, [min_x, max_x], [0, SCREEN_W]))
                        sy = float(np.interp(raw_y, [min_y, max_y], [0, SCREEN_H]))
                        curr_x = alpha * max(0, min(SCREEN_W - 1, sx)) + (1 - alpha) * st.session_state.prev_x
                        curr_y = alpha * max(0, min(SCREEN_H - 1, sy)) + (1 - alpha) * st.session_state.prev_y

                        inch_dist = get_distance(thumb_tip, index_tip)
                        index_ext = is_finger_extended(index_tip, index_mcp, wrist)
                        middle_ext = is_finger_extended(middle_tip, middle_mcp, wrist)
                        now = time.time()

                        if inch_dist < pinch_thresh:
                            tx, ty = int(raw_x * CAM_WIDTH), int(raw_y * CAM_HEIGHT)
                            cv2.circle(frame, (tx, ty), 15, (0, 255, 0), -1)
                            
                            if not st.session_state.is_dragging:
                                pyautogui.mouseDown(button='left')
                                st.session_state.is_dragging = True
                            
                            pyautogui.moveTo(int(curr_x), int(curr_y))
                                
                        elif index_ext and middle_ext and get_distance(index_tip, middle_tip) < 0.08:
                            if st.session_state.is_dragging:
                                pyautogui.mouseUp(button='left')
                                st.session_state.is_dragging = False
                                
                            tx, ty = int(raw_x * CAM_WIDTH), int(raw_y * CAM_HEIGHT)
                            cv2.circle(frame, (tx, ty), 15, (255, 100, 0), -1)
                            cv2.putText(frame, "SCROLL MODE", (tx+20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
                            
                            if st.session_state.prev_scroll_y != 0.0:
                                dy = -1 * (curr_y - st.session_state.prev_scroll_y)
                                if abs(dy) > 2: pyautogui.scroll(int(dy * scroll_sens))
                            st.session_state.prev_scroll_y = curr_y
                        else:
                            if st.session_state.is_dragging:
                                pyautogui.mouseUp(button='left')
                                st.session_state.is_dragging = False
                                
                            pyautogui.moveTo(int(curr_x), int(curr_y))
                            st.session_state.prev_scroll_y = 0.0

                        st.session_state.prev_x, st.session_state.prev_y = curr_x, curr_y
                        cv2.putText(frame, f"Hand Cursor: ({int(curr_x)}, {int(curr_y)})", (10, CAM_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    cv2.putText(frame, "No hand detected", (CAM_WIDTH // 2 - 100, CAM_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_rgb, channels="RGB", width='stretch')

    finally:
        cap.release()

    info_placeholder.success("✅ Tracking stopped.")
else:
    st.info("👆 Press **▶ Start** to begin tracking.")
