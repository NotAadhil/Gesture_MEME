import sys

# Proactive dependency check to help users on clean Windows/Linux/macOS setups
try:
    import cv2
except ImportError:
    print("\n" + "="*80)
    print(" [ERROR] Failed to load OpenCV (cv2)!")
    print(" This is a common Windows issue usually caused by a missing dependency.")
    print(" Please try the following:")
    print(" 1. Run: pip install msvc-runtime")
    print(" 2. If you are on Windows 'N' edition, install the 'Media Feature Pack'.")
    print(" 3. Install Microsoft Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("="*80 + "\n")
    input("Press Enter to exit...")
    sys.exit(1)

try:
    import mediapipe as mp
    import numpy as np
except ImportError as e:
    print("\n" + "="*80)
    print(f" [ERROR] Failed to import core dependencies: {e}")
    print(" Please ensure all dependencies are installed via:")
    print(" pip install -r requirements.txt")
    print("="*80 + "\n")
    input("Press Enter to exit...")
    sys.exit(1)

import math
import os
from collections import deque, Counter

# Resolve script directory to load images reliably from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



mp_face  = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.7, min_tracking_confidence=0.7)
hands_det = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global variables for the interactive Quit button
quit_clicked = False
btn_rect = [0, 0, 0, 0]  # [x1, y1, x2, y2]

def mouse_callback(event, x, y, flags, param):
    global quit_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_rect[0] <= x <= btn_rect[2] and btn_rect[1] <= y <= btn_rect[3]:
            quit_clicked = True

def d(a, b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)

def esc(lm):
    return d(lm[152], lm[10]) + 1e-6

def px(pt, W, H):
    return (int(pt.x * W), int(pt.y * H))

def get_finger_states(lm, left=False):
    tip = [8,12,16,20]
    mid_j = [6,10,14,18]
    out = [1 if (lm[4].x > lm[3].x if left else lm[4].x < lm[3].x) else 0]
    for t, m in zip(tip, mid_j):
        out.append(1 if lm[t].y < lm[m].y else 0)
    return out

class Calibrator:
    N = 45

    def __init__(self):
        self.buf = {k: [] for k in ['le','re','bc','mo','td','le_y','re_y','brow_gap']}
        self.done = False
        self.thr = dict(
            le=0.180, re=0.180, bc_lo=0.185,
            mo=0.055, td=0.145,
            le_y_lo=0.30, re_y_lo=0.30,
            brow_gap_lo=0.10
        )

    def feed(self, lm):
        if self.done:
            return
        e = esc(lm)
        self.buf['le'].append(d(lm[52], lm[159]) / e)
        self.buf['re'].append(d(lm[282], lm[386]) / e)
        self.buf['bc'].append(d(lm[55], lm[285]) / e)
        self.buf['mo'].append(d(lm[13], lm[14]) / e)
        self.buf['td'].append(d(lm[17], lm[152]) / e)
        self.buf['le_y'].append(lm[55].y - lm[9].y)
        self.buf['re_y'].append(lm[285].y - lm[9].y)
        self.buf['brow_gap'].append(abs(lm[55].x - lm[285].x))
        if len(self.buf['le']) >= self.N:
            self._calculate()

    def _calculate(self):
        m  = lambda k: float(np.median(self.buf[k]))
        s  = lambda k: float(np.std(self.buf[k]))
        mg_c = lambda k: max(1.5 * s(k), 0.015)
        mg_b = lambda k, mn: max(3 * s(k), mn)
        self.thr['le']          = m('le')  + mg_c('le')
        self.thr['re']          = m('re')  + mg_c('re')
        self.thr['bc_lo']       = m('bc')  - mg_c('bc')
        self.thr['mo']          = m('mo')  + mg_b('mo', 0.032)
        self.thr['td']          = m('td')  - mg_b('td', 0.018)
        self.thr['le_y_lo']     = m('le_y') + mg_c('le_y')
        self.thr['re_y_lo']     = m('re_y') + mg_c('re_y')
        self.thr['brow_gap_lo'] = m('brow_gap') - mg_c('brow_gap')
        self.done = True

    @property
    def progress(self):
        return min(len(self.buf['le']) / self.N, 1.0)


def detect_tongue(lm, cal):
    e = esc(lm)
    mouth_open  = d(lm[13], lm[14]) / e > cal.thr['mo']
    tongue_down = d(lm[17], lm[152]) / e < cal.thr['td']
    tip_out     = lm[17].y > lm[14].y + 0.012
    return mouth_open and tongue_down and tip_out

def detect_eyebrow(lm, cal):
    e        = esc(lm)
    le       = d(lm[52],  lm[159]) / e
    re       = d(lm[282], lm[386]) / e
    bc       = d(lm[55],  lm[285]) / e
    le_y     = lm[55].y  - lm[9].y
    re_y     = lm[285].y - lm[9].y
    brow_gap = abs(lm[55].x - lm[285].x)
    return (
        le       > cal.thr['le']          or
        re       > cal.thr['re']          or
        bc       < cal.thr['bc_lo']       or
        le_y     > cal.thr['le_y_lo']     or
        re_y     > cal.thr['re_y_lo']     or
        brow_gap < cal.thr['brow_gap_lo']
    )

def detect_bite(hands, face_lm):
    # If both hands are present and covering the mouth/nose region, it should be the 'son' meme, not bite
    if len(hands) == 2:
        nose = face_lm[1]
        hands_x = (hands[0][1][9].x + hands[1][1][9].x) / 2
        hands_y = (hands[0][1][9].y + hands[1][1][9].y) / 2
        if abs(hands_x - nose.x) < 0.15 and abs(hands_y - nose.y) < 0.20:
            return False

    mouth = face_lm[13]
    close_fingers = 0
    for _, lm in hands:
        if d(lm[8], mouth) < 0.09:
            close_fingers += 1
        if d(lm[12], mouth) < 0.09:
            close_fingers += 1
    # Biting/shushing gesture requires exactly one finger close to the mouth
    return close_fingers == 1



def detect_rat(fingers):
    return fingers == [0, 1, 1, 0, 0]

def detect_sonic(hands, face_lm):
    if len(hands) != 2:
        return False
    # Use the bridge of the nose (landmark 168) as reference to require hands to be higher
    ref_y = face_lm[168].y
    return all(lm[9].y < ref_y for _, lm in hands)

def detect_cinema(hands):
    if len(hands) != 2:
        return False
    for fingers, lm in hands:
        if fingers[1:] != [1, 1, 1, 1] or lm[0].y < 0.50:
            return False
    return abs(hands[0][1][0].x - hands[1][1][0].x) >= 0.20

def detect_pray(hands, face_lm):
    if len(hands) != 2:
        return False
    # Check if wrists are close to each other
    wrist_dist = d(hands[0][1][0], hands[1][1][0])
    # Check if fingertips are close to each other
    tip_dist = sum(d(hands[0][1][i], hands[1][1][i]) for i in [8, 12, 16, 20]) / 4
    # Check if index and middle fingers are extended
    extended = all(h[0][1] == 1 and h[0][2] == 1 for h in hands)
    # Check if fingers are pointing upwards (tip is higher than MCP)
    pointing_up = all(h[1][8].y < h[1][5].y for h in hands)
    
    # Check position in front of face
    nose = face_lm[1]
    hands_y = (hands[0][1][9].y + hands[1][1][9].y) / 2
    hands_x = (hands[0][1][9].x + hands[1][1][9].x) / 2
    
    centered = abs(hands_x - nose.x) < 0.15
    in_front = abs(hands_y - nose.y) < 0.20
    
    # Looser thresholds to cover the whole nose/mouth region when both hands are close together
    return wrist_dist < 0.16 and tip_dist < 0.12 and extended and pointing_up and centered and in_front

def is_timeout_pair(h_horiz, h_vert):
    # Use wrist (0) and middle knuckle (9) for palm orientation (very stable even in side profile)
    dx_horiz = abs(h_horiz[1][9].x - h_horiz[1][0].x)
    dy_horiz = abs(h_horiz[1][9].y - h_horiz[1][0].y)
    
    dx_vert = abs(h_vert[1][9].x - h_vert[1][0].x)
    dy_vert = abs(h_vert[1][9].y - h_vert[1][0].y)
    
    # 1. Orientation checks: horizontal hand is wider than tall, vertical hand is taller than wide
    is_horiz = dx_horiz > 0.8 * dy_horiz
    is_vert = dy_vert > 0.8 * dx_vert and h_vert[1][9].y < h_vert[1][0].y
    
    if not (is_horiz and is_vert):
        return False
        
    # 2. Touching check: vertical hand's knuckles/tips must be close to the horizontal hand's palm/wrist/knuckles
    vert_len = d(h_vert[1][0], h_vert[1][9])
    dist = min(
        d(h_vert[1][9], h_horiz[1][9]),
        d(h_vert[1][9], h_horiz[1][5]),
        d(h_vert[1][9], h_horiz[1][0]),
        d(h_vert[1][5], h_horiz[1][9]),
        d(h_vert[1][12], h_horiz[1][9]),
        d(h_vert[1][8], h_horiz[1][9])
    )
    # Using 2.0 * vert_len (knuckle length) as scale-invariant touch threshold
    touching = dist < 2.0 * vert_len
    
    return touching

def detect_timeout(hands):
    if len(hands) != 2:
        return False
    # Try both combinations of which hand is horizontal and which is vertical
    return is_timeout_pair(hands[0], hands[1]) or is_timeout_pair(hands[1], hands[0])




FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,
             397,365,379,378,400,377,152,148,176,149,150,136,
             172,58,132,93,234,127,162,21,54,103,67,109,10]
EYE_L  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33]
EYE_R  = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362]
BROW_L = [70,63,105,66,107,55,65,52,53,46]
BROW_R = [300,293,334,296,336,285,295,282,283,276]
LIPS_OUT = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
LIPS_IN  = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78]
NOSE = [168,6,197,195,5,4,1,19,94,2]

def draw_face_minimal(frame, lm, W, H, cal):
    e        = esc(lm)
    le       = d(lm[52],  lm[159]) / e
    re       = d(lm[282], lm[386]) / e
    bc       = d(lm[55],  lm[285]) / e
    mouth_act = (d(lm[13], lm[14]) / e > cal.thr['mo'] and
                 d(lm[17], lm[152]) / e < cal.thr['td'])
    brow_act = le > cal.thr['le'] or re > cal.thr['re'] or bc < cal.thr['bc_lo']

    COL_BASE = (140, 200, 140)
    COL_ACT  = (80,  240,  80)
    COL_BROW = COL_ACT if brow_act else COL_BASE
    COL_MOUTH = COL_ACT if mouth_act else COL_BASE

    def draw_path(indices, col, close=False):
        pts = [px(lm[i], W, H) for i in indices]
        for j in range(len(pts) - 1):
            cv2.line(frame, pts[j], pts[j+1], col, 1, cv2.LINE_AA)
        if close and len(pts) > 1:
            cv2.line(frame, pts[-1], pts[0], col, 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 1, col, -1, cv2.LINE_AA)

    draw_path(FACE_OVAL, COL_BASE, close=False)
    draw_path(EYE_L,     COL_BASE, close=True)
    draw_path(EYE_R,     COL_BASE, close=True)
    draw_path(BROW_L,    COL_BROW)
    draw_path(BROW_R,    COL_BROW)
    draw_path(NOSE,      COL_BASE)
    draw_path(LIPS_OUT,  COL_MOUTH, close=True)
    draw_path(LIPS_IN,   COL_MOUTH, close=True)


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_hand_minimal(frame, lm, W, H, fingers):
    COL = (140, 200, 140)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px(lm[a], W, H), px(lm[b], W, H), COL, 1, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(frame, px(lm[i], W, H), 2, COL, -1, cv2.LINE_AA)
    for i, tip in enumerate([4, 8, 12, 16, 20]):
        if fingers[i]:
            cv2.circle(frame, px(lm[tip], W, H), 3, (80, 240, 80), -1, cv2.LINE_AA)


def draw_hud(frame, current_img, hands_info, W, H):
    name = current_img if current_img else "neutral"
    col  = (80, 220, 80) if current_img else (160, 160, 160)
    ov   = frame.copy()
    cv2.rectangle(ov, (8, 8), (min(W - 8, 14 + len(name) * 14 + 20), 36), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, name, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    for i, (side, fingers) in enumerate(hands_info):
        cv2.putText(frame, f"{side}: {fingers}", (14, 58 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)
    # Hint to toggle mesh overlay
    cv2.putText(frame, "[M] Toggle Mesh Overlay", (14, H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)



def main():
    global quit_clicked
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("\n" + "="*80)
        print(" [ERROR] Could not open the camera!")
        print(" Please ensure your webcam is plugged in and not in use by another application.")
        print("="*80 + "\n")
        input("Press Enter to exit...")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(5):
        cap.read()

    ret, frame0 = cap.read()
    if not ret:
        print("\n" + "="*80)
        print(" [ERROR] Could not read the initial frame from camera!")
        print(" Please ensure your webcam is working properly.")
        print("="*80 + "\n")
        cap.release()
        input("Press Enter to exit...")
        return

    frame0 = cv2.flip(frame0, 1)
    H, W   = frame0.shape[:2]
    background  = np.full((H, W, 3), 30, dtype=np.uint8)

    cv2.namedWindow("Your Camera",      cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Detected Meme", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Your Camera", mouse_callback)

    cv2.imshow("Your Camera",      frame0)
    cv2.imshow("Detected Meme", background)
    cv2.waitKey(1)

    cal         = Calibrator()
    buf         = deque(maxlen=10)
    current_img = None
    MINVOTOS    = 6
    draw_mesh   = False  # Default to invisible face/hand outlines


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        
        # Set button coordinates relative to current frame width
        global btn_rect
        btn_rect[0], btn_rect[1], btn_rect[2], btn_rect[3] = W - 95, 10, W - 10, 42

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fr    = face_mesh.process(rgb)
        hr    = hands_det.process(rgb)

        det        = None
        face_lm    = None
        hands      = []
        hands_info = []

        if not cal.done:
            pct = cal.progress
            ov  = frame.copy()
            cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cy  = H // 2
            cv2.putText(frame, "Look straight ahead with a neutral face",
                        (W // 2 - 230, cy - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2, cv2.LINE_AA)
            bx1, bx2 = W // 2 - 140, W // 2 + 140
            cv2.rectangle(frame, (bx1, cy + 10), (bx2, cy + 28), (40, 40, 40), -1)
            cv2.rectangle(frame, (bx1, cy + 10),
                          (bx1 + int(280 * pct), cy + 28), (80, 220, 80), -1)
            cv2.rectangle(frame, (bx1, cy + 10), (bx2, cy + 28), (120, 120, 120), 1)
            cv2.putText(frame, f"{int(pct * 100)}%",
                        (W // 2 - 18, cy + 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (160, 160, 160), 1, cv2.LINE_AA)
            
            # Draw Quit Button in calibration mode
            cv2.rectangle(frame, (btn_rect[0], btn_rect[1]), (btn_rect[2], btn_rect[3]), (60, 60, 220), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (btn_rect[0], btn_rect[1]), (btn_rect[2], btn_rect[3]), (80, 80, 240), 1, cv2.LINE_AA)
            cv2.putText(frame, "QUIT", (btn_rect[0] + 20, btn_rect[1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            if fr.multi_face_landmarks:
                cal.feed(fr.multi_face_landmarks[0].landmark)
            
            cv2.imshow("Your Camera",      frame)
            cv2.imshow("Detected Meme", background)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q') or quit_clicked:
                break
            continue

        if fr.multi_face_landmarks:
            face_lm = fr.multi_face_landmarks[0].landmark
            if draw_mesh:
                draw_face_minimal(frame, face_lm, W, H, cal)

        if hr.multi_hand_landmarks:
            for i, hl in enumerate(hr.multi_hand_landmarks):
                lm   = hl.landmark
                left = hr.multi_handedness[i].classification[0].label == "Left"
                fingers = get_finger_states(lm, left)
                if draw_mesh:
                    draw_hand_minimal(frame, lm, W, H, fingers)
                hands.append((fingers, lm))
                hands_info.append(("L" if left else "R", fingers))

        # Check if any hands are near the face to prevent face mesh distortion from mistriggering dog/cat memes
        hands_near_face = face_lm and any(d(h[1][9], face_lm[1]) < 0.25 for h in hands)

        if len(hands) == 2 and detect_timeout(hands):
            det = "Timeout.png"
        elif face_lm and len(hands) == 2 and detect_pray(hands, face_lm):
            det = "son.png"
        elif face_lm and len(hands) == 2 and detect_sonic(hands, face_lm):
            det = "Sonic.jpeg"
        elif len(hands) == 2 and detect_cinema(hands):
            det = "cinema.jpg"
        elif face_lm and hands and detect_bite(hands, face_lm):
            det = "bite.png"
        elif face_lm and detect_tongue(face_lm, cal) and not hands_near_face:
            det = "cat.png"
        elif face_lm and detect_eyebrow(face_lm, cal) and not hands_near_face:
            det = "dog.jpeg"
        elif len(hands) == 1:
            fingers_m, lm_m = hands[0]
            if detect_rat(fingers_m):
                det = "rat.jpeg"

        buf.append(det)
        counts     = Counter(buf)
        top, votes = counts.most_common(1)[0]
        if votes >= MINVOTOS:
            current_img = top

        draw_hud(frame, current_img, hands_info, W, H)

        # Draw Quit Button in detection mode
        cv2.rectangle(frame, (btn_rect[0], btn_rect[1]), (btn_rect[2], btn_rect[3]), (60, 60, 220), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (btn_rect[0], btn_rect[1]), (btn_rect[2], btn_rect[3]), (80, 80, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "QUIT", (btn_rect[0] + 20, btn_rect[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Your Camera", frame)

        if current_img:
            meme_path = os.path.join(SCRIPT_DIR, current_img)
            meme = cv2.imread(meme_path)
            if meme is not None and meme.size > 0:
                cv2.imshow("Detected Meme", cv2.resize(meme, (W, H)))
            else:
                err = background.copy()
                cv2.putText(err, f"Missing: {current_img}", (20, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 220), 2)
                cv2.imshow("Detected Meme", err)
        else:
            cv2.imshow("Detected Meme", background)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q') or quit_clicked:
            break
        elif key == ord('m') or key == ord('M'):
            draw_mesh = not draw_mesh

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()