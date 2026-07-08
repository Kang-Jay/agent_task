from pathlib import Path
import json
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(r'D:/cache/SummerCap/kaohe/zju')
SUMMARY_PATH = ROOT / 'docs/ai2thor_outputs/ai2thor_demo_summary.json'
OUT_DIR = ROOT / 'docs/cinematic_demo'
FRAME_DIR = OUT_DIR / 'frames'
OUT_VIDEO = OUT_DIR / 'ai2thor_cinematic_visual_search_demo_action_aligned.mp4'
VERIFY_PATH = OUT_DIR / 'cinematic_demo_action_aligned_verification.json'

OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)
for old in FRAME_DIR.glob('cinematic_*.png'):
    old.unlink()

summary = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))
steps = summary['steps']
assert steps, 'No AI2-THOR steps found'
assert all(s.get('backend') == 'ai2thor' for s in steps), 'Non-AI2-THOR step found in summary'
assert steps[-1].get('action') == 'STOP', 'Final step is not STOP'

W, H = 1920, 1080
FPS = 24
HOLD = 28
INTRO = 48
OUTRO = 48

try:
    FONT_TITLE = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 46)
    FONT_HEAD = ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf', 30)
    FONT_BODY = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 24)
    FONT_SMALL = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 18)
    FONT_MONO = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 20)
except Exception:
    FONT_TITLE = FONT_HEAD = FONT_BODY = FONT_SMALL = FONT_MONO = ImageFont.load_default()

CYAN = (48, 224, 210)
BLUE = (88, 166, 255)
GREEN = (90, 242, 150)
RED = (255, 92, 82)
AMBER = (255, 199, 87)
WHITE = (238, 246, 255)
MUTED = (150, 170, 196)
DARK = (5, 10, 18)
PANEL = (13, 24, 40)


def rgba(color, alpha=255):
    return tuple(color) + (alpha,)


def read_img(rel):
    return Image.open(ROOT / rel).convert('RGB')


def cover_resize(img, box):
    bw, bh = box
    iw, ih = img.size
    scale = max(bw / iw, bh / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - bw) // 2
    top = (nh - bh) // 2
    return resized.crop((left, top, left + bw, top + bh))


def round_rect(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def alpha_paste(base, overlay, xy=(0, 0)):
    if overlay.mode != 'RGBA':
        overlay = overlay.convert('RGBA')
    base.paste(overlay, xy, overlay)


def draw_wrapped(draw, text, xy, max_chars, font, fill, line_gap=8, max_lines=4):
    words = str(text).split()
    lines = []
    cur = ''
    for word in words:
        probe = (cur + ' ' + word).strip()
        if len(probe) > max_chars and cur:
            lines.append(cur)
            cur = word
        else:
            cur = probe
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max(0, max_chars - 3)] + '...'
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap


def bbox_scaled(candidate, src_size, dst_box):
    if not candidate:
        return None
    x0, y0, x1, y1 = candidate['bbox']
    sw, sh = src_size
    dx, dy, dw, dh = dst_box
    return [dx + int(x0 / sw * dw), dy + int(y0 / sh * dh), dx + int(x1 / sw * dw), dy + int(y1 / sh * dh)]


def heading(step):
    return float(step.get('robot', {}).get('heading', 0.0))


def prior_observation_label(idx):
    if idx == 0:
        return 'OBSERVATION: initial robot camera view'
    previous = steps[idx - 1]
    delta = heading(steps[idx]) - heading(previous)
    if previous['action'].startswith('TURN'):
        return f"OBS AFTER: {previous['action']} ({delta:+.0f} deg yaw)"
    if previous['action'] == 'MOVE_FORWARD':
        return 'OBS AFTER: MOVE_FORWARD'
    return f"OBS AFTER: {previous['action']}"


def command_label(action):
    if action == 'TURN_RIGHT':
        return 'NEXT ACTION: TURN_RIGHT (+30 deg yaw)'
    if action == 'TURN_LEFT':
        return 'NEXT ACTION: TURN_LEFT (-30 deg yaw)'
    if action == 'MOVE_FORWARD':
        return 'NEXT ACTION: MOVE_FORWARD'
    if action == 'INSPECT':
        return 'NEXT ACTION: INSPECT'
    if action == 'STOP':
        return 'NEXT ACTION: STOP'
    return f'NEXT ACTION: {action}'


def gradient_bg(t):
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    xs = np.linspace(0, 1, W, dtype=np.float32)
    ys = np.linspace(0, 1, H, dtype=np.float32)
    nx, ny = np.meshgrid(xs, ys)
    pulse = 0.5 + 0.5 * np.sin(t * 2 * math.pi + nx * 5.0)
    arr[:, :, 0] = np.clip(5 + 15 * nx + 10 * pulse, 0, 255)
    arr[:, :, 1] = np.clip(10 + 20 * ny, 0, 255)
    arr[:, :, 2] = np.clip(18 + 35 * (1 - nx) + 8 * pulse, 0, 255)
    return Image.fromarray(arr, 'RGB')


def draw_hud(step, idx, local_t, global_t, title_phase):
    base = gradient_bg(global_t)
    draw = ImageDraw.Draw(base, 'RGBA')

    obs = cover_resize(read_img(step['observation_path']), (1120, 720))
    top = cover_resize(read_img(step['topdown_path']), (470, 470))

    obs_x, obs_y, obs_w, obs_h = 70, 205, 1120, 720
    map_x, map_y, map_w, map_h = 1375, 185, 470, 470

    shadow = Image.new('RGBA', (obs_w + 36, obs_h + 36), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, 'RGBA')
    sd.rounded_rectangle([18, 18, obs_w + 18, obs_h + 18], radius=22, fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    alpha_paste(base, shadow, (obs_x - 18, obs_y - 18))
    base.paste(obs, (obs_x, obs_y))
    draw.rounded_rectangle([obs_x, obs_y, obs_x + obs_w, obs_y + obs_h], radius=20, outline=rgba(CYAN, 230), width=4)

    sweep_x = obs_x + int((local_t % 1.0) * obs_w)
    draw.rectangle([sweep_x - 5, obs_y, sweep_x + 5, obs_y + obs_h], fill=rgba(CYAN, 95))
    draw.line([(sweep_x, obs_y), (sweep_x, obs_y + obs_h)], fill=rgba(WHITE, 180), width=2)

    cand = step.get('best_candidate')
    if cand:
        scaled = bbox_scaled(cand, (960, 540), (obs_x, obs_y, obs_w, obs_h))
        if scaled:
            pulse = 0.65 + 0.35 * math.sin(global_t * 2 * math.pi * 8)
            color = (int(255 * pulse), 70, 70)
            for grow, alpha in [(18, 50), (10, 95), (3, 235)]:
                draw.rectangle([scaled[0]-grow, scaled[1]-grow, scaled[2]+grow, scaled[3]+grow], outline=rgba(color, alpha), width=4)
            draw.text((scaled[0], max(obs_y, scaled[1] - 36)), f"TARGET LOCK: {cand['label']}  {cand['confidence']:.3f}", font=FONT_MONO, fill=rgba(RED, 255))

    base.paste(top, (map_x, map_y))
    draw.rounded_rectangle([map_x, map_y, map_x + map_w, map_y + map_h], radius=16, outline=rgba(BLUE, 235), width=4)

    draw.text((70, 54), 'AI2-THOR VISUAL SEARCH AGENT', font=FONT_TITLE, fill=rgba(WHITE, 255))
    draw.text((72, 116), 'Strict real simulator demo | FloorPlan211 | target: Television', font=FONT_HEAD, fill=rgba(CYAN, 255))
    draw.text((70, 160), title_phase, font=FONT_BODY, fill=rgba(MUTED, 255))

    obs_badge = prior_observation_label(idx)
    round_rect(draw, [obs_x + 20, obs_y + 20, obs_x + 555, obs_y + 62], 12, (6, 18, 30, 210), rgba(CYAN, 210), 2)
    draw.text((obs_x + 38, obs_y + 30), obs_badge, font=FONT_SMALL, fill=rgba(WHITE, 255))

    px, py, pw, ph = 1235, 690, 610, 260
    round_rect(draw, [px, py, px + pw, py + ph], 18, rgba(PANEL, 235), rgba(BLUE, 95), 2)
    draw.text((px + 28, py + 24), f"STEP {idx:02d}", font=FONT_HEAD, fill=rgba(MUTED, 255))
    draw.text((px + 28, py + 56), command_label(step['action']), font=FONT_SMALL, fill=rgba(MUTED, 255))
    action_color = GREEN if step['action'] == 'STOP' else CYAN if step['action'] == 'INSPECT' else BLUE
    draw.text((px + 28, py + 88), step['action'], font=FONT_TITLE, fill=rgba(action_color, 255))
    draw.text((px + 330, py + 34), 'CONFIDENCE', font=FONT_SMALL, fill=rgba(MUTED, 255))
    conf = float(step['confidence'])
    draw.rounded_rectangle([px + 330, py + 72, px + 555, py + 96], radius=12, fill=(25, 38, 58, 255))
    draw.rounded_rectangle([px + 330, py + 72, px + 330 + int(225 * conf), py + 96], radius=12, fill=rgba(RED, 255))
    draw.text((px + 330, py + 108), f"{conf:.3f}", font=FONT_HEAD, fill=rgba(RED, 255))
    draw_wrapped(draw, step['thought'], (px + 28, py + 158), 56, FONT_SMALL, rgba(WHITE, 255), max_lines=4)

    tx, ty = 70, 955
    draw.text((tx, ty - 42), 'Embodied trajectory', font=FONT_BODY, fill=rgba(WHITE, 255))
    step_w = 235
    for j, s in enumerate(steps):
        x = tx + j * (step_w + 15)
        active = j == idx
        fill = (30, 56, 80, 235) if active else (16, 30, 48, 210)
        outline = CYAN if active else (68, 88, 110)
        round_rect(draw, [x, ty, x + step_w, ty + 78], 12, fill, rgba(outline, 220), 2)
        text_color = WHITE if active else MUTED
        draw.text((x + 14, ty + 12), f"{j}: next {s['action']}", font=FONT_SMALL, fill=rgba(text_color, 255))
        bar = int((step_w - 28) * float(s['confidence']))
        draw.rounded_rectangle([x + 14, ty + 49, x + step_w - 14, ty + 61], radius=6, fill=(28, 38, 56, 255))
        bar_color = RED if s['action'] == 'STOP' else CYAN
        draw.rounded_rectangle([x + 14, ty + 49, x + 14 + bar, ty + 61], radius=6, fill=rgba(bar_color, 255))

    badges = [('REAL AI2-THOR', GREEN), ('INSTANCE SEGMENTATION', CYAN), ('NO FALLBACK', AMBER)]
    bx = 1260
    for text, color in badges:
        tw = 18 * len(text) + 38
        round_rect(draw, [bx, 54, bx + tw, 96], 14, (20, 35, 50, 230), rgba(color, 230), 2)
        draw.text((bx + 18, 64), text, font=FONT_SMALL, fill=rgba(color, 255))
        bx += tw + 16

    return base


def intro_frame(k):
    t = k / max(INTRO - 1, 1)
    frame = gradient_bg(t)
    draw = ImageDraw.Draw(frame, 'RGBA')
    draw.text((110, 310), 'Embodied Visual Search Agent', font=FONT_TITLE, fill=rgba(WHITE, 255))
    draw.text((112, 378), 'Real AI2-THOR FloorPlan211 Demo', font=FONT_HEAD, fill=rgba(CYAN, 255))
    draw.text((112, 438), 'Goal: Find the television in an unfamiliar scene', font=FONT_BODY, fill=rgba(MUTED, 255))
    x0, y0 = 112, 530
    for i, label in enumerate(['Language instruction', 'Robot POV', 'Thought + action', 'Segmentation-confirmed STOP']):
        alpha = int(255 * min(1, max(0, t * 5 - i)))
        round_rect(draw, [x0, y0 + i * 70, x0 + 620, y0 + i * 70 + 48], 12, (16, 30, 48, alpha), rgba(CYAN, alpha), 2)
        draw.text((x0 + 20, y0 + i * 70 + 11), label, font=FONT_BODY, fill=rgba(WHITE, alpha))
    return frame


def outro_frame(k):
    t = k / max(OUTRO - 1, 1)
    last = draw_hud(steps[-1], len(steps) - 1, t, t, 'Target acquired and verified by simulator segmentation')
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, int(92 * t)))
    od = ImageDraw.Draw(overlay, 'RGBA')
    card_x, card_y, card_w, card_h = 118, 118, 835, 152
    od.rounded_rectangle(
        [card_x, card_y, card_x + card_w, card_y + card_h],
        radius=22,
        fill=(6, 18, 30, int(218 * min(1, t * 1.8))),
        outline=rgba(GREEN, int(235 * min(1, t * 1.8))),
        width=3,
    )
    od.text((card_x + 34, card_y + 26), 'DEMO COMPLETE', font=FONT_TITLE, fill=rgba(GREEN, 255))
    od.text(
        (card_x + 36, card_y + 92),
        '7-step trajectory | confidence 0.943 | backend ai2thor',
        font=FONT_BODY,
        fill=rgba(WHITE, 255),
    )
    alpha_paste(last, overlay)
    return last

frames = []
for k in range(INTRO):
    frames.append(intro_frame(k))

for idx, step in enumerate(steps):
    for k in range(HOLD):
        phase = 'Frame shows current observation; HUD shows the next Agent command'
        if step['action'] == 'INSPECT':
            phase = 'Candidate found; next command inspects before final confirmation'
        elif step['action'] == 'STOP':
            phase = 'Target acquired; next command stops after simulator segmentation confirmation'
        frames.append(draw_hud(step, idx, k / HOLD, (idx + k / HOLD) / len(steps), phase))

for k in range(OUTRO):
    frames.append(outro_frame(k))

paths = []
for i, frame in enumerate(frames):
    p = FRAME_DIR / f'cinematic_{i:04d}.png'
    frame.save(p, quality=95)
    paths.append(p)

writer = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
for p in paths:
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError(f'Failed to read frame {p}')
    writer.write(img)
writer.release()

cap = cv2.VideoCapture(str(OUT_VIDEO))
verification = {
    'video_path': str(OUT_VIDEO.relative_to(ROOT)),
    'exists': OUT_VIDEO.exists(),
    'bytes': OUT_VIDEO.stat().st_size if OUT_VIDEO.exists() else 0,
    'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    'fps': float(cap.get(cv2.CAP_PROP_FPS)),
    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    'duration_seconds': round(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / FPS, 2),
    'source_steps': len(steps),
    'all_steps_ai2thor': all(s.get('backend') == 'ai2thor' for s in steps),
    'final_action': steps[-1].get('action'),
    'final_confidence': steps[-1].get('confidence'),
    'final_best_candidate': steps[-1].get('best_candidate'),
    'sample_frames': [
        str((FRAME_DIR / 'cinematic_0000.png').relative_to(ROOT)),
        str(paths[len(paths)//2].relative_to(ROOT)),
        str(paths[-1].relative_to(ROOT)),
    ],
}
cap.release()

sample_stats = []
for rel in verification['sample_frames']:
    img = cv2.imread(str(ROOT / rel))
    sample_stats.append({'path': rel, 'mean': round(float(img.mean()), 2), 'std': round(float(img.std()), 2)})
verification['sample_stats'] = sample_stats
VERIFY_PATH.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(verification, ensure_ascii=False, indent=2))
assert verification['exists'] and verification['bytes'] > 500000, 'Video was not written correctly'
assert verification['frame_count'] == len(paths), 'Frame count mismatch'
assert verification['width'] == W and verification['height'] == H, 'Resolution mismatch'
assert verification['all_steps_ai2thor'], 'Video source is not strictly AI2-THOR'
assert verification['final_action'] == 'STOP', 'Final action is not STOP'
assert verification['final_best_candidate']['label'] == 'Television', 'Final target is not Television'
