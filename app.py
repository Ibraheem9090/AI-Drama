import asyncio
import json
import os
import re
import textwrap
import uuid
from pathlib import Path

import edge_tts
import numpy as np
import requests
import streamlit as st
from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# App, Model & Storage Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DramaPulse AI — Short Animated Stories",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent / "generated_dramas"
BASE_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_FILE = BASE_DIR / "catalog.json"

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_STORY_MODEL = "mistralai/mistral-large-4-1t-a52b"


def get_nvidia_api_key() -> str:
    try:
        key = str(st.secrets.get("NVIDIA_API_KEY", "")).strip()
        if key:
            return key
    except Exception:
        pass
    return os.getenv("NVIDIA_API_KEY", "").strip()


# ---------------------------------------------------------------------------
# Styling (Dark Mobile-First Drama Feed)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b0b0f;
            color: #f5f5f7;
        }
        .top-search-banner {
            background: #1a1a22;
            border: 1px solid #2c2c38;
            border-radius: 999px;
            padding: 8px 16px;
            margin-bottom: 10px;
            color: #d6d6e0;
            font-size: 0.92rem;
        }
        .drama-card {
            background: #14141c;
            border: 1px solid #23232f;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 14px;
            transition: transform 0.15s ease;
        }
        .drama-card:hover {
            border-color: #ff2e63;
        }
        .badge-hot {
            display: inline-block;
            background: #ff2e2e;
            color: #ffffff;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 4px;
            margin-bottom: 4px;
        }
        .badge-new {
            display: inline-block;
            background: #16a34a;
            color: #ffffff;
            font-size: 0.7rem;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 4px;
            margin-bottom: 4px;
        }
        .drama-title {
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.25;
            color: #ffffff;
            margin-top: 6px;
        }
        .drama-tag {
            display: inline-block;
            background: #23202c;
            color: #ff85a1;
            font-size: 0.72rem;
            padding: 2px 8px;
            border-radius: 6px;
            margin-top: 5px;
        }
        .subtitle-box {
            background: rgba(15, 15, 22, 0.9);
            border-left: 4px solid #ff2e63;
            padding: 12px 16px;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Font & Catalog Helpers
# ---------------------------------------------------------------------------
def get_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_catalog() -> list[dict]:
    if CATALOG_FILE.exists():
        try:
            return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_catalog(items: list[dict]) -> None:
    CATALOG_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Agentic Story & Scene Planner (Mistral Large 4 on NVIDIA NIM)
# ---------------------------------------------------------------------------
def fallback_story_plan(prompt: str, category: str) -> dict:
    clean = prompt.strip().rstrip(".")
    title = clean[:52] if len(clean) > 4 else "The Silent Comeback"
    if len(clean) > 52:
        title = title.rsplit(" ", 1)[0] + "..."

    return {
        "title": title,
        "category": category,
        "badge": "HOT",
        "views": "148.2K",
        "character_name": "Aria",
        "character_style": "Expressive modern anime drama character, dramatic lighting",
        "scenes": [
            {
                "scene_number": 1,
                "setting": "Rain-streaked glass penthouse overlooking city lights at night",
                "mood": "Betrayal & resolve",
                "motion": "zoom_in",
                "duration": 4.0,
                "subtitle_en": f"They thought I broke down over '{clean}'. They didn't know I was just getting started.",
                "narration": f"They thought I broke down over {clean}. They didn't know I was just getting started.",
            },
            {
                "scene_number": 2,
                "setting": "Minimalist creative studio with warm amber desk lamp and blueprints",
                "mood": "Focused ambition",
                "motion": "pan_right",
                "duration": 4.0,
                "subtitle_en": "Every late night turned doubt into proof. By dawn, the whole room was watching.",
                "narration": "Every late night turned doubt into proof. By dawn, the whole room was watching.",
            },
            {
                "scene_number": 3,
                "setting": "Golden hour rooftop terrace with sweeping skyline view",
                "mood": "Triumphant calm",
                "motion": "zoom_out",
                "duration": 4.0,
                "subtitle_en": "No drama, no revenge speeches—just results they can't ignore.",
                "narration": "No drama, no revenge speeches—just results they can't ignore.",
            },
        ],
    }


def plan_story_with_ai(prompt: str, category: str) -> dict:
    key = get_nvidia_api_key()
    if not key:
        st.info("No `NVIDIA_API_KEY` found in `st.secrets`. Using built-in story template.")
        return fallback_story_plan(prompt, category)

    system_msg = (
        "You are a vertical short-drama script & animation director agent. "
        "Return ONLY valid JSON with keys: title, category, badge ('HOT' or 'NEW'), "
        "views (e.g. '215.4K'), character_name, character_style, and scenes (list of 3 dicts). "
        "Each scene dict must have: scene_number (int), setting (str), mood (str), "
        "motion ('zoom_in', 'zoom_out', or 'pan_right'), duration (float 3.5-5.0), "
        "subtitle_en (concise English subtitle, max 18 words), narration (str)."
    )
    user_msg = f"Create a gripping short drama from this prompt: {prompt}\nCategory: {category}"

    try:
        resp = requests.post(
            NVIDIA_CHAT_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": NVIDIA_STORY_MODEL,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.7,
                "max_tokens": 1200,
            },
            timeout=35,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if "scenes" in data and len(data["scenes"]) >= 1:
                data["category"] = category
                return data
    except Exception as exc:
        st.warning(f"Model `{NVIDIA_STORY_MODEL}` fallback triggered ({exc}). Using local story template.")

    return fallback_story_plan(prompt, category)


# ---------------------------------------------------------------------------
# Character & Scene Visual Renderer
# ---------------------------------------------------------------------------
def draw_stylized_character_frame(
    width: int,
    height: int,
    scene_num: int,
    mood: str,
    title: str,
    character_name: str,
) -> Image.Image:
    palettes = [
        ((24, 14, 46), (120, 28, 82), (255, 166, 77)),
        ((14, 28, 48), (32, 92, 120), (255, 198, 109)),
        ((38, 18, 28), (140, 42, 68), (255, 214, 153)),
    ]
    c_top, c_mid, c_accent = palettes[(scene_num - 1) % len(palettes)]

    img = Image.new("RGB", (width, height), c_top)
    draw = ImageDraw.Draw(img)

    for y in range(height):
        t = y / max(1, height - 1)
        if t < 0.65:
            k = t / 0.65
            r = int(c_top[0] * (1 - k) + c_mid[0] * k)
            g = int(c_top[1] * (1 - k) + c_mid[1] * k)
            b = int(c_top[2] * (1 - k) + c_mid[2] * k)
        else:
            k = (t - 0.65) / 0.35
            r = int(c_mid[0] * (1 - k) + 18 * k)
            g = int(c_mid[1] * (1 - k) + 18 * k)
            b = int(c_mid[2] * (1 - k) + 26 * k)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    cx, cy = width // 2, int(height * 0.44)
    for rad in range(260, 40, -25):
        alpha_factor = 1.0 - (rad / 280.0)
        glow_col = (
            min(255, int(c_accent[0] * 0.45 * alpha_factor + c_mid[0] * 0.5)),
            min(255, int(c_accent[1] * 0.45 * alpha_factor + c_mid[1] * 0.5)),
            min(255, int(c_accent[2] * 0.45 * alpha_factor + c_mid[2] * 0.5)),
        )
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=glow_col)

    bob_y = (scene_num % 2) * 8 - 4
    head_y = int(height * 0.36) + bob_y

    draw.rounded_rectangle(
        [cx - 145, head_y + 78, cx + 145, head_y + 295],
        radius=70,
        fill=(22, 22, 34),
        outline=c_accent,
        width=3,
    )
    draw.polygon(
        [(cx - 55, head_y + 82), (cx + 55, head_y + 82), (cx, head_y + 165)],
        fill=c_accent,
    )

    draw.ellipse([cx - 92, head_y - 95, cx + 92, head_y + 105], fill=(26, 22, 36))
    draw.ellipse(
        [cx - 68, head_y - 72, cx + 68, head_y + 76],
        fill=(246, 206, 178),
        outline=(190, 145, 120),
        width=2,
    )
    draw.pieslice(
        [cx - 72, head_y - 84, cx + 72, head_y + 10],
        start=185,
        end=355,
        fill=(26, 22, 36),
    )

    for ex in (cx - 28, cx + 28):
        draw.ellipse([ex - 14, head_y - 10, ex + 14, head_y + 12], fill=(255, 255, 255))
        draw.ellipse([ex - 7, head_y - 7, ex + 7, head_y + 9], fill=(58, 34, 92))
        draw.ellipse([ex - 3, head_y - 5, ex + 2, head_y], fill=(255, 255, 255))

    if scene_num == 1:
        draw.arc(
            [cx - 14, head_y + 28, cx + 14, head_y + 44],
            start=190,
            end=350,
            fill=(165, 75, 85),
            width=3,
        )
    elif scene_num == 2:
        draw.line(
            [(cx - 12, head_y + 36), (cx + 12, head_y + 34)],
            fill=(165, 75, 85),
            width=3,
        )
    else:
        draw.arc(
            [cx - 18, head_y + 24, cx + 18, head_y + 44],
            start=15,
            end=165,
            fill=(165, 75, 85),
            width=3,
        )

    font_sm = get_font(22, bold=True)
    font_md = get_font(26, bold=True)
    draw.rounded_rectangle([32, 34, width - 32, 86], radius=16, fill=(12, 12, 18))
    draw.text(
        (52, 47),
        f"{title[:28]}  •  EP {scene_num} ({mood})",
        fill=c_accent,
        font=font_sm,
    )
    draw.text(
        (36, head_y + 315),
        f"Character: {character_name}",
        fill=(235, 235, 245),
        font=font_md,
    )
    return img


def generate_scene_image(
    scene: dict, drama_meta: dict, out_path: Path, width: int = 540, height: int = 960
) -> Path:
    img = draw_stylized_character_frame(
        width=width,
        height=height,
        scene_num=int(scene.get("scene_number", 1)),
        mood=str(scene.get("mood", "Drama")),
        title=str(drama_meta.get("title", "Short Drama")),
        character_name=str(drama_meta.get("character_name", "Aria")),
    )
    img.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Voiceover & Burned-In English Subtitle Video Builder
# ---------------------------------------------------------------------------
async def _synthesize_tts(text: str, mp3_path: Path) -> bool:
    try:
        communicate = edge_tts.Communicate(text=text, voice="en-US-AriaNeural")
        await communicate.save(str(mp3_path))
        return mp3_path.exists() and mp3_path.stat().st_size > 0
    except Exception:
        return False


def synthesize_narration(text: str, mp3_path: Path) -> bool:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ok = loop.run_until_complete(_synthesize_tts(text, mp3_path))
        loop.close()
        return ok
    except Exception:
        return False


def render_animated_frame_with_subtitle(
    base_img: Image.Image,
    subtitle_text: str,
    progress: float,
    motion: str,
) -> np.ndarray:
    w, h = base_img.size

    scale = 1.0 + 0.08 * progress
    if motion == "zoom_out":
        scale = 1.08 - 0.08 * progress

    nw, nh = int(w * scale), int(h * scale)
    zoomed = base_img.resize((nw, nh), Image.Resampling.BILINEAR)

    if motion == "pan_right":
        max_dx = max(0, nw - w)
        left = int(max_dx * progress)
        top = max(0, (nh - h) // 2)
    else:
        left = max(0, (nw - w) // 2)
        top = max(0, (nh - h) // 2)

    frame = zoomed.crop((left, top, left + w, top + h)).copy()

    sub_font = get_font(24, bold=True)
    lines = textwrap.wrap(subtitle_text, width=34)[:3]
    line_h = 32
    box_h = len(lines) * line_h + 24
    box_top = h - box_h - 68
    box_bottom = h - 68

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle(
        [24, box_top, w - 24, box_bottom],
        radius=14,
        fill=(10, 10, 15, 205),
    )
    frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(frame)

    y_text = box_top + 12
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=sub_font)
        tw = bbox[2] - bbox[0]
        x_text = (w - tw) // 2
        for ox, oy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((x_text + ox, y_text + oy), line, fill=(0, 0, 0), font=sub_font)
        draw.text((x_text, y_text), line, fill=(255, 255, 255), font=sub_font)
        y_text += line_h

    return np.array(frame)


def build_drama_episode(prompt: str, category: str) -> dict:
    drama_id = uuid.uuid4().hex[:8]
    drama_dir = BASE_DIR / drama_id
    drama_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_story_with_ai(prompt, category)
    clips = []
    audio_clips_to_close = []
    subtitles_log = []

    fps = 15
    for idx, scene in enumerate(plan["scenes"], start=1):
        img_path = drama_dir / f"scene_{idx}.png"
        generate_scene_image(scene, plan, img_path, width=540, height=960)
        base_img = Image.open(img_path).convert("RGB")

        duration = float(scene.get("duration", 4.0))
        motion = str(scene.get("motion", "zoom_in"))
        subtitle_en = str(scene.get("subtitle_en", ""))
        narration = str(scene.get("narration", subtitle_en))

        mp3_path = drama_dir / f"scene_{idx}.mp3"
        has_voice = synthesize_narration(narration, mp3_path)

        audio_clip = None
        if has_voice:
            try:
                audio_clip = AudioFileClip(str(mp3_path))
                duration = max(duration, float(audio_clip.duration) + 0.4)
                audio_clips_to_close.append(audio_clip)
            except Exception:
                audio_clip = None

        def make_frame(t, img=base_img, sub=subtitle_en, mot=motion, dur=duration):
            prog = min(1.0, max(0.0, t / max(0.1, dur)))
            return render_animated_frame_with_subtitle(img, sub, prog, mot)

        vclip = ImageClip(make_frame, duration=duration).with_fps(fps)
        if audio_clip is not None:
            vclip = vclip.with_audio(audio_clip)
        clips.append(vclip)

        subtitles_log.append(
            {
                "scene": idx,
                "subtitle_en": subtitle_en,
                "duration_sec": round(duration, 1),
            }
        )

    poster_path = drama_dir / "poster.png"
    first_scene_img = Image.open(drama_dir / "scene_1.png").convert("RGB")
    first_scene_img.save(poster_path)

    video_path = drama_dir / "episode.mp4"
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(
        str(video_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    for c in clips:
        c.close()
    for ac in audio_clips_to_close:
        ac.close()
    final_clip.close()

    item = {
        "id": drama_id,
        "title": plan.get("title", "Untitled Drama"),
        "category": plan.get("category", category),
        "badge": plan.get("badge", "NEW"),
        "views": plan.get("views", "98.4K"),
        "character_name": plan.get("character_name", "Aria"),
        "prompt": prompt,
        "poster_path": str(poster_path),
        "video_path": str(video_path),
        "subtitles": subtitles_log,
    }

    catalog = load_catalog()
    catalog.insert(0, item)
    save_catalog(catalog)
    return item


# ---------------------------------------------------------------------------
# Seed Default Dramas on First Run
# ---------------------------------------------------------------------------
catalog = load_catalog()
if not catalog:
    with st.spinner("Preparing starter animated dramas..."):
        build_drama_episode(
            "He gave her my room and doubted my talent, so I launched my own studio",
            "Career Growth",
        )
        build_drama_episode(
            "They laughed when I started from zero, until my midnight project went global",
            "Popular",
        )
        catalog = load_catalog()


# ---------------------------------------------------------------------------
# UI: Search Bar, Generator Form, Category Tabs & Grid
# ---------------------------------------------------------------------------
if "selected_drama_id" not in st.session_state:
    st.session_state.selected_drama_id = None

st.markdown(
    f'<div class="top-search-banner">🔍 &nbsp;He Gave Her My Room. I Took Back My Life. &nbsp;|&nbsp; Model: <code>{NVIDIA_STORY_MODEL}</code></div>',
    unsafe_allow_html=True,
)

with st.expander("✨ Create New Animated Story from Prompt", expanded=False):
    with st.form("create_drama_form"):
        story_prompt = st.text_area(
            "Enter your story prompt",
            placeholder="e.g., A quiet engineer uncovers a hidden truth and turns the tables in 30 days...",
            height=80,
        )
        story_cat = st.selectbox(
            "Category",
            ["Popular", "Career Growth", "Marriage Crisis", "Family Ethics", "New"],
        )
        submitted = st.form_submit_button("🎬 Generate Animated Drama")
        if submitted and story_prompt.strip():
            with st.status(f"Generating story with {NVIDIA_STORY_MODEL} & rendering English subtitles...", expanded=True):
                created = build_drama_episode(story_prompt.strip(), story_cat)
                st.session_state.selected_drama_id = created["id"]
            st.rerun()

categories = [
    "Popular",
    "Rankings",
    "New",
    "Categories",
    "Career Growth",
    "Marriage Crisis",
    "Family Ethics",
]
active_cat = st.radio("Browse Feed", categories, horizontal=True, label_visibility="collapsed")
search_query = st.text_input(
    "Filter stories", placeholder="Search title or trope...", label_visibility="collapsed"
)

# ---------------------------------------------------------------------------
# Episode Player View
# ---------------------------------------------------------------------------
selected_item = next(
    (d for d in catalog if d["id"] == st.session_state.selected_drama_id), None
)

if selected_item:
    st.markdown("---")
    col_back, col_title = st.columns([1, 5])
    with col_back:
        if st.button("⬅ Back to Grid"):
            st.session_state.selected_drama_id = None
            st.rerun()
    with col_title:
        st.subheader(f"🎭 {selected_item['title']}")
        st.caption(
            f"Lead Character: **{selected_item['character_name']}** • "
            f"Category: **{selected_item['category']}** • Subtitles: **English**"
        )

    player_col, script_col = st.columns([1, 1.3])
    with player_col:
        if os.path.exists(selected_item["video_path"]):
            st.video(selected_item["video_path"], autoplay=True)
    with script_col:
        st.markdown("### 📝 English Subtitles & Scene Breakdown")
        for s in selected_item.get("subtitles", []):
            st.markdown(
                f'<div class="subtitle-box"><strong>Scene {s["scene"]} ({s["duration_sec"]}s):</strong><br/>{s["subtitle_en"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("---")

# ---------------------------------------------------------------------------
# Vertical Poster Grid
# ---------------------------------------------------------------------------
filtered = catalog
if active_cat not in ("Popular", "Rankings", "Categories"):
    filtered = [d for d in filtered if d.get("category") == active_cat]
if search_query.strip():
    q = search_query.strip().lower()
    filtered = [
        d
        for d in filtered
        if q in d.get("title", "").lower() or q in d.get("category", "").lower()
    ]

st.markdown("### Recommend")
cols_per_row = 3
for row_start in range(0, len(filtered), cols_per_row):
    row_items = filtered[row_start : row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for idx, item in enumerate(row_items):
        with cols[idx]:
            badge_html = (
                '<span class="badge-hot">Hot</span>'
                if item.get("badge") == "HOT"
                else '<span class="badge-new">New</span>'
            )
            st.markdown(f'<div class="drama-card">{badge_html}', unsafe_allow_html=True)
            if os.path.exists(item["poster_path"]):
                st.image(item["poster_path"], use_container_width=True)
            st.markdown(
                f'<div class="drama-title">{item["title"]}</div>'
                f'<div style="color:#b8b8c6;font-size:0.78rem;">▶ {item.get("views", "120K")}</div>'
                f'<div class="drama-tag">{item.get("category", "Drama")}</div></div>',
                unsafe_allow_html=True,
            )
            if st.button("▶ Watch", key=f"watch_{item['id']}", use_container_width=True):
                st.session_state.selected_drama_id = item["id"]
                st.rerun()