# app.py
import os
import sys
import shutil
import subprocess
import tempfile
import json
import re

import streamlit as st
from PIL import Image
import cv2

# ──────────────────────────────────────────────────────────────────────────────
# Force UTF‑8 for subprocess I/O
# ──────────────────────────────────────────────────────────────────────────────
ENV = os.environ.copy()
ENV["PYTHONIOENCODING"] = "utf-8"

# ──────────────────────────────────────────────────────────────────────────────
# Resolve paths to your helper scripts
# ──────────────────────────────────────────────────────────────────────────────
ROOT           = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR    = os.path.join(ROOT, "scripts")
SEMANTIC_PY    = os.path.join(SCRIPTS_DIR, "semantic.py")
PANOPTIC_PY    = os.path.join(SCRIPTS_DIR, "panoptic.py")
CLEAN_BOXES_PY = os.path.join(SCRIPTS_DIR, "clean_boxes.py")
CROP_BOXES_PY  = os.path.join(SCRIPTS_DIR, "crop_boxes.py")

# ──────────────────────────────────────────────────────────────────────────────
# Helper: run a shell command, optionally hide per-box lines (like "12: x=…")
# ──────────────────────────────────────────────────────────────────────────────
def run_step(cmd: str, hide_box_prints: bool = False):
    box_re = re.compile(r"^\d+:")
    p = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=ENV,
    )
    for line in p.stdout:
        line = line.rstrip()
        if hide_box_prints and box_re.match(line):
            continue
        yield line
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Step failed (exit {p.returncode})")

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("📦 Food Detection Pipeline")

uploaded = st.file_uploader("Upload a fridge image", type=["jpg","jpeg","png"])
cluster  = st.checkbox("Enable Cluster Merge (--cluster)")

if uploaded and st.button("Run Pipeline"):
    # ─── derive a safe basename (alphanumeric + _ + -) ─────────────────────────
    raw  = os.path.splitext(uploaded.name)[0]
    safe = re.sub(r"[^\w\-]", "_", raw)

    # ─── prepare output folders ────────────────────────────────────────────────
    out_base   = os.path.join(ROOT, "output", safe)
    sem_nested = os.path.join(out_base, safe)    # semantic.py will nest here
    boxes_dir  = os.path.join(out_base, "boxes")
    crops_dir  = os.path.join(ROOT, "crops", safe)

    # ─── clear previous runs ───────────────────────────────────────────────────
    shutil.rmtree(os.path.join(ROOT, "output"), ignore_errors=True)
    shutil.rmtree(os.path.join(ROOT, "crops"),  ignore_errors=True)

    # ─── save uploaded file to a fixed path ───────────────────────────────────
    os.makedirs(out_base, exist_ok=True)
    img_path = os.path.join(out_base, f"{safe}.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded.getbuffer())

    log = st.empty()

    try:
        # 1) semantic.py → out_base/safe/sam_masks/masks.npy
        cmd1 = (
            f'{sys.executable} "{SEMANTIC_PY}" '
            f'--img_path "{img_path}" '
            f'--output "{out_base}" '
            f'--SAM_checkpoint "{ROOT}/ckpts/sam_vit_h_4b8939.pth" '
            f'--device cpu --area_thr 1000 --top_k 60'
        )
        log.text(f"▶️ {cmd1}")
        for L in run_step(cmd1):
            log.text(L)

        # 2) panoptic.py → reads from sem_nested
        cmd2 = (
            f'{sys.executable} "{PANOPTIC_PY}" '
            f'--semantic_dir "{sem_nested}" '
            f'--output "{boxes_dir}" '
            f'--area_thr 1000 --top_k 60'
        )
        log.text(f"▶️ {cmd2}")
        for L in run_step(cmd2):
            log.text(L)

        # 3) clean_boxes.py → no window & hide per-box prints
        cmd3 = (
            f'{sys.executable} "{CLEAN_BOXES_PY}" '
            f'--image "{boxes_dir}/{safe}/input.jpg" '
            f'--boxes_json "{boxes_dir}/{safe}/{safe}.json" '
            f'--no_window --disable_tiny_filter --disable_gap_glue '
            + ("--cluster" if cluster else "")
        )
        log.text(f"▶️ {cmd3}")
        for L in run_step(cmd3, hide_box_prints=True):
            log.text(L)

        # 4) crop_boxes.py → writes to crops_dir
        os.makedirs(crops_dir, exist_ok=True)
        cmd4 = (
            f'{sys.executable} "{CROP_BOXES_PY}" '
            f'--image "{boxes_dir}/{safe}/input.jpg" '
            f'--json "{boxes_dir}/{safe}/{safe}_cleaned.json" '
            f'--out_dir "{crops_dir}" --clean'
        )
        log.text(f"▶️ {cmd4}")
        for L in run_step(cmd4):
            log.text(L)

        # ─── annotate & save final cleaned image ────────────────────────────────
        img_bgr = cv2.imread(f"{boxes_dir}/{safe}/input.jpg")
        with open(f"{boxes_dir}/{safe}/{safe}_cleaned.json") as f:
            ann = json.load(f)
        for obj in ann["annotations"]:
            x,y,w,h = obj["bbox"]
            cv2.rectangle(img_bgr,(x,y),(x+w,y+h),(0,255,255),2)

        cleaned_path = os.path.join(out_base, f"{safe}_cleaned.jpg")
        cv2.imwrite(cleaned_path, img_bgr)

        # ─── Results Comparison only ────────────────────────────────────────────
        proc_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.subheader("Results Comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original")
            st.image(img_path, use_container_width=True)
        with c2:
            st.caption("Cleaned + Clustered")
            st.image(proc_rgb, use_container_width=True)

        st.success(f"✅ Cleaned image saved: `{cleaned_path}`")
        st.info(f"Crops saved to: `{crops_dir}`")

    except Exception as e:
        st.error(f"❌ Pipeline failed:\n{e}")
