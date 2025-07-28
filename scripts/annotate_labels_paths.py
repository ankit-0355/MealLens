#!/usr/bin/env python3
"""
annotate_and_view_fixed_scroller.py

Draw bounding boxes and display food name labels above each box.
Image is resized to fit screen — no scrolling needed.
"""

import os
import cv2
import json
import tkinter as tk
from tkinter import BOTH, X, Frame, Canvas, Text, Scrollbar, HORIZONTAL
from PIL import Image, ImageTk

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMAGE_PATH        = r"C:\Users\sahil\Desktop\Meal_final\output\fridge image (2)\input.jpg"
CLEANED_JSON_PATH = r"C:\Users\sahil\Desktop\Meal_final\output\boxes\fridge image (2)\fridge image (2)_cleaned.json"
LABELS_JSON_PATH  = r"C:\Users\sahil\Desktop\Meal_final\crops\lm_output.json"
OUTPUT_PATH       = r"C:\Users\sahil\Desktop\Meal_final\output\fridge image (2)\output_labeled.jpg"

SCALE             = 0.9        # shrink to fit screen without scrollbars
SCROLLER_HEIGHT   = 18
FONT_NAME         = "Arial"
FONT_SIZE         = 10
BOX_COLOR         = (0, 255, 255)  # Yellow
MAX_WIDTH         = 220            # max label width
# ──────────────────────────────────────────────────────────────────────────────

def annotate_boxes(image_path, ann_path):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    for obj in ann.get("annotations", []):
        x, y, w, h = map(int, obj["bbox"])
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), BOX_COLOR, 2)
    return img_bgr, ann

def load_labels(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)

def show_gui(img_bgr, ann, name_map):
    # Resize image
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    W, H = pil.width, pil.height
    new_w, new_h = int(W * SCALE), int(H * SCALE)
    pil = pil.resize((new_w, new_h), resample=Image.LANCZOS)

    root = tk.Tk()
    root.title("Food Labels")

    canv = Canvas(root, width=new_w, height=new_h)
    canv.pack(expand=True, fill=BOTH)

    tk_img = ImageTk.PhotoImage(pil)
    canv.create_image(0, 0, image=tk_img, anchor="nw")

    placed = []
    for obj in ann.get("annotations", []):
        x, y, w, h = map(int, obj["bbox"])
        sx, sy, sw = int(x * SCALE), int(y * SCALE), int(w * SCALE)

        # Load label and clean
        key = f"box_{obj['id']:02d}.jpg"
        raw = name_map.get(key, "")
        if raw.startswith("ERROR:"):
            text = "LLM issue"
        else:
            text = raw or "NA"

        width = min(max(sw, 60), MAX_WIDTH)  # stretch label up to MAX_WIDTH

        # label position
        ly = sy - SCROLLER_HEIGHT - 2
        if ly < 0:
            ly = sy + 2

        # frame and text
        frame = Frame(canv, width=width, height=SCROLLER_HEIGHT,
                      highlightthickness=1, highlightbackground="black")
        frame.pack_propagate(False)

        txt = Text(frame, height=1, wrap="none",
                   font=(FONT_NAME, FONT_SIZE),
                   bg="white", fg="red", bd=0)
        sb = Scrollbar(frame, orient=HORIZONTAL, command=txt.xview)
        txt.config(xscrollcommand=sb.set)
        txt.insert("1.0", text)
        txt.config(state="disabled")
        txt.pack(side="top", fill=X, expand=True)
        sb.pack(side="bottom", fill=X)

        canv.create_window(sx, ly, anchor="nw", window=frame)

    canv.image = tk_img
    root.mainloop()

if __name__ == "__main__":
    img_bgr, ann_data = annotate_boxes(IMAGE_PATH, CLEANED_JSON_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, img_bgr)
    print(f"✅ Saved boxed image to {OUTPUT_PATH}")

    labels = load_labels(LABELS_JSON_PATH)
    show_gui(img_bgr, ann_data, labels)