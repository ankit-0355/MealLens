"""
crop_boxes.py  – Crop each bounding-box region to a JPEG
──────────────────────────────────────────────────────────
usage examples
──────────────
# overwrite old crops
python scripts/crop_boxes.py \
  --image "output/boxes/fridge image (2)/input.jpg" \
  --json  "output/boxes/fridge image (2)/fridge image (2)_cleaned.json" \
  --out_dir crops --clean

# keep previous runs in a timestamped folder
python scripts/crop_boxes.py \
  --image output/boxes/img_X/input.jpg \
  --json  output/boxes/img_X/img_X_cleaned.json \
  --out_dir crops/2025-07-11_10-30-25
"""

import cv2, json, os, glob, argparse, datetime

# ── arguments ────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--image",   required=True, help="Path to input.jpg")
ap.add_argument("--json",    required=True, help="Path to *boxes*.json (cleaned or merged)")
ap.add_argument("--out_dir", default="crops", help="Where to write box_XX.jpg")
ap.add_argument("--clean",   action="store_true", help="Delete old JPGs in out_dir first")
args = ap.parse_args()

img_path  = args.image
json_path = args.json
#crop_dir  = args.out_dir
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))        # .../MEAL_FINAL/scripts
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))   # .../MEAL_FINAL
crop_dir     = os.path.join(PROJECT_ROOT, args.out_dir)    

# ── 1) optional wipe ─────────────────────────────────────────────────────────
if args.clean and os.path.isdir(crop_dir):
    for f in glob.glob(os.path.join(crop_dir, "*.jpg")):
        os.remove(f)

os.makedirs(crop_dir, exist_ok=True)

# ── 2) load image & boxes ────────────────────────────────────────────────────
img  = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {img_path}")

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ── 3) crop & save ───────────────────────────────────────────────────────────
for i, obj in enumerate(data["annotations"]):
    x,y,w,h = map(int, obj["bbox"])
    crop = img[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(crop_dir, f"box_{i:02d}.jpg"), crop)

print(f"✅  Saved {len(data['annotations'])} crops → {os.path.abspath(crop_dir)}")
