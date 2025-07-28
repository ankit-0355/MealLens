"""
clean_boxes.py  –  MealLens utility
-----------------------------------
Loads merged_boxes.json from scripts/panoptic.py and performs:
1. Containment pruning
2. IoU-0.5 Non-Max Suppression
3. Tiny + aspect filter         (default on)
4. Gap glue (≤ 3 px gap)        (default on)
5. Optional cluster merge (IoU≥0.2)  --cluster
Outputs:
* <json>_cleaned.json
* Opens a window with yellow boxes   (--no-window to skip)
* Prints each bbox to terminal
"""

import os, json, argparse, cv2, numpy as np, sys
import matplotlib.pyplot as plt

# ── helper functions ──────────────────────────────────────────────────────────
def to_xyxy(b):  # [x,y,w,h] → [x1,y1,x2,y2]
    x,y,w,h = b
    return [x, y, x+w, y+h]

def iou(a,b):
    x1,y1,x2,y2 = a; X1,Y1,X2,Y2 = b
    xi1, yi1 = max(x1,X1), max(y1,Y1)
    xi2, yi2 = min(x2,X2), min(y2,Y2)
    if xi2<=xi1 or yi2<=yi1: return 0.
    inter = (xi2-xi1)*(yi2-yi1)
    return inter / ((x2-x1)*(y2-y1) + (X2-X1)*(Y2-Y1) - inter)

def fully_inside(a,b,margin=2):
    x1,y1,x2,y2 = a; X1,Y1,X2,Y2 = b
    return x1>=X1+margin and y1>=Y1+margin and x2<=X2-margin and y2<=Y2-margin

def gap_glue(boxes, gap=3, min_overlap=0.7):
    """
    Merge boxes that are separated by <= gap px horizontally OR vertically
    while overlapping strongly in the orthogonal axis.
    """
    boxes = [list(b) for b in boxes]             # mutable
    merged=[]
    while boxes:
        x1,y1,x2,y2 = boxes.pop(0)
        j=0
        while j < len(boxes):
            X1,Y1,X2,Y2 = boxes[j]
            horiz_gap = max(0, max(X1-x2, x1-X2))
            vert_gap  = max(0, max(Y1-y2, y1-Y2))
            horiz_overlap = max(0, min(y2,Y2) - max(y1,Y1))
            vert_overlap  = max(0, min(x2,X2) - max(x1,X1))
            cond_h = horiz_gap<=gap and horiz_overlap >= min_overlap*min(y2-y1, Y2-Y1)
            cond_v = vert_gap <=gap and vert_overlap  >= min_overlap*min(x2-x1, X2-X1)
            if cond_h or cond_v:
                # merge
                x1, y1 = min(x1,X1), min(y1,Y1)
                x2, y2 = max(x2,X2), max(y2,Y2)
                boxes.pop(j)   # remove merged box
            else:
                j+=1
        merged.append([x1,y1,x2,y2])
    return merged

# ── argument parsing ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="input.jpg (copied by panoptic.py)")
ap.add_argument("--boxes_json", required=True, help="merged_boxes.json from panoptic.py")
ap.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for NMS")
ap.add_argument("--cluster", action="store_true", help="extra cluster merge IoU>=0.2")
ap.add_argument("--disable_gap_glue", action="store_true", help="turn off gap merge")
ap.add_argument("--disable_tiny_filter", action="store_true", help="keep tiny/aspect boxes")
ap.add_argument("--no_window", action="store_true", help="skip OpenCV pop-up")
args = ap.parse_args()

# ── load image & boxes ────────────────────────────────────────────────────────
img = cv2.imread(args.image)
if img is None:
    sys.exit(f"[ERR] cannot read image {args.image}")
with open(args.boxes_json,"r",encoding="utf-8") as f:
    ann = json.load(f)
boxes = [to_xyxy(a["bbox"]) for a in ann["annotations"]]

# ── 1. containment pruning ────────────────────────────────────────────────────
boxes = [b for i,b in enumerate(boxes)
         if not any(i!=j and fully_inside(b, boxes[j]) for j in range(len(boxes)))]

# ── 2. IoU-NMS (largest area first) ───────────────────────────────────────────
areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
order = np.argsort(areas)[::-1]
keep_flags = [True]*len(boxes)
for i in range(len(order)):
    if not keep_flags[order[i]]: continue
    for j in range(i+1,len(order)):
        if keep_flags[order[j]] and iou(boxes[order[i]], boxes[order[j]]) >= args.iou_thr:
            keep_flags[order[j]] = False
boxes = [b for b,k in zip(boxes, keep_flags) if k]

# ── 3a. tiny/aspect-ratio filter ─────────────────────────────────────────────
if not args.disable_tiny_filter and boxes:
    areas = np.array([(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes])
    median = np.median(areas)
    boxes = [b for b,a in zip(boxes,areas)
             if a >= 0.02*median and 0.25 <= (b[2]-b[0])/(b[3]-b[1]) <= 4]

# ── 3b. gap glue ──────────────────────────────────────────────────────────────
if not args.disable_gap_glue:
    boxes = gap_glue(boxes, gap=3, min_overlap=0.7)

# ── 4. optional cluster merge (IoU>=0.2) ─────────────────────────────────────
if args.cluster:
    clusters=[]
    for b in boxes:
        placed=False
        for c in clusters:
            if any(iou(b,o)>=0.20 for o in c):
                c.append(b); placed=True; break
        if not placed: clusters.append([b])
    boxes=[]
    for c in clusters:
        xs,ys,x2s,y2s = zip(*c)
        boxes.append([min(xs),min(ys),max(x2s),max(y2s)])

# ── 5. save cleaned JSON ──────────────────────────────────────────────────────
out_json = os.path.splitext(args.boxes_json)[0] + "_cleaned.json"
clean_ann = {
    "image": os.path.basename(args.image),
    "annotations": [
        {"id": i, "bbox": [int(x1),int(y1),int(x2-x1),int(y2-y1)]}
        for i,(x1,y1,x2,y2) in enumerate(boxes)
    ]
}
with open(out_json,"w",encoding="utf-8") as f: json.dump(clean_ann,f,indent=2)
print(f" Saved {len(boxes)} cleaned boxes -> {out_json}\n")

# ── 6. print list & visualise ────────────────────────────────────────────────
for i,(x1,y1,x2,y2) in enumerate(boxes):
    print(f"{i:02d}: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")

if not args.no_window:
    vis = img.copy()
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,255),2)
    # try:
    #     cv2.imshow("Cleaned Boxes", vis)
    #     print("\n[INFO] Press any key in the image window to close…")
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # except cv2.error:
    #     # fallback to matplotlib
    #     plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    #     plt.axis("off"); plt.title("Cleaned Boxes"); plt.show()
