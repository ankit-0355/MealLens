import os
import cv2
import json
import argparse
import numpy as np
import shutil

def parse_args():
    p = argparse.ArgumentParser(description="Convert SAM masks -> bounding boxes JSON")
    p.add_argument("--semantic_dir", type=str, required=True,
                   help="Folder created by semantic_samonly (contains sam_masks/ and input.jpg)")
    p.add_argument("--output",       type=str, default="Output/SAM_Boxes",
                   help="Where to write bounding box JSON + copy of input.jpg")
    p.add_argument("--area_thr",     type=int, default=1000, help="Min area for bbox (px²)")
    p.add_argument("--top_k",        type=int, default=60, help="Max number of boxes")
    return p.parse_args()

def extract_boxes(masks_np, area_thr, top_k):
    boxes, areas = [], []
    for m in masks_np:
        area = int((m>0).sum())
        if area < area_thr:
            continue
        ys, xs = np.where(m>0)
        x1,y1,x2,y2 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes.append([int(x1),int(y1),int(x2-x1),int(y2-y1)])
        areas.append(area)
    idxs = np.argsort(areas)[::-1][:top_k]
    return [boxes[i] for i in idxs]

def main():
    args = parse_args()
    mask_file = os.path.join(args.semantic_dir, "sam_masks", "masks.npy")
    masks_np = np.load(mask_file)

    boxes = extract_boxes(masks_np, args.area_thr, args.top_k)

    base = os.path.basename(args.semantic_dir)
    ann = {"image": base, "annotations": []}
    for i, (x, y, w, h) in enumerate(boxes):
        ann["annotations"].append({"id": i, "bbox": [x, y, w, h]})

    outdir = os.path.join(args.output, base)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)

    shutil.copy(os.path.join(args.semantic_dir, "input.jpg"),
                os.path.join(outdir, "input.jpg"))

    print(f"Saved {len(boxes)} boxes -> {outdir}/{base}.json")

if __name__ == "__main__":
    main()
