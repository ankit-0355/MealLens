import os
import cv2
import argparse
import numpy as np
import shutil
import logging
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import Any, Dict, List

def parse_args():
    p = argparse.ArgumentParser(description="Generate SAM masks (filtered) for one image")
    p.add_argument("--img_path",   type=str, required=True, help="Path to input image")
    p.add_argument("--output",     type=str, default="Output/SAM_Only_Results", help="Base output folder")
    p.add_argument("--SAM_checkpoint", type=str, default="ckpts/sam_vit_h_4b8939.pth", help="SAM model .pth")
    p.add_argument("--model_type", type=str, default="vit_h", help="SAM model type")
    p.add_argument("--device",     type=str, default="cpu", help="'cpu' or 'cuda'")
    p.add_argument("--area_thr",   type=int, default=1000, help="Minimum mask area to keep (px²)")
    p.add_argument("--top_k",      type=int, default=60, help="Keep only the top-K largest masks")
    return p.parse_args()

def enhance_masks_simple(masks: List[Dict[str, Any]], area_thr: int, top_k: int):
    filtered = [m for m in masks if m["area"] >= area_thr]
    return sorted(filtered, key=lambda m: m["area"], reverse=True)[:top_k]

def write_masks(masks: List[Dict[str, Any]], folder: str):
    os.makedirs(os.path.join(folder, "sam_masks"), exist_ok=True)
    meta = ["id,area,x,y,w,h"]
    arrs = []
    for i, m in enumerate(masks):
        mask = (m["segmentation"].astype("uint8") * 255)
        cv2.imwrite(os.path.join(folder, "sam_masks", f"{i}.png"), mask)
        arrs.append(mask)
        x, y, w, h = m["bbox"]
        meta.append(f"{i},{m['area']},{x},{y},{w},{h}")
    np.save(os.path.join(folder, "sam_masks", "masks.npy"), np.stack(arrs))
    with open(os.path.join(folder, "sam_metadata.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta))

def create_logger(path: str):
    os.makedirs(path, exist_ok=True)
    logf = os.path.join(path, "sam_process.log")
    fh = logging.FileHandler(logf, encoding="utf-8", mode='w')
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def main():
    args = parse_args()
    logger = create_logger(args.output)
    logger.info("Loading SAM model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.SAM_checkpoint)
    sam.to(args.device)
    gen = SamAutomaticMaskGenerator(sam)

    logger.info(f"Reading image {args.img_path}")
    img = cv2.imread(args.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    logger.info("Generating masks...")
    masks = gen.generate(img)

    logger.info(f"Enhancing masks by area >= {args.area_thr} and top_k={args.top_k}")
    masks = enhance_masks_simple(masks, args.area_thr, args.top_k)

    base = os.path.splitext(os.path.basename(args.img_path))[0]
    outdir = os.path.join(args.output, base)
    logger.info(f"Saving {len(masks)} masks -> {outdir}")
    write_masks(masks, outdir)
    shutil.copy(args.img_path, os.path.join(outdir, "input.jpg"))
    logger.info("Done.")

if __name__ == "__main__":
    main()
