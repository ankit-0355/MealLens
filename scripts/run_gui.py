import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess, os, sys, shlex

# ─── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SEMANTIC_PY    = os.path.join(SCRIPT_DIR, "semantic.py")
PANOPTIC_PY    = os.path.join(SCRIPT_DIR, "panoptic.py")
CLEAN_BOXES_PY = os.path.join(SCRIPT_DIR, "clean_boxes.py")
CROP_BOXES_PY  = os.path.join(SCRIPT_DIR, "crop_boxes.py")

# ─── GUI callbacks ────────────────────────────────────────────────────────────
def select_image():
    path = filedialog.askopenfilename(
        title="Select a fridge image",
        filetypes=[("Image","*.jpg *.jpeg *.png")]
    )
    if path:
        image_path_var.set(path)

def run_pipeline():
    img = image_path_var.get()
    if not img:
        return messagebox.showwarning("No image", "Please select an image first.")
    if not os.path.isfile(img):
        return messagebox.showerror("Not found", f"Cannot find:\n{img}")

    base = os.path.splitext(os.path.basename(img))[0]
    out_sem   = os.path.join(PROJECT_ROOT, "output", base)
    out_boxes = os.path.join(PROJECT_ROOT, "output", "boxes", base)

    # 1) semantic.py
    # 2) panoptic.py
    # 3) clean_boxes.py
    # 4) crop_boxes.py
    steps = [
        (SEMANTIC_PY, [
            "--img_path",       img,
            "--output",         os.path.join(PROJECT_ROOT, "output"),
            "--SAM_checkpoint", os.path.join(PROJECT_ROOT, "ckpts","sam_vit_h_4b8939.pth"),
            "--device",         "cpu",
            "--area_thr",       "1000",
            "--top_k",          "60",
        ]),
        (PANOPTIC_PY, [
            "--semantic_dir",   out_sem,
            "--output",         os.path.join(PROJECT_ROOT,"output","boxes"),
            "--area_thr",       "1000",
            "--top_k",          "60",
        ]),
        (CLEAN_BOXES_PY, [
            "--image",          os.path.join(out_boxes,"input.jpg"),
            "--boxes_json",     os.path.join(out_boxes,f"{base}.json"),
            "--disable_tiny_filter",
            "--disable_gap_glue"
        ] + (["--cluster"] if cluster_var.get() else [])),
        (CROP_BOXES_PY, [
            "--image",          os.path.join(out_boxes,"input.jpg"),
            "--json",           os.path.join(out_boxes,f"{base}_cleaned.json"),
            "--out_dir",        "crops",
            "--clean"
        ]),
    ]

    for script, args_list in steps:
        cmd = [sys.executable, script] + args_list
        # Print the exact command
        print(f"\n▶️ Running: {shlex.join(cmd)}")
        # Run and let stdout/stderr flow directly to your terminal
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            return messagebox.showerror(
                "Pipeline Error",
                f"Step failed:\n  {shlex.join(cmd)}\n\n"
                f"Exit code: {proc.returncode}"
            )

    #messagebox.showinfo("Done", f"Pipeline completed for:\n{base}")
    root.destroy()

# ─── Build GUI ────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Food Detection Pipeline")

frm = tk.Frame(root, padx=20, pady=20)
frm.pack()

tk.Label(frm, text="Select an image to run detection:").pack(pady=(0,5))
image_path_var = tk.StringVar()
tk.Entry(frm, textvariable=image_path_var, width=60, state="readonly").pack()
tk.Button(frm, text="Browse Image", command=select_image).pack(pady=(5,15))

cluster_var = tk.BooleanVar()
tk.Checkbutton(
    frm,
    text="Enable Cluster Merge (--cluster)",
    variable=cluster_var
).pack(pady=(0,15))

tk.Button(frm, text="Run Pipeline", command=run_pipeline).pack()

root.mainloop()
