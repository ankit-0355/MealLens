#!/usr/bin/env python3
import os
import json
import time
from huggingface_hub import InferenceClient

# ─── LOAD CONFIG ───────────────────────────────────────────────────────────────
# config.json should live alongside this script.
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")

if os.path.isfile(config_path):
    with open(config_path, "r", encoding="utf-8") as cfg:
        config = json.load(cfg)
else:
    raise FileNotFoundError(f"config.json not found at {config_path}")

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
HF_MODEL        = config.get("hf_model")
HF_PROVIDER     = config.get("hf_provider")
HF_TOKEN        = config.get("hf_token")
PROMPT_TEMPLATE = config.get(
    "prompt_template",
    "Summarize or extract key items from the following description:\n{content}"
)

# Paths from config.json may be absolute or relative to the project root.
input_path_cfg  = config.get("input_json_path")
output_path_cfg = config.get("output_json_path")

# Resolve input/output paths: if relative, interpret relative to script directory
if not os.path.isabs(input_path_cfg):
    INPUT_JSON_PATH = os.path.join(script_dir, input_path_cfg)
else:
    INPUT_JSON_PATH = input_path_cfg

if not os.path.isabs(output_path_cfg):
    OUTPUT_JSON_PATH = os.path.join(script_dir, output_path_cfg)
else:
    OUTPUT_JSON_PATH = output_path_cfg

# Validate required settings
required = {
    "hf_model": HF_MODEL,
    "hf_provider": HF_PROVIDER,
    "hf_token": HF_TOKEN,
    "input_json_path": INPUT_JSON_PATH,
    "output_json_path": OUTPUT_JSON_PATH
}
for name, val in required.items():
    if not val:
        raise ValueError(f"Missing '{name}' in config.json or invalid path")
# ────────────────────────────────────────────────────────────────────────────────

# Initialize the Hugging Face Inference Client
client = InferenceClient(
    provider=HF_PROVIDER,
    api_key=HF_TOKEN,
)

def query_llm(prompt: str) -> str:
    """
    Send a prompt to the HF InferenceClient and return the generated text.
    """
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def process_json(input_path: str, output_path: str):
    """
    Read the input JSON, query the LLM for each image's description,
    and save the results to the output JSON.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}
    for img, desc_list in data.items():
        content = " ".join(desc_list)
        prompt = PROMPT_TEMPLATE.format(content=content)
        print(f"Processing {img}...", end="", flush=True)
        try:
            text = query_llm(prompt)
            results[img] = text
            print(" ✅")
        except Exception as e:
            results[img] = f"ERROR: {e}"
            print(" ❌")
        time.sleep(1.5)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nAll done. Results written to {output_path}")


if __name__ == "__main__":
    process_json(INPUT_JSON_PATH, OUTPUT_JSON_PATH)