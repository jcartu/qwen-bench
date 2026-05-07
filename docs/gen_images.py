#!/usr/bin/env python3
"""Generate hero & section images for qwen-bench hub via Gemini 3.1 nano banana."""
import os, sys, pathlib
from google import genai

OUT = pathlib.Path(__file__).parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"])
MODEL = "gemini-3.1-flash-image-preview"

PROMPTS = {
    "hub_hero": (
        "A wide cinematic 16:9 hero banner for an open-source LLM benchmarking research hub. "
        "Centered: a glowing blueprint-style table or matrix grid with floating data points and "
        "tiny multi-colored result tiles, each tile tagged with a different study label. "
        "Behind the matrix, dual NVIDIA Blackwell-class data-center GPUs are visible in soft "
        "background bokeh. Above the matrix, a constellation of interconnected nodes glows "
        "(representing many studies forming one body of work). Deep navy and charcoal background, "
        "electric cyan, lime, and amber accents. Clean technical-poster aesthetic. "
        "No readable text, no logos, no watermarks. "
        "Style: precise modern infographic meets cinematic 3D render."
    ),
    "studies_index": (
        "A horizontal infographic illustration of a research studies index: a vertical timeline "
        "running left-to-right, with 4-5 glowing study cards anchored along it. Each card is "
        "stylized as a folder or report icon with a colored tag (cyan, amber, magenta, lime, "
        "violet) and small abstract chart silhouettes inside. Connecting glowing line traces "
        "show how each study links to the next. Dark navy backdrop, neon technical aesthetic. "
        "No readable text, no logos, no watermarks."
    ),
    "sota_leaderboard": (
        "A clean illustration of a leaderboard / SOTA scoreboard for ML inference benchmarks. "
        "Three horizontal podium-style bars stacked vertically, each glowing a different color "
        "(gold, silver, bronze accent), with abstract performance graph silhouettes behind them. "
        "To the side, a small grid of comparison heatmap cells. Dark cinematic background, "
        "neon cyan and amber highlights, modern data-visualization poster aesthetic. "
        "No readable text labels, no logos, no watermarks."
    ),
}

for name, prompt in PROMPTS.items():
    out_path = OUT / f"{name}.png"
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"[skip] {out_path} exists ({out_path.stat().st_size} bytes)")
        continue
    print(f"[gen]  {name}: {prompt[:80]}...")
    try:
        resp = client.models.generate_content(model=MODEL, contents=prompt)
        wrote = False
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.data:
                out_path.write_bytes(part.inline_data.data)
                print(f"[ok]   {out_path} ({out_path.stat().st_size} bytes)")
                wrote = True
                break
        if not wrote:
            print(f"[warn] no image data for {name}")
    except Exception as e:
        print(f"[err]  {name}: {e}", file=sys.stderr)
        sys.exit(1)
