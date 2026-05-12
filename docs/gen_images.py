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
        "A wide cinematic 16:9 hero banner for an open-source LLM benchmarking research "
        "hub on NVIDIA Blackwell hardware. Centered foreground: a large glowing 3D matrix "
        "of result tiles arranged as a 4x6 grid, each tile a different accent color and "
        "showing stylized abstract chart silhouettes inside (bar charts, line plots, heat "
        "map cells, sparkline pulses). The matrix has subtle depth, slight tilt for "
        "3D perspective, with translucent glass-like tiles glowing from within. Behind "
        "and below the matrix, two NVIDIA Blackwell-class data-center GPUs sit in a server "
        "rack rendered in soft deep bokeh, with thin glowing tensor-parallel link traces "
        "flowing between them. Above the matrix, instead of callout bubbles or speech "
        "balloons, there is a CONSTELLATION of 5 to 7 small glowing solid orb nodes "
        "floating in space, each orb a different solid color (cyan, amber, magenta, lime, "
        "violet, gold), connected only by thin faint glowing line traces between them. "
        "The orbs are simple geometric spheres with internal glow — they have NO callout "
        "boxes, NO speech bubbles, NO label boxes, NO empty rectangles attached. To the "
        "far right side of the matrix, a single bright lime-green checkmark seal glows "
        "subtly, hinting at production validation. AVOID any letters, characters, numerals, "
        "code, glyphs that look like text, pseudo-text squiggles, OR any empty rectangular "
        "frames / callout shapes that LOOK like they should contain text. Use only pure "
        "geometric chart shapes (bars, dots, lines, ticks). Deep navy and charcoal "
        "background, electric cyan and amber as the dominant accents, lime green and "
        "magenta as secondary highlights. Clean modern technical-poster aesthetic, precise "
        "infographic feel with cinematic 3D render quality, sharp focus on the matrix, "
        "soft depth-of-field behind. "
        "ABSOLUTELY NO TEXT, NO LETTERS, NO NUMBERS, NO LOGOS, NO WATERMARKS, NO LABELS, "
        "NO EMPTY CALLOUT BOXES, NO SPEECH BUBBLES."
    ),
    "studies_index": (
        "A wide 16:9 horizontal infographic illustration of a research studies index. A "
        "glowing horizontal timeline runs from left to right across the image, with 5 "
        "distinct study cards anchored along it at evenly-spaced positions. Each card is "
        "stylized as a translucent glass folder or report tile, with a different colored "
        "tag at its top (cyan, amber, magenta, lime, violet from left to right) and small "
        "abstract chart silhouettes inside (bar charts, line plots, scatter dots). The "
        "cards progressively gain visual richness left-to-right — the leftmost is simple, "
        "the rightmost is most elaborate — suggesting cumulative depth. Connecting glowing "
        "line traces flow between cards along the timeline, with small data-point sparks "
        "traveling along them. The rightmost card has a subtle green checkmark seal in "
        "its corner, suggesting it's the production-validated outcome. Deep navy backdrop, "
        "neon technical aesthetic with electric cyan + amber + lime accents. AVOID any "
        "letters, numerals, or text-like glyphs — use only pure geometric shapes. "
        "ABSOLUTELY NO TEXT, NO LETTERS, NO NUMBERS, NO LOGOS, NO WATERMARKS, NO LABELS."
    ),
    "sota_leaderboard": (
        "A clean cinematic 16:9 illustration of a leaderboard / SOTA scoreboard for ML "
        "inference benchmarks. Centered: three horizontal podium-style bars stacked "
        "vertically, each glowing a different metallic accent (gold on top, silver in "
        "middle, bronze at bottom), each bar with stylized abstract bar-chart silhouettes "
        "flowing inside it as if it contains live data. To the right of the podium, a "
        "medium-size grid of comparison heatmap cells glows with varying cyan-to-amber "
        "intensities. To the left, three small floating result tiles hover with mini line "
        "plots inside. In the background, a soft bokeh of dual Blackwell-class GPUs hints "
        "at the hardware platform. Deep navy and charcoal cinematic background, neon cyan "
        "and amber highlights as primary accents, metallic gold/silver/bronze on the podium "
        "bars. Modern data-visualization poster aesthetic, sharp and precise. AVOID any "
        "letters, numerals, or text-like glyphs — use only pure geometric shapes. "
        "ABSOLUTELY NO TEXT, NO LETTERS, NO NUMBERS, NO LOGOS, NO WATERMARKS, NO LABELS."
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
