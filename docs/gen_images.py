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
        "A wide 16:9 horizontal infographic illustration showing a horizontal sequence of "
        "five glowing translucent glass folder tiles, anchored along a glowing horizontal "
        "timeline line that runs across the full width of the image. Each tile has a "
        "thin colored band of light glowing at the top of its rounded rectangular shape "
        "(this band is the tile's color identity, NOT a label). The five tiles' top-band "
        "colors, in their order along the timeline, are: electric cyan, warm amber, hot "
        "magenta, lime green, deep violet. The five tile interiors contain only pure "
        "abstract geometric chart silhouettes (vertical bars, plotted dots, smooth line "
        "curves, scatter clouds). Each tile's interior is progressively richer than the "
        "previous one in the sequence — the first tile has a single simple bar chart, the "
        "final tile has a complex layered combination of bars, lines, and dots. "
        "Four glowing connecting arrows flow along the timeline between adjacent tiles, "
        "each arrow rendered as a smooth color gradient that matches its two neighboring "
        "tiles: arrow 1 fades cyan into amber, arrow 2 fades amber into magenta, arrow 3 "
        "fades magenta into lime green (its arrowhead is PURE LIME GREEN where it meets "
        "the lime tile, never any other color), arrow 4 fades lime green into violet. "
        "Small bright data-point sparks travel along the arrows. The final (violet) tile "
        "has a small glowing lime-green checkmark seal in its bottom-right corner. "
        "Deep navy and charcoal background with a soft technical grid pattern, neon "
        "aesthetic. Tile interiors must contain ONLY chart shapes (bars, dots, lines, "
        "ticks, scatter points). "
        "DO NOT add any of the following anywhere in the image: "
        "  - readable words, letters, alphabet characters, numerals, or digits "
        "  - color names or position words rendered as visible writing "
        "  - tab labels, folder labels, file names, or caption text "
        "  - alphabetic markers on the arrows (no A, B, C, D, no Roman numerals) "
        "  - horizontal stripes, bars, or dashed rows inside tiles that resemble text "
        "  - watermarks, logos, brand names, or printed badges "
        "  - smudged glyphs or pseudo-text squiggles "
        "  - empty rectangles or callout boxes that look like missing labels "
        "Every surface that could carry text must instead be a clean blank color band or "
        "a pure geometric chart silhouette. The image must be 100% wordless."
    ),
    "sota_leaderboard": (
        "A clean cinematic 16:9 illustration of a leaderboard / SOTA scoreboard for ML "
        "inference benchmarks running on dual data-center GPUs. Centered: three horizontal "
        "podium-style bars stacked vertically, each glowing a different metallic accent "
        "(gold on top, silver in middle, bronze at bottom), each bar with stylized "
        "abstract bar-chart silhouettes flowing inside it as if it contains live data. "
        "To the right of the podium, a medium-size grid of comparison heatmap cells glows "
        "with varying cyan-to-amber intensities. To the left, three small floating result "
        "tiles hover with mini line plots inside. In the BACKGROUND (in soft depth-of-field "
        "bokeh, but still CLEARLY recognizable as actual GPU hardware), there are TWO "
        "generic high-end server GPU cards visible. Each GPU is a long rectangular "
        "dual-slot PCIe card with a dark matte-black metallic shroud, a large circular "
        "blower-style cooling fan visible at one end of each card, fine parallel "
        "heat-sink fin lines along the side, and a thin green/black printed circuit board "
        "edge at the bottom with subtle gold-colored PCIe contact teeth. The two GPU "
        "cards sit side-by-side, connected by a thin glowing horizontal bridge connector "
        "spanning their tops. CRITICAL: The GPU shrouds must be COMPLETELY BLANK — NO "
        "brand names, NO manufacturer logos, NO product names, NO model numbers, NO "
        "vents shaped like letters, NO printed badges, NO stickers, NO smudged glyphs that "
        "resemble text or logos. The shroud surface should be plain dark metal with only "
        "the cooling fan cutout, fin pattern, and PCB edge visible. Render them with "
        "realistic hardware detail (shroud, fan, fins, PCB, bridge) but with NO printed "
        "markings of any kind on any surface. "
        "Deep navy and charcoal cinematic background, neon cyan and amber highlights as "
        "primary accents, metallic gold/silver/bronze on the podium bars. Modern "
        "data-visualization poster aesthetic, sharp and precise on the foreground podium "
        "and tiles, soft bokeh on the GPU hardware. AVOID any letters, numerals, or "
        "text-like glyphs anywhere — use only pure geometric shapes for charts. "
        "ABSOLUTELY NO TEXT, NO LETTERS, NO NUMBERS, NO LOGOS, NO BRAND NAMES, NO "
        "WATERMARKS, NO LABELS, NO PRINTED BADGES, NO SMUDGED PSEUDO-LOGOS."
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
