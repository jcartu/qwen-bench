#!/usr/bin/env python3
"""Apply the filled HUB_FRONTDOOR_EDITS edits to SOTA.md, STUDIES.md, README.md.

Idempotent: each insertion is gated on a unique marker so re-running is safe.

Strategy: rather than parsing the markdown template by hand, we encode each
edit as a sentinel-anchored splice in this script. The template file remains
the human-readable reference, but the splicing logic lives here.
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path


# -- Sentinels (each must be present exactly once in the target file) --
SOTA_ANCHOR = "> 🆕 **2026-05-15 single-user update**"
STUDIES_ANCHOR = "## 2026-05-15 · Single-user thinking-budget addendum"
README_BADGE_FROM = "studies-10_published"
README_BADGE_TO = "studies-11_published"
README_STUDY_ANCHOR = "### 📖 2026-05 · Single-user thinking-budget addendum"


def already_applied(text: str, marker: str) -> bool:
    return marker in text


def splice_before(text: str, anchor: str, payload: str) -> str:
    if anchor not in text:
        raise RuntimeError(f"anchor not found: {anchor!r}")
    return text.replace(anchor, payload + "\n\n" + anchor, 1)


def splice_after_paragraph(text: str, anchor: str, payload: str) -> str:
    """Insert payload after the paragraph that begins with `anchor`."""
    if anchor not in text:
        raise RuntimeError(f"anchor not found: {anchor!r}")
    # Find end of paragraph: next blank line after the anchor.
    idx = text.index(anchor)
    after = text.index("\n\n", idx)
    return text[:after] + "\n\n" + payload + text[after:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-root", required=True)
    ap.add_argument("--filled", required=True,
                    help="HUB_FRONTDOOR_EDITS.filled.md produced by fill_placeholders.py")
    args = ap.parse_args()
    hub = Path(args.hub_root)
    filled = Path(args.filled).read_text()

    # -- Extract payloads from filled template using fenced blocks --
    blocks = re.findall(r"```markdown\n(.*?)```", filled, flags=re.DOTALL)
    if len(blocks) < 3:
        print(f"expected >=3 markdown payload blocks in template, got {len(blocks)}", file=sys.stderr)
        return 2
    sota_payload, studies_payload, readme_payload = blocks[0].rstrip(), blocks[1].rstrip(), blocks[2].rstrip()

    # -- 1) SOTA.md --
    sota = hub / "SOTA.md"
    s = sota.read_text()
    marker_sota = "🧪 **2026-05-15 generalization"
    if already_applied(s, marker_sota):
        print(f"SOTA.md: already applied (found marker {marker_sota!r})")
    else:
        s = splice_after_paragraph(s, SOTA_ANCHOR, sota_payload)
        sota.write_text(s)
        print(f"SOTA.md: spliced after {SOTA_ANCHOR!r}")

    # -- 2) STUDIES.md --
    stud = hub / "STUDIES.md"
    s = stud.read_text()
    marker_studies = "Thinking-budget generalization study (evening)"
    if already_applied(s, marker_studies):
        print(f"STUDIES.md: already applied")
    else:
        s = splice_before(s, STUDIES_ANCHOR, studies_payload)
        stud.write_text(s)
        print(f"STUDIES.md: spliced before {STUDIES_ANCHOR!r}")

    # -- 3) README.md --
    rd = hub / "README.md"
    s = rd.read_text()
    if README_BADGE_TO in s:
        print(f"README.md: badge already bumped")
    elif README_BADGE_FROM not in s:
        print(f"README.md: WARNING expected badge {README_BADGE_FROM!r} not found", file=sys.stderr)
    else:
        s = s.replace(README_BADGE_FROM, README_BADGE_TO, 1)
    marker_readme = "Thinking-budget generalization study"
    if already_applied(s, marker_readme):
        print(f"README.md: study entry already applied")
    else:
        s = splice_before(s, README_STUDY_ANCHOR, readme_payload)
    rd.write_text(s)
    print(f"README.md: applied")

    # -- 4) data/README.md --
    dr = hub / "data" / "README.md"
    if dr.exists():
        s = dr.read_text()
        marker_data = "2026-05-15-thinking-budget-generalization.csv"
        if already_applied(s, marker_data):
            print(f"data/README.md: already applied")
        else:
            blocks_data = re.findall(r"```markdown\n(.*?)```", filled, flags=re.DOTALL)
            if len(blocks_data) >= 4:
                payload = blocks_data[3].strip()
                s = s.rstrip() + "\n\n" + payload + "\n"
                dr.write_text(s)
                print(f"data/README.md: appended row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
