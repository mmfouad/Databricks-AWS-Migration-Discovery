"""Regenerate the notebook artifacts from the Databricks source file.

``aws_databricks_migration_discovery.py`` is the single source of truth. It is written in
Databricks source format (``# COMMAND ----------`` cell separators and ``# MAGIC %md`` markdown),
which is what you get when you export a notebook from the workspace and what the workspace accepts
on import.

This script derives the two other artifacts from it:

* ``aws_databricks_migration_discovery.ipynb`` - a Jupyter notebook, for GitHub rendering and for
  importing into workspaces that prefer ``.ipynb``.
* ``aws_databricks_migration_discovery_preview.html`` - a static, styled preview so the notebook can
  be read without a Databricks workspace or a Jupyter install.

Usage::

    python tools/build_notebook.py             # rebuild both artifacts
    python tools/build_notebook.py --check     # fail if the artifacts are stale (for CI)
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_STEM = "aws_databricks_migration_discovery"
SOURCE_PATH = REPO_ROOT / f"{NOTEBOOK_STEM}.py"
IPYNB_PATH = REPO_ROOT / f"{NOTEBOOK_STEM}.ipynb"
HTML_PATH = REPO_ROOT / f"{NOTEBOOK_STEM}_preview.html"

CELL_SEPARATOR = re.compile(r"^# COMMAND -+$", re.MULTILINE)
DATABRICKS_HEADER = "# Databricks notebook source"
MAGIC_PREFIX = "# MAGIC "
MAGIC_BARE = "# MAGIC"

NOTEBOOK_METADATA = {
    "databricks": {"notebook": {"language": "python"}},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "python"},
}


# --------------------------------------------------------------------------------------------
# Parsing the Databricks source format
# --------------------------------------------------------------------------------------------
def strip_magic(lines: List[str]) -> List[str]:
    """Remove the ``# MAGIC`` prefix Databricks uses to store non-Python cell content."""
    cleaned = []
    for line in lines:
        if line.startswith(MAGIC_PREFIX):
            cleaned.append(line[len(MAGIC_PREFIX):])
        elif line == MAGIC_BARE:
            cleaned.append("")
        else:
            cleaned.append(line)
    return cleaned


def parse_source(text: str) -> List[Dict[str, object]]:
    """Split Databricks source format into ``{"cell_type", "source"}`` dictionaries."""
    text = text.replace("\r\n", "\n")
    if text.startswith(DATABRICKS_HEADER):
        text = text[len(DATABRICKS_HEADER):].lstrip("\n")

    cells: List[Dict[str, object]] = []
    for block in CELL_SEPARATOR.split(text):
        lines = block.strip("\n").split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue

        first = lines[0].strip()
        if first in ("# MAGIC %md", "# MAGIC %md-sandbox"):
            cells.append({"cell_type": "markdown", "source": strip_magic(lines[1:])})
        elif first.startswith("# MAGIC %"):
            # Any other magic (%sql, %sh, ...) stays a code cell with the magic on the first line.
            cells.append({"cell_type": "code", "source": strip_magic(lines)})
        else:
            cells.append({"cell_type": "code", "source": lines})
    return cells


def as_ipynb_source(lines: List[str]) -> List[str]:
    """nbformat stores source as a list of lines with trailing newlines except on the last line."""
    if not lines:
        return []
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def build_ipynb(cells: List[Dict[str, object]]) -> Dict[str, object]:
    """Assemble an nbformat 4.5 notebook, with stable ids so rebuilds produce clean diffs."""
    ipynb_cells = []
    for index, cell in enumerate(cells):
        source = as_ipynb_source(list(cell["source"]))  # type: ignore[arg-type]
        base: Dict[str, object] = {
            "cell_type": cell["cell_type"],
            "id": f"cell-{index:03d}",
            "metadata": {},
            "source": source,
        }
        if cell["cell_type"] == "code":
            base["execution_count"] = None
            base["outputs"] = []
        ipynb_cells.append(base)
    return {"cells": ipynb_cells, "metadata": NOTEBOOK_METADATA, "nbformat": 4, "nbformat_minor": 5}


# --------------------------------------------------------------------------------------------
# Minimal markdown rendering for the static HTML preview
# --------------------------------------------------------------------------------------------
INLINE_CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*([^*]+)\*\*")
ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
AUTOLINK = re.compile(r"&lt;(https?://[^&]+)&gt;")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
BLOCKQUOTE = re.compile(r"^>\s?(.*)$")
LIST_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
ENTITY = re.compile(r"&amp;([a-zA-Z]+|#\d+);")


def anchor_for(title_html: str) -> str:
    """Build a stable heading anchor from already-rendered inline HTML."""
    plain = re.sub(r"<[^>]+>", "", title_html)
    plain = ENTITY.sub("", plain)
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-") or "section"


def render_inline(text: str) -> str:
    """Escape HTML, then re-apply the inline markdown the notebook actually uses."""
    out = html.escape(text, quote=False)
    out = INLINE_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    out = AUTOLINK.sub(lambda m: f'<a href="{m.group(1)}" rel="noopener">{m.group(1)}</a>', out)
    out = LINK.sub(lambda m: f'<a href="{m.group(2)}" rel="noopener">{m.group(1)}</a>', out)
    # Entities written directly in the markdown (&rarr;, &mdash;) survive escaping as &amp;rarr;.
    return ENTITY.sub(r"&\1;", out)


def render_table(rows: List[str]) -> str:
    """Render a GitHub-style pipe table."""

    def cells(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    header = cells(rows[0])
    body = [cells(row) for row in rows[2:]]
    head_html = "".join(f"<th>{render_inline(cell)}</th>" for cell in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{render_inline(cell)}</td>" for cell in row) + "</tr>" for row in body
    )
    return f"<table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>"


def render_markdown(lines: List[str]) -> str:
    """Render the markdown subset used by this notebook: headings, lists, tables, quotes, code, rules."""
    parts: List[str] = []
    paragraph: List[str] = []
    list_stack: List[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            parts.append(f"<p>{render_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_lists(to_depth: int = 0) -> None:
        while len(list_stack) > to_depth:
            parts.append(f"</{list_stack.pop()}>")

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            close_lists()
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            close_lists()
            index += 1
            block: List[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            parts.append(f'<pre class="md-code"><code>{html.escape(chr(10).join(block))}</code></pre>')
            continue

        if stripped in ("---", "***", "___"):
            flush_paragraph()
            close_lists()
            parts.append("<hr />")
            index += 1
            continue

        heading = HEADING.match(stripped)
        if heading:
            flush_paragraph()
            close_lists()
            level = len(heading.group(1))
            title = render_inline(heading.group(2))
            parts.append(f'<h{level} id="{anchor_for(title)}">{title}</h{level}>')
            index += 1
            continue

        is_table = (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and set(lines[index + 1].strip()) <= set("|-: ")
            and "-" in lines[index + 1]
        )
        if is_table:
            flush_paragraph()
            close_lists()
            table_rows: List[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_rows.append(lines[index])
                index += 1
            parts.append(render_table(table_rows))
            continue

        if BLOCKQUOTE.match(stripped):
            flush_paragraph()
            close_lists()
            quote_lines: List[str] = []
            while index < len(lines):
                match = BLOCKQUOTE.match(lines[index].strip())
                if not match:
                    break
                quote_lines.append(match.group(1))
                index += 1
            parts.append(f"<blockquote>{render_inline(' '.join(quote_lines))}</blockquote>")
            continue

        item = LIST_ITEM.match(line)
        if item:
            flush_paragraph()
            indent, marker, content = item.groups()
            depth = len(indent) // 2 + 1
            tag = "ul" if marker in ("-", "*") else "ol"
            close_lists(depth)
            while len(list_stack) < depth:
                list_stack.append(tag)
                parts.append(f"<{tag}>")
            parts.append(f"<li>{render_inline(content)}</li>")
            index += 1
            continue

        # A wrapped continuation line belongs to the list item above it.
        if list_stack and line.startswith("  ") and parts and parts[-1].startswith("<li>"):
            parts[-1] = parts[-1][: -len("</li>")] + " " + render_inline(stripped) + "</li>"
            index += 1
            continue

        close_lists()
        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    close_lists()
    return "\n".join(parts)


HTML_STYLE = """
:root {
  --bg: #f6f8fa; --panel: #ffffff; --ink: #1f2328; --muted: #59636e; --line: #d1d9e0;
  --accent: #0969da; --code-bg: #f6f8fa; --md-kicker: #8250df; --code-kicker: #1a7f37;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.65 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); }
.toc { position: sticky; top: 0; height: 100vh; overflow-y: auto; padding: 24px 18px; background: var(--panel); border-right: 1px solid var(--line); }
.toc h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 0 0 12px; }
.toc a { display: block; padding: 5px 8px; border-radius: 6px; color: var(--ink); text-decoration: none; font-size: 13px; }
.toc a:hover { background: var(--bg); color: var(--accent); }
.toc a.lvl-1 { font-weight: 700; margin-top: 14px; }
.toc a.lvl-2 { padding-left: 16px; }
.toc a.lvl-3 { padding-left: 28px; color: var(--muted); font-size: 12.5px; }
main { padding: 32px 40px 96px; max-width: 1100px; min-width: 0; }
.banner { background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--accent); border-radius: 8px; padding: 16px 20px; margin-bottom: 28px; }
.banner h1 { margin: 0 0 6px; font-size: 20px; }
.banner p { margin: 4px 0 0; color: var(--muted); font-size: 13.5px; }
.cell { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin: 0 0 16px; overflow: hidden; }
.kicker { font: 600 11px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; letter-spacing: .06em; text-transform: uppercase; padding: 8px 16px; border-bottom: 1px solid var(--line); background: var(--code-bg); color: var(--muted); }
.cell.markdown .kicker { color: var(--md-kicker); }
.cell.code .kicker { color: var(--code-kicker); }
.body { padding: 4px 20px 12px; }
pre { margin: 0; padding: 16px 20px; overflow-x: auto; background: var(--panel); font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre.md-code { background: var(--code-bg); border-radius: 6px; margin: 10px 0; }
code { background: rgba(129,139,152,.14); border-radius: 5px; padding: .15em .4em; font: 12.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre code { background: none; padding: 0; font-size: inherit; }
h1, h2, h3, h4 { line-height: 1.3; margin: 20px 0 10px; }
h1 { font-size: 26px; border-bottom: 1px solid var(--line); padding-bottom: 8px; }
h2 { font-size: 20px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 16px; }
h4 { font-size: 14px; color: var(--muted); }
table { border-collapse: collapse; margin: 12px 0; width: 100%; font-size: 13.5px; }
th, td { border: 1px solid var(--line); padding: 7px 11px; text-align: left; vertical-align: top; }
th { background: var(--code-bg); font-weight: 600; }
blockquote { margin: 12px 0; padding: 8px 16px; border-left: 4px solid var(--line); color: var(--muted); background: var(--code-bg); border-radius: 0 6px 6px 0; }
a { color: var(--accent); }
hr { border: 0; border-top: 1px solid var(--line); margin: 20px 0; }
ul, ol { padding-left: 24px; }
li { margin: 3px 0; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } .toc { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); } main { padding: 24px 18px 64px; } }
"""


def build_html(cells: List[Dict[str, object]], source_name: str) -> str:
    """Render a self-contained static preview with a table-of-contents sidebar."""
    toc: List[Tuple[int, str, str]] = []
    rendered: List[str] = []

    for index, cell in enumerate(cells, start=1):
        lines = list(cell["source"])  # type: ignore[arg-type]
        if cell["cell_type"] == "markdown":
            for line in lines:
                heading = HEADING.match(line.strip())
                if heading and len(heading.group(1)) <= 3:
                    title = render_inline(heading.group(2))
                    toc.append((len(heading.group(1)), anchor_for(title), title))
            body = f'<div class="body">{render_markdown(lines)}</div>'
            kind = "markdown"
        else:
            body = f"<pre><code>{html.escape(chr(10).join(lines))}</code></pre>"
            kind = "code"
        rendered.append(
            f'<section class="cell {kind}" id="cell-{index}">'
            f'<div class="kicker">Cell {index} &middot; {kind.capitalize()}</div>{body}</section>'
        )

    toc_html = "\n".join(f'<a class="lvl-{level}" href="#{anchor}">{title}</a>' for level, anchor, title in toc)
    code_cells = sum(1 for cell in cells if cell["cell_type"] == "code")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AWS to Azure Databricks Migration Sizing &amp; Pricing &mdash; notebook preview</title>
<style>{HTML_STYLE}</style>
</head>
<body>
<div class="layout">
<nav class="toc"><h2>Contents</h2>
{toc_html}
</nav>
<main>
<div class="banner">
<h1>AWS &rarr; Azure Databricks Migration Sizing &amp; Pricing</h1>
<p>Static preview of <code>{html.escape(source_name)}</code> &mdash; {len(cells)} cells
({code_cells} code, {len(cells) - code_cells} markdown). Generated by <code>tools/build_notebook.py</code>.</p>
<p>This is a read-only rendering. Run the notebook in Databricks to produce results.</p>
</div>
{chr(10).join(rendered)}
</main>
</div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Verify the artifacts are current instead of writing them.")
    args = parser.parse_args()

    source = SOURCE_PATH.read_text(encoding="utf-8")
    cells = parse_source(source)
    if not cells:
        print(f"error: no cells parsed from {SOURCE_PATH.name}", file=sys.stderr)
        return 1

    artifacts = {
        IPYNB_PATH: json.dumps(build_ipynb(cells), indent=1, ensure_ascii=False) + "\n",
        HTML_PATH: build_html(cells, SOURCE_PATH.name),
    }

    if args.check:
        stale = [
            path.name
            for path, text in artifacts.items()
            if not path.exists() or path.read_text(encoding="utf-8").replace("\r\n", "\n") != text
        ]
        if stale:
            print(f"error: stale artifacts: {', '.join(stale)}. Run 'python tools/build_notebook.py'.", file=sys.stderr)
            return 1
        print(f"Artifacts are up to date ({len(cells)} cells).")
        return 0

    for path, text in artifacts.items():
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path.name} ({path.stat().st_size / 1024:.0f} KB)")

    code_cells = sum(1 for cell in cells if cell["cell_type"] == "code")
    print(f"{len(cells)} cells ({code_cells} code, {len(cells) - code_cells} markdown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
