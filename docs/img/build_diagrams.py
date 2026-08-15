"""Generate the README architecture diagrams as themed SVG pairs.

GitHub renders an SVG referenced from Markdown in its own context, so
`currentColor` resolves to the file's own default rather than the page
foreground: a single theme-adaptive file would be black-on-black in dark mode.
Each diagram is therefore emitted twice with an explicit palette, and the
README selects between them with `<picture>` + `prefers-color-scheme`.

Palettes are GitHub Primer values so the figures read as native on both grounds.

    python docs/img/build_diagrams.py
"""

from __future__ import annotations

from pathlib import Path


PALETTES = {
    "light": {
        "fg": "#1f2328",
        "muted": "#59636e",
        "faint": "#818b98",
        "line": "#d1d9e0",
        "surface": "#f6f8fa",
        "surface_alt": "#ffffff",
        "accent": "#0969da",
        "accent_soft": "#ddf4ff",
        "agentic": "#8250df",
        "agentic_soft": "#fbefff",
        "danger": "#cf222e",
        "ok": "#1a7f37",
        "band": "#eff2f5",
    },
    "dark": {
        "fg": "#f0f6fc",
        "muted": "#9198a1",
        "faint": "#7d8590",
        "line": "#3d444d",
        "surface": "#151b23",
        "surface_alt": "#0d1117",
        "accent": "#4493f8",
        "accent_soft": "#121d2f",
        "agentic": "#ab7df8",
        "agentic_soft": "#1d1b2e",
        "danger": "#ff7b72",
        "ok": "#3fb950",
        "band": "#12171f",
    },
}

FONT = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', "
    "Helvetica, Arial, sans-serif"
)
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"


def _text(x, y, s, *, fill, size=13, anchor="middle", weight="400", font=FONT):
    return (
        f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>'
    )


def _box(x, y, w, h, *, fill, stroke, rx=8, width=1.5):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def _defs(p):
    """Arrowheads. Ids are file-scoped, so the same names are safe per theme."""
    heads = [("arrow", p["muted"]), ("arrow-accent", p["accent"]),
             ("arrow-agentic", p["agentic"]), ("arrow-danger", p["danger"])]
    markers = "".join(
        f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
        for name, color in heads
    )
    return f"<defs>{markers}</defs>"


def _svg(w, h, label, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" aria-label="{label}">'
        f"{body}</svg>\n"
    )


# --------------------------------------------------------------------------
# Figure 1 — where deterministic control ends and model agency begins
# --------------------------------------------------------------------------

def agent_graph(p: str) -> str:
    c = PALETTES[p]
    W, H = 960, 380
    NODE_W, NODE_H, ROW_Y = 150, 54, 150
    xs = {"safety": 52, "summarize": 232, "select": 412, "chat": 640}
    tool_y = 286

    out = [_defs(c)]

    # Bands carry the claim: three harness nodes run unconditionally; only the
    # chat/tool loop is decided by the model.
    out.append(_box(28, 104, 556, 148, fill=c["band"], stroke="none", rx=12))
    out.append(_box(616, 104, 198, 246, fill=c["agentic_soft"],
                    stroke=c["agentic"], rx=12, width=1))
    out.append(_text(306, 126, "Deterministic — the harness runs these every eligible turn",
                     fill=c["muted"], size=12, weight="600"))
    out.append(_text(715, 126, "Model-directed", fill=c["agentic"],
                     size=12, weight="600"))

    # START / END terminals
    out.append(f'<circle cx="20" cy="{ROW_Y + NODE_H / 2}" r="7" fill="{c["fg"]}"/>')
    out.append(f'<circle cx="900" cy="{ROW_Y + NODE_H / 2}" r="7" fill="none" '
               f'stroke="{c["fg"]}" stroke-width="2.5"/>')
    out.append(f'<circle cx="900" cy="{ROW_Y + NODE_H / 2}" r="3" fill="{c["fg"]}"/>')

    nodes = [
        ("safety", "safety_check", "regex crisis detector"),
        ("summarize", "summarize", "token-based compaction"),
        ("select", "select_memory", "user-scoped Store read"),
    ]
    for key, title, sub in nodes:
        x = xs[key]
        out.append(_box(x, ROW_Y, NODE_W, NODE_H, fill=c["surface_alt"],
                        stroke=c["accent"]))
        out.append(_text(x + NODE_W / 2, ROW_Y + 23, title, fill=c["fg"],
                         size=13, weight="600", font=MONO))
        out.append(_text(x + NODE_W / 2, ROW_Y + 40, sub, fill=c["faint"], size=10.5))

    # chat + use_tool
    out.append(_box(xs["chat"], ROW_Y, NODE_W, NODE_H, fill=c["surface_alt"],
                    stroke=c["agentic"]))
    out.append(_text(xs["chat"] + NODE_W / 2, ROW_Y + 32, "chat", fill=c["fg"],
                     size=14, weight="600", font=MONO))
    out.append(_box(xs["chat"], tool_y, NODE_W, NODE_H, fill=c["surface_alt"],
                    stroke=c["agentic"]))
    out.append(_text(xs["chat"] + NODE_W / 2, tool_y + 23, "use_tool", fill=c["fg"],
                     size=13, weight="600", font=MONO))
    out.append(_text(xs["chat"] + NODE_W / 2, tool_y + 40, "RAG retrieval",
                     fill=c["faint"], size=10.5))

    mid = ROW_Y + NODE_H / 2

    def arrow(x1, y1, x2, y2, *, color, head="arrow", dash=None, width=1.8):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                f'stroke-width="{width}"{d} marker-end="url(#{head})"/>')

    out.append(arrow(29, mid, xs["safety"] - 6, mid, color=c["muted"]))
    out.append(arrow(xs["safety"] + NODE_W, mid, xs["summarize"] - 6, mid,
                     color=c["muted"]))
    out.append(arrow(xs["summarize"] + NODE_W, mid, xs["select"] - 6, mid,
                     color=c["muted"]))
    out.append(arrow(xs["select"] + NODE_W, mid, xs["chat"] - 6, mid,
                     color=c["accent"], head="arrow-accent"))
    out.append(_text(601, mid - 20, "memory block", fill=c["accent"], size=11))
    out.append(_text(601, mid - 8, "(untrusted)", fill=c["faint"], size=10))
    out.append(arrow(xs["chat"] + NODE_W, mid, 890, mid, color=c["muted"]))

    # The bounded loop — the one place the model chooses
    lx, rx_ = xs["chat"] + 42, xs["chat"] + 108
    out.append(arrow(lx, ROW_Y + NODE_H, lx, tool_y - 6, color=c["agentic"],
                     head="arrow-agentic"))
    out.append(arrow(rx_, tool_y, rx_, ROW_Y + NODE_H + 6, color=c["agentic"],
                     head="arrow-agentic"))
    out.append(_text(xs["chat"] + NODE_W / 2, 243, "model decides", fill=c["agentic"],
                     size=11, weight="600"))
    out.append(_text(xs["chat"] + NODE_W / 2, 258, "bounded loop", fill=c["faint"],
                     size=10))

    # Crisis is a flag set on the state, not a separate edge: drawing it as an
    # arrow to another node would imply a branch the graph does not have.
    sx = xs["safety"] + NODE_W / 2
    out.append(f'<line x1="{sx}" y1="{ROW_Y + NODE_H}" x2="{sx}" y2="282" '
               f'stroke="{c["danger"]}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    out.append(_box(52, 282, 532, 56, fill=c["surface"], stroke=c["danger"], rx=8,
                    width=1.2))
    out.append(_text(318, 303, "crisis detected — a flag on the state, not a branch",
                     fill=c["danger"], size=11.5, weight="600"))
    out.append(_text(318, 322, "select_memory reads nothing · chat switches to the "
                     "crisis prompt · a 988 block is appended deterministically",
                     fill=c["muted"], size=10.5))

    out.append(_text(480, 34, "MindBridge agent graph", fill=c["fg"],
                     size=17, weight="700"))
    out.append(_text(480, 58, "The harness decides what runs on every turn; the model's "
                     "only choice is whether to retrieve.",
                     fill=c["muted"], size=12))
    out.append(_text(480, 78, "Memory recall is never an optional tool call.",
                     fill=c["faint"], size=11))

    label = ("MindBridge agent graph: safety_check, summarize and select_memory run "
             "deterministically on every eligible turn, and only the chat to use_tool "
             "loop is decided by the model.")
    return _svg(W, H, label, "".join(out))


# --------------------------------------------------------------------------
# Figure 2 — the row lock covers the claim; the lease covers the work
# --------------------------------------------------------------------------

def memory_write(p: str) -> str:
    c = PALETTES[p]
    W, H = 980, 690
    lanes = [
        (118, "Chat request", "FastAPI"),
        (330, "PostgreSQL", "relational + outbox"),
        (545, "Memory worker", "leased, out of band"),
        (730, "LLM provider", "BYOK"),
        (890, "Store", "LangGraph"),
    ]
    top, bottom = 96, 636

    out = [_defs(c)]

    for x, title, sub in lanes:
        out.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" '
                   f'stroke="{c["line"]}" stroke-width="1.5" stroke-dasharray="4 4"/>')
        out.append(_box(x - 70, 58, 140, 38, fill=c["surface"], stroke=c["line"], rx=8))
        out.append(_text(x, 75, title, fill=c["fg"], size=12.5, weight="600"))
        out.append(_text(x, 89, sub, fill=c["faint"], size=10))

    def msg(y, x1, x2, text, *, color=None, note=None, dash=None):
        color = color or c["muted"]
        head = "arrow-accent" if color == c["accent"] else "arrow"
        d = f' stroke-dasharray="{dash}"' if dash else ""
        seg = (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" '
               f'stroke-width="1.8"{d} marker-end="url(#{head})"/>')
        mid_x = (x1 + x2) / 2
        parts = [seg, _text(mid_x, y - 8, text, fill=c["fg"], size=11.5, font=MONO)]
        if note:
            parts.append(_text(mid_x, y + 15, note, fill=c["faint"], size=10))
        return "".join(parts)

    # 1. The outbox write shares the assistant transaction.
    out.append(_box(206, 122, 248, 46, fill=c["accent_soft"], stroke=c["accent"], rx=8))
    out.append(msg(145, 118 + 6, 206 - 4, "reply", color=c["accent"]))
    out.append(_text(330, 140, "assistant row + memory_revision++",
                     fill=c["fg"], size=11, weight="600"))
    out.append(_text(330, 156, "+ outbox job — one transaction", fill=c["accent"],
                     size=10.5, weight="600"))
    out.append(_text(330, 182, "if this commits, the job exists", fill=c["faint"],
                     size=10))

    # 2. Claim: short transaction, row lock only.
    out.append(msg(238, 545 - 6, 330 + 6, "claim_next()",
                   note="FOR UPDATE SKIP LOCKED"))
    out.append(msg(288, 330 + 6, 545 - 6, "job + lease_until = now + 90s"))
    out.append(_text(437, 312, "commit — row lock released here", fill=c["ok"],
                     size=10.5, weight="600"))

    # 3. Second transaction: re-lock, re-check, then leave the database.
    out.append(msg(352, 545 - 6, 330 + 6, "re-lock User → Conversation → Job",
                   note="lease token must still match"))
    out.append(msg(402, 545 - 6, 330 + 6, "re-check consent, epoch, revision, crisis"))
    out.append(msg(452, 545 + 6, 730 - 6, "structured extraction",
                   note="30s cap, temperature 0"))
    out.append(msg(492, 730 - 6, 545 + 6, "schema-bound draft", dash="5 4"))
    out.append(msg(532, 545 - 6, 330 + 6, "verify every quote against stored messages"))
    out.append(msg(572, 545 + 6, 890 - 6, "aput — idempotent on target_revision",
                   color=c["accent"]))
    out.append(msg(614, 545 - 6, 330 + 6, "mark succeeded, commit"))

    # The two spans that make the design work.
    out.append(f'<path d="M 36 232 h -10 v 64 h 10" fill="none" stroke="{c["ok"]}" '
               f'stroke-width="2"/>')
    out.append(_text(30, 258, "row", fill=c["ok"], size=10.5, anchor="end", weight="600"))
    out.append(_text(30, 272, "lock", fill=c["ok"], size=10.5, anchor="end", weight="600"))

    out.append(f'<path d="M 72 232 h -10 v 392 h 10" fill="none" stroke="{c["agentic"]}" '
               f'stroke-width="2"/>')
    out.append(_text(66, 420, "lease", fill=c["agentic"], size=10.5, anchor="end",
                     weight="600"))
    out.append(_text(66, 434, "90s", fill=c["agentic"], size=10.5, anchor="end",
                     weight="600"))

    out.append(_text(490, 30, "How one turn becomes durable memory", fill=c["fg"],
                     size=17, weight="700"))
    out.append(_text(490, 48, "The row lock covers only the claim. The lease covers the "
                     "work — including the call that leaves the database.",
                     fill=c["muted"], size=12))

    out.append(_text(490, 666, "A crashed worker lets the lease lapse and another worker "
                     "takes over; its stale token can no longer complete the job.",
                     fill=c["faint"], size=11))

    label = ("Memory write path: the assistant reply and its outbox job commit in one "
             "transaction; a worker claims the job with FOR UPDATE SKIP LOCKED, holds a "
             "90 second lease across the LLM call, and writes an idempotent Store value.")
    return _svg(W, H, label, "".join(out))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    figures = {"agent-graph": agent_graph, "memory-write": memory_write}
    for name, build in figures.items():
        for theme in PALETTES:
            suffix = "" if theme == "light" else "-dark"
            path = out_dir / f"{name}{suffix}.svg"
            path.write_text(build(theme), encoding="utf-8")
            print(f"wrote {path.relative_to(out_dir.parent.parent)}")


if __name__ == "__main__":
    main()
