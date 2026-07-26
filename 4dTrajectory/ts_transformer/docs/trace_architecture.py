"""Render the REAL traced architecture of each model as an SVG dataflow graph.

The figures in the two tutorial documents are otherwise hand-drawn schematics. This script
produces the opposite: nothing is authored by hand. ``torch.fx.symbolic_trace`` records the
operations the model actually executes, ``ShapeProp`` runs a real tensor through and
annotates every edge with the tensor shape it carries, and the result is emitted as
graphviz DOT and rendered with the system ``dot`` binary.

So the boxes are the model's own modules, the arrows are its own dataflow (including the
residual adds, which a module-level hook trace cannot see), and the shapes are measured
rather than derived on paper.

Usage (from the repo root, with the thesis env active)::

    source scripts/activate_aeroviz_env.sh && aeroviz_activate_env
    export PYTHONPATH=$PWD:$PWD/4dTrajectory/ts_transformer
    python 4dTrajectory/ts_transformer/docs/trace_architecture.py

Writes ``arch_<model>_normalized_time.svg`` next to this file, plus a layer table on stdout.

**Shape-bookkeeping nodes.** A traced graph contains nodes that only read a tensor's shape
(``getattr(x, 'shape')`` and the ``getitem``s that unpack it) so a later ``reshape`` can be
built. They carry no activations. By default they are folded away and the count is REPORTED
-- never silently dropped; pass ``--keep-shape-ops`` to draw them.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from config import TSConfig
from models import build_model, parameter_count

HERE = Path(__file__).resolve().parent

# Sentinel colours swapped for CSS custom properties after rendering, so the embedded SVG
# follows the document's light/dark theme instead of baking graphviz's palette.
PALETTE = {
    "#ff0001": "var(--panel)",        # node fill, generic
    "#ff0002": "var(--teal-soft)",    # node fill, nn.Module
    "#ff0003": "var(--indigo-soft)",  # node fill, input/output
    "#ff0004": "var(--ink)",          # text
    "#ff0005": "var(--line-strong)",  # borders / edges
    "#ff0006": "var(--muted)",        # edge labels
    "#ff0007": "var(--teal)",         # module borders
    "#ff0008": "var(--indigo)",       # io borders
    "#ff0009": "var(--panel-strong)", # cluster fill
}

# Concrete light-theme equivalents, for the standalone files that page CSS cannot reach.
STANDALONE = {
    "#ff0001": "#ffffff", "#ff0002": "#e2f3f4", "#ff0003": "#eef2ff",
    "#ff0004": "#16212f", "#ff0005": "#a9bac7", "#ff0006": "#5c6b7e",
    "#ff0007": "#007c89", "#ff0008": "#3d58a8", "#ff0009": "#f2f6f8",
}

SHAPE_OPS = {"getattr", "getitem"}


def _fmt_shape(node) -> str:
    meta = node.meta.get("tensor_meta")
    if meta is None or not hasattr(meta, "shape"):
        return ""
    return "×".join(str(int(d)) for d in meta.shape)


def _label(node, module_of) -> tuple[str, str]:
    """(display label, kind) for a traced node."""
    if node.op == "placeholder":
        return "input", "io"
    if node.op == "output":
        return "output", "io"
    if node.op == "call_module":
        mod = module_of[node.target]
        cls = type(mod).__name__
        extra = ""
        if isinstance(mod, torch.nn.Linear):
            extra = f"{mod.in_features} → {mod.out_features}"
        elif isinstance(mod, torch.nn.Conv1d):
            extra = f"{mod.in_channels} → {mod.out_channels}, k={mod.kernel_size[0]}"
        elif isinstance(mod, (torch.nn.LayerNorm,)):
            extra = f"{tuple(mod.normalized_shape)[0]}"
        elif isinstance(mod, torch.nn.BatchNorm1d):
            extra = f"{mod.num_features}"
        elif isinstance(mod, torch.nn.Dropout):
            extra = f"p={mod.p}"
        n_par = sum(p.numel() for p in mod.parameters(recurse=True))
        head = node.target.rsplit(".", 1)[-1]
        lines = [f"{head}", cls]
        if extra:
            lines.append(extra)
        if n_par:
            lines.append(f"{n_par:,} params")
        return "\\n".join(lines), "module"
    name = node.target if isinstance(node.target, str) else getattr(node.target, "__name__", str(node.target))
    return str(name), "op"


def _truncate_scope(scope: str) -> str:
    """Cut the module path just after its first numeric component.

    ``inner.encoder.attn_layers.0.attention``          -> ``inner.encoder.attn_layers.0``
    ``inner.model.backbone.encoder.layers.0.self_attn`` -> ``inner.model.backbone.encoder.layers.0``

    That numeric component is the repeated-layer index, so this makes one cluster per
    encoder layer instead of one per sub-block. Nesting clusters three deep is what made
    graphviz fan the graph out diagonally over 5 000 px.
    """
    parts = scope.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            return ".".join(parts[: i + 1])
    return scope


def _scope(node, module_of, prev_scope: str) -> str:
    if node.op == "call_module":
        parent = node.target.rsplit(".", 1)[0] if "." in node.target else ""
        return _truncate_scope(parent)
    return prev_scope


def _cluster_label(scope: str) -> str:
    s = scope.replace("inner.", "").replace("model.", "")
    return s or "top level"


def _filter_shape_ops(nodes) -> tuple[list, int]:
    """Drop nodes that only read a tensor's ``.shape``; return (kept, folded_count)."""
    kept, folded = [], 0
    for n in nodes:
        nm = n.target if isinstance(n.target, str) else getattr(n.target, "__name__", "")
        if (
            n.op in {"call_function", "call_method"}
            and str(nm) in SHAPE_OPS
            and not _fmt_shape(n)
        ):
            folded += 1
        else:
            kept.append(n)
    return kept, folded


def build_collapsed_dot(model, example, title: str) -> str:
    """One node per traced scope: the block diagram, still derived from the real trace.

    Same graph as :func:`build_dot`, with every node belonging to a module scope merged
    into a single box carrying that scope's aggregated parameter count and the distinct
    module classes it executed. Edges are the tensor flows that actually cross a scope
    boundary, labelled with the propagated shape.
    """
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(example)
    module_of = dict(gm.named_modules())
    # scopes must be assigned over the FULL node list (execution order carries the scope
    # forward), then the shape-bookkeeping nodes dropped.
    scopes, prev = {}, ""
    for n in gm.graph.nodes:
        prev = _scope(n, module_of, prev)
        scopes[n.name] = prev
    nodes, _ = _filter_shape_ops(list(gm.graph.nodes))
    kept_names = {n.name for n in nodes}

    # group key: the scope, or the node itself when it sits at top level
    def key(n):
        return scopes[n.name] or f"@{n.name}"

    order, members = [], {}
    for n in nodes:
        k = key(n)
        if k not in members:
            members[k] = []
            order.append(k)
        members[k].append(n)

    out = [
        "digraph arch {",
        '  graph [rankdir=TB, splines=true, nodesep=0.32, ranksep=0.34, bgcolor="transparent",'
        f' fontname="Helvetica", label="{title}", labelloc=t, fontsize=15, fontcolor="#ff0004"];',
        '  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,'
        ' color="#ff0007", fontcolor="#ff0004", fillcolor="#ff0002", penwidth=1.4, margin="0.20,0.11"];',
        '  edge  [color="#ff0005", fontname="Helvetica", fontsize=9.5, fontcolor="#ff0006", arrowsize=0.8];',
    ]

    for k in order:
        ms = members[k]
        if k.startswith("@"):
            n = ms[0]
            lab, kind = _label(n, module_of)
            fill = {"module": "#ff0002", "io": "#ff0003", "op": "#ff0001"}[kind]
            border = {"module": "#ff0007", "io": "#ff0008", "op": "#ff0005"}[kind]
            style = '"rounded,filled"' if kind != "op" else '"rounded,filled,dashed"'
            out.append(f'  "{k}" [label="{lab}", fillcolor="{fill}", color="{border}", style={style}];')
        else:
            mods = [module_of[n.target] for n in ms if n.op == "call_module"]
            n_par = sum(
                sum(p.numel() for p in m.parameters(recurse=False)) for m in mods
            )
            classes, seen = [], set()
            for m in mods:
                c = type(m).__name__
                if c not in seen and c != "Dropout":
                    seen.add(c)
                    classes.append(c)
            # When a scope executed exactly one distinct submodule, name it: the bare scope
            # path ("inner") is the adapter's attribute name and says nothing to a reader,
            # whereas "projector" is what that box actually is.
            leaves = {n.target.rsplit(".", 1)[-1] for n in ms if n.op == "call_module"}
            head = _cluster_label(k)
            if len(leaves) == 1:
                head = f"{head}.{leaves.pop()}".removeprefix("inner.")
            lines = [head]
            if classes:
                lines.append(" · ".join(classes[:4]))
            lines.append(f"{len(ms)} ops · {n_par:,} params")
            out.append(f'  "{k}" [label="' + "\\n".join(lines) + '"];')

    def surviving_users(n, bridged=False, seen=None):
        """Users of n, hopping over dropped shape-bookkeeping nodes.

        ``bridged`` marks a path that went THROUGH a folded node: those carry a shape
        scalar, not a tensor, so they must not be labelled with a tensor shape.
        """
        seen = seen or set()
        for u in n.users:
            if u.name in kept_names:
                yield u, bridged
            elif u.name not in seen:
                seen.add(u.name)
                yield from surviving_users(u, True, seen)

    # Dedup on (src, dst, SHAPE), not (src, dst): more than one tensor can cross the same
    # scope boundary. PatchTST's encoder layers pass both the hidden state AND the
    # pre-softmax attention scores forward (res_attention), and collapsing those onto one
    # edge would hide the residual-attention path entirely.
    drawn = set()
    for n in nodes:
        for user, bridged in surviving_users(n):
            a, b = key(n), key(user)
            shape = _fmt_shape(n)
            tag = (a, b, "shape-only" if bridged else shape)
            if a == b or tag in drawn:
                continue
            drawn.add(tag)
            if bridged:
                out.append(f'  "{a}" -> "{b}" [style=dashed, constraint=false, '
                           f'label="shape only", fontsize=8];')
            else:
                out.append(f'  "{a}" -> "{b}" [label="{shape}"];')
    out.append("}")
    return "\n".join(out)


def build_dot(model, example, title: str, keep_shape_ops: bool) -> tuple[str, int]:
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(example)
    module_of = dict(gm.named_modules())

    nodes = list(gm.graph.nodes)
    folded = 0
    if not keep_shape_ops:
        keep = []
        for n in nodes:
            nm = n.target if isinstance(n.target, str) else getattr(n.target, "__name__", "")
            is_shape_op = (
                n.op in {"call_function", "call_method"}
                and str(nm) in SHAPE_OPS
                and not _fmt_shape(n)
            )
            if is_shape_op:
                folded += 1
            else:
                keep.append(n)
        nodes = keep
    kept = {id(n) for n in nodes}

    # scope per node, inherited in execution order
    scopes, prev = {}, ""
    for n in nodes:
        prev = _scope(n, module_of, prev)
        scopes[n.name] = prev

    by_scope: dict[str, list] = {}
    for n in nodes:
        by_scope.setdefault(scopes[n.name], []).append(n)

    out = [
        "digraph arch {",
        # NOT splines=ortho: graphviz silently drops edge labels under ortho routing, and the
        # edge labels are the propagated tensor shapes -- the point of the figure.
        '  graph [rankdir=TB, splines=true, nodesep=0.30, ranksep=0.46, bgcolor="transparent",'
        f' fontname="Helvetica", label="{title}", labelloc=t, fontsize=15, fontcolor="#ff0004"];',
        '  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10,'
        ' color="#ff0005", fontcolor="#ff0004", fillcolor="#ff0001", penwidth=1.2, margin="0.14,0.08"];',
        '  edge  [color="#ff0005", fontname="Helvetica", fontsize=9, fontcolor="#ff0006", arrowsize=0.7];',
    ]

    for ci, (scope, members) in enumerate(by_scope.items()):
        indent = "  "
        if scope:
            out.append(f'  subgraph cluster_{ci} {{')
            out.append(f'    label="{_cluster_label(scope)}"; fontsize=10; fontcolor="#ff0006";')
            out.append('    style="rounded,filled"; fillcolor="#ff0009"; color="#ff0005"; penwidth=1;')
            indent = "    "
        for n in members:
            lab, kind = _label(n, module_of)
            fill = {"module": "#ff0002", "io": "#ff0003", "op": "#ff0001"}[kind]
            border = {"module": "#ff0007", "io": "#ff0008", "op": "#ff0005"}[kind]
            shape = "box" if kind != "op" else "box"
            style = '"rounded,filled"' if kind != "op" else '"rounded,filled,dashed"'
            out.append(
                f'{indent}"{n.name}" [label="{lab}", fillcolor="{fill}", color="{border}",'
                f' shape={shape}, style={style}];'
            )
        if scope:
            out.append("  }")

    for n in nodes:
        for user in n.users:
            if id(user) in kept:
                out.append(f'  "{n.name}" -> "{user.name}" [label="{_fmt_shape(n)}"];')
            else:  # edge through a folded shape-op: reconnect to its surviving users
                for u2 in user.users:
                    if id(u2) in kept:
                        out.append(f'  "{n.name}" -> "{u2.name}" [style=dotted, label=""];')
    out.append("}")
    return "\n".join(out), folded


def render(dot_src: str, svg_path: Path, *, inline: bool) -> None:
    """Render DOT to SVG.

    ``inline=True``  -> colours become CSS custom properties and the XML prologue is
                        stripped, so the file can be pasted into the tutorial HTML and
                        follow its light/dark theme.
    ``inline=False`` -> concrete light-theme colours, so the file stands alone in a browser
                        or an ``<img>`` (where page CSS cannot reach it).
    """
    if not shutil.which("dot"):
        sys.exit("graphviz 'dot' not found on PATH — install graphviz to render the SVG")
    svg = subprocess.run(
        ["dot", "-Tsvg"], input=dot_src, capture_output=True, text=True, check=True
    ).stdout
    palette = PALETTE if inline else STANDALONE
    for sentinel, colour in palette.items():
        svg = svg.replace(sentinel, colour)
    if inline:
        # drop the XML prologue/doctype so the SVG can be embedded in an HTML document
        svg = re.sub(r"^.*?(?=<svg)", "", svg, flags=re.S)
        svg = svg.replace("<svg ", '<svg class="arch-graph" ', 1)
    svg_path.write_text(svg)


def layer_table(model, example) -> str:
    """A torchinfo-style table, built from real forward hooks (no extra dependency)."""
    rows, handles = [], []

    def hook(path):
        def fn(mod, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            shape = tuple(out.shape) if torch.is_tensor(out) else "-"
            own = sum(p.numel() for p in mod.parameters(recurse=False))
            rows.append((path, type(mod).__name__, str(shape), own))
        return fn

    for name, mod in model.named_modules():
        if name:
            handles.append(mod.register_forward_hook(hook(name)))
    with torch.no_grad():
        model(example)
    for h in handles:
        h.remove()

    w = max(len(r[0]) for r in rows)
    lines = [f"{'module':<{w}}  {'class':<22} {'output shape':<22} {'own params':>11}",
             "-" * (w + 60)]
    for path, cls, shape, own in rows:
        lines.append(f"{path:<{w}}  {cls:<22} {shape:<22} {own:>11,}")
    return "\n".join(lines)


DOC_OF = {
    "itransformer": "itransformer_tutorial.en.html",
    "patchtst": "patchtst_tutorial.en.html",
}


def embed(name: str, marker: str, svg_path: Path) -> str:
    """Splice an inline SVG between the ``<marker>:<name>:start/end`` comments in its tutorial.

    Keeping this a scripted step rather than a paste means regenerating the figure and
    updating the document are one command, so the two cannot drift apart.
    """
    doc = HERE / DOC_OF[name]
    html = doc.read_text()
    start, end = f"<!-- {marker}:{name}:start -->", f"<!-- {marker}:{name}:end -->"
    if start not in html or end not in html:
        return f"{marker} markers missing"
    head, rest = html.split(start, 1)
    _, tail = rest.split(end, 1)
    doc.write_text(head + start + "\n" + svg_path.read_text().rstrip() + "\n" + end + tail)
    return f"{marker} ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-segments", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--keep-shape-ops", action="store_true",
                    help="draw the shape-bookkeeping nodes instead of folding them")
    ap.add_argument("--table", action="store_true", help="also print the per-layer table")
    ap.add_argument("--embed", action="store_true",
                    help="splice the overview SVGs into the two tutorial HTML documents")
    args = ap.parse_args()

    for name in ("itransformer", "patchtst"):
        cfg = TSConfig(model=name, n_segments=args.n_segments)
        model = build_model(cfg).eval()
        example = torch.randn(args.batch, cfg.seq_len, cfg.enc_in)
        title = (f"{name}  ·  normalized time  ·  L={cfg.seq_len} N={cfg.n_segments} "
                 f"C={cfg.enc_in}  ·  {parameter_count(model):,} trainable parameters")

        dot_src, folded = build_dot(model, example, title, args.keep_shape_ops)
        # The full graph ships twice: a standalone file (concrete colours, for opening in a
        # browser tab where page CSS cannot reach it) and an inline copy (CSS variables, so
        # the embedded version follows the document's theme).
        out = HERE / f"arch_{name}_normalized_time.svg"
        render(dot_src, out, inline=False)
        full_inline = HERE / f"arch_{name}_normalized_time_full_inline.svg"
        render(dot_src, full_inline, inline=True)

        overview_dot = build_collapsed_dot(model, example, title)
        overview = HERE / f"arch_{name}_normalized_time_overview.svg"
        render(overview_dot, overview, inline=True)

        if args.embed:
            msg = f"{embed(name, 'ARCH', overview)}, {embed(name, 'ARCHFULL', full_inline)}"
        else:
            msg = "not embedded (--embed to splice)"
        print(f"{name:13s} -> {out.name}, {overview.name}, {full_inline.name}  [{msg}]")
        if folded:
            print(f"{'':13s}    folded {folded} shape-bookkeeping node(s) "
                  f"(getattr/getitem on .shape); rerun with --keep-shape-ops to draw them")
        if args.table:
            print()
            print(layer_table(model, example))
            print()


if __name__ == "__main__":
    main()
