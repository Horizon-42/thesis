"""Draw the two-tier model architecture figure (SVG -> PDF) for the thesis.

Summary figure of 4dTrajectory/ts_transformer/docs/two_tier/two_tier_framework.zh.md:
the prior (tier 1) speaks instruction words, the executor (tier 2) flies them, and the words
are the only interface. Offline: observed tracks -> labeller -> sentences -> prior training.

    python docs/architecture/make_two_tier_architecture.py            # en + zh, svg + pdf
    python docs/architecture/make_two_tier_architecture.py --lang en

Only the stdlib plus `rsvg-convert` (librsvg) for the PDF.
"""
import argparse
import subprocess
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
W, H = 800, 535        # the content starts at y = 30 (viewBox offset below)
TOP = 30

PALETTE = {
    "ink": "#222222",
    "mute": "#555555",
    "plane": "#f4f5f6",
    "plane_line": "#b8bec3",
    "data": ("#eceff1", "#5f6b73"),
    "prior": ("#d9e7f7", "#2f5f98"),
    "exec": ("#fbe6cc", "#b4661a"),
    "mask": ("#fff3c4", "#9c8200"),
    "state": ("#e1f0de", "#3f7a3a"),
    "train": ("#eef3fa", "#2f5f98"),
}

FONTS = {
    "en": "Helvetica, Arial, sans-serif",
    "zh": "Heiti SC, STHeiti, Helvetica, sans-serif",
}

LABELS = {
    "en": {
        "offline": "Offline: data → language",
        "observed": ("Observed arrivals", "ADS-B approach tracks"),
        "labeller": ("Labeller", "tracks → instructions"),
        "sentences": ("Instruction sentences", "six columns per 2 s step"),
        "columns": ["Runway", "Approach", "Heading", "Altitude", "Descent angle", "Speed"],
        "loop": "Closed loop: two-tier generation",
        "context": ("Context", "candidate runways,", "landing history, traffic"),
        "prior": ("Tier 1 · Prior", "autoregressive Transformer", "speaks every 2 s"),
        "masks": ("Decode masks", "vocabulary rules", "glidepath floor", "separation"),
        "executor": ("Tier 2 · Executor", "rule-based autopilot", "point-mass dynamics", "flies every 1 s"),
        "words": ("instruction words", "the only interface"),
        "state": ("Flight state", "4D trajectory"),
        "feedback": "state feedback",
        "training": "Training of the prior",
        "stages": [
            ("① Pre-training", "teacher forcing", "on sentences"),
            ("② Post-training", "landing reward in loop,", "procedure masks"),
            ("③ Multi-aircraft", "traffic attention,", "separation (ongoing)"),
        ],
        "trains": "trains",
        "outcome": "landing outcome",
        "legend": ["Data / language", "Learned", "Rules / physics", "Decode-time check", "In progress"],
    },
    "zh": {
        "offline": "离线：数据 → 语言",
        "observed": ("观测进场航迹", "ADS-B 进近轨迹"),
        "labeller": ("标注器", "航迹 → 指令"),
        "sentences": ("指令句子", "每 2 s 一步，六列词"),
        "columns": ["跑道", "进近", "航向", "高度", "下降角", "速度"],
        "loop": "闭环：两层生成",
        "context": ("上下文", "候选跑道、", "落地情况、其他飞机"),
        "prior": ("上层 · 先验", "自回归 Transformer", "每 2 s 说一步指令"),
        "masks": ("解码屏蔽", "词表相容规则", "下滑道下沿", "间隔"),
        "executor": ("下层 · 执行器", "规则自动驾驶", "点质量动力学", "每 1 s 飞一步"),
        "words": ("指令词", "两层之间唯一的接口"),
        "state": ("飞行状态", "4D 轨迹"),
        "feedback": "状态反馈",
        "training": "先验的训练",
        "stages": [
            ("① 预训练", "teacher forcing", "在句子产物上"),
            ("② 后训练", "闭环落地奖励、", "程序屏蔽"),
            ("③ 多机", "交通注意力、", "间隔（进行中）"),
        ],
        "trains": "训练",
        "outcome": "落地结果",
        "legend": ["数据 / 语言", "学习得到", "规则 / 物理", "解码时检查", "进行中"],
    },
}


class Svg:
    def __init__(self, font):
        self.font = font
        self.parts = []

    def rect(self, x, y, w, h, fill, stroke, dash=False, r=8, sw=1.4):
        dash_attr = ' stroke-dasharray="6 4"' if dash else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>')

    def text(self, x, y, s, size=12, bold=False, anchor="middle", fill=None, italic=False):
        weight = ' font-weight="bold"' if bold else ""
        style = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill or PALETTE["ink"]}"{weight}{style}>{s}</text>')

    def box(self, x, y, w, h, colours, lines, title_size=13.5, size=12, dash=False, line_h=17):
        """A box whose first line is the bold title, the rest plain, vertically centred."""
        fill, stroke = colours
        self.rect(x, y, w, h, fill, stroke, dash=dash)
        top = y + (h - line_h * (len(lines) - 1)) / 2 + 4
        for i, s in enumerate(lines):
            self.text(x + w / 2, top + i * line_h, s,
                      size=title_size if i == 0 else size, bold=i == 0,
                      fill=None if i == 0 else PALETTE["mute"])

    def arrow(self, points, dash=False, colour=None, sw=1.8):
        colour = colour or PALETTE["ink"]
        marker = "arrow-dash" if dash else "arrow"
        dash_attr = ' stroke-dasharray="5 4"' if dash else ""
        pts = " ".join(f"{px},{py}" for px, py in points)
        self.parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="{sw}"{dash_attr} '
            f'marker-end="url(#{marker})" stroke-linejoin="round"/>')

    def render(self):
        defs = ""
        for name, colour in (("arrow", PALETTE["ink"]), ("arrow-dash", PALETTE["mute"])):
            defs += (f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                     f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
                     f'fill="{colour}"/></marker>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
                f'viewBox="0 {TOP} {W} {H}" font-family="{self.font}">'
                f'<defs>{defs}</defs><rect y="{TOP}" width="{W}" height="{H}" fill="white"/>'
                + "".join(self.parts) + "</svg>")


def draw(lang):
    t = LABELS[lang]
    g = Svg(FONTS[lang])
    plane, plane_line = PALETTE["plane"], PALETTE["plane_line"]

    # ---- planes ----------------------------------------------------------------------------
    g.rect(5, 40, 180, 480, plane, plane_line)
    g.text(15, 62, t["offline"], size=12.5, bold=True, anchor="start", fill=PALETTE["mute"])
    g.rect(200, 40, 585, 320, plane, plane_line)
    g.text(215, 62, t["loop"], size=12.5, bold=True, anchor="start", fill=PALETTE["mute"])
    g.rect(200, 395, 585, 125, plane, plane_line)
    g.text(215, 415, t["training"], size=12.5, bold=True, anchor="start", fill=PALETTE["mute"])

    # ---- offline column: tracks -> labeller -> sentences ------------------------------------
    g.box(15, 78, 160, 58, PALETTE["data"], list(t["observed"]))
    g.box(15, 168, 160, 58, PALETTE["data"], list(t["labeller"]))
    g.rect(15, 262, 160, 238, *PALETTE["data"])
    g.text(95, 283, t["sentences"][0], size=13, bold=True)
    g.text(95, 300, t["sentences"][1], size=11.5, fill=PALETTE["mute"])
    for i, name in enumerate(t["columns"]):
        y = 314 + i * 28
        g.rect(30, y, 130, 24, "white", PALETTE["data"][1], r=5, sw=1)
        g.text(95, y + 16.5, name, size=12)
    g.arrow([(95, 136), (95, 168)])
    g.arrow([(95, 226), (95, 262)])

    # ---- closed loop ------------------------------------------------------------------------
    g.box(215, 74, 170, 56, PALETTE["data"], list(t["context"]), title_size=12.5, size=11.5, line_h=15)
    g.box(215, 152, 170, 104, PALETTE["prior"], list(t["prior"]))
    g.box(430, 160, 120, 88, PALETTE["mask"], list(t["masks"]), title_size=12.5, size=11.5, line_h=16)
    g.box(595, 152, 170, 104, PALETTE["exec"], list(t["executor"]), line_h=18)
    g.box(550, 292, 215, 54, PALETTE["state"], list(t["state"]))

    g.arrow([(300, 130), (300, 152)])                       # context -> prior
    g.arrow([(385, 204), (430, 204)])                       # prior -> masks
    g.arrow([(550, 204), (595, 204)])                       # masks -> executor
    g.arrow([(680, 256), (680, 292)])                       # executor -> state
    g.arrow([(550, 319), (300, 319), (300, 256)])           # state -> prior (feedback)
    g.text(425, 311, t["feedback"], size=11.5, italic=True, fill=PALETTE["mute"])
    g.text(490, 128, t["words"][0], size=12, bold=True, fill=PALETTE["prior"][1])
    g.text(490, 143, t["words"][1], size=11, italic=True, fill=PALETTE["mute"])

    # ---- training of the prior --------------------------------------------------------------
    for i, lines in enumerate(t["stages"]):
        x = 215 + i * 192
        g.box(x, 430, 170, 76, PALETTE["train"], list(lines), title_size=13, size=11.5, line_h=16,
              dash=(i == 2))
    g.arrow([(385, 468), (407, 468)], sw=1.5)
    g.arrow([(577, 468), (599, 468)], sw=1.5)
    g.arrow([(175, 468), (215, 468)])                       # sentences -> pre-training
    g.arrow([(250, 430), (250, 256)], dash=True)            # training -> prior
    g.text(258, 384, t["trains"], size=11.5, italic=True, anchor="start", fill=PALETTE["mute"])
    g.arrow([(565, 346), (565, 430)], dash=True)            # flight state -> post-training reward
    g.text(573, 384, t["outcome"], size=11.5, italic=True, anchor="start", fill=PALETTE["mute"])

    # ---- legend -----------------------------------------------------------------------------
    swatches = [PALETTE["data"], PALETTE["prior"], PALETTE["exec"], PALETTE["mask"], PALETTE["train"]]
    x = 15
    for i, (name, colours) in enumerate(zip(t["legend"], swatches)):
        g.rect(x, 538, 16, 12, colours[0], colours[1], dash=(i == 4), r=3, sw=1.2)
        g.text(x + 22, 548, name, size=11.5, anchor="start", fill=PALETTE["mute"])
        x += 22 + (6.4 if lang == "en" else 12) * len(name) + 22
    return g.render()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--lang", choices=sorted(LABELS), nargs="+", default=sorted(LABELS))
    args = parser.parse_args()
    for lang in args.lang:
        svg = OUT_DIR / f"two_tier_architecture.{lang}.svg"
        pdf = svg.with_suffix(".pdf")
        svg.write_text(draw(lang), encoding="utf-8")
        subprocess.run(["rsvg-convert", "--format=pdf", "--output", str(pdf), str(svg)], check=True)
        print(f"wrote {svg.relative_to(OUT_DIR.parent.parent)} and {pdf.name}")


if __name__ == "__main__":
    main()
