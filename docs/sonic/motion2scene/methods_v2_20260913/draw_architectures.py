"""Paper-style model schematics with exact interfaces and local reference renders."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

P = dict(
    ink="#172638",
    green="#468453",
    greenbg="#E6F0DE",
    blue="#4478A8",
    bluebg="#E7EFF9",
    gold="#B78B20",
    goldbg="#FFF2CD",
    violet="#7455A8",
    violetbg="#EEE8F7",
    red="#B55349",
    gray="#677684",
    frozen="#EAF0F2",
)
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})


def txt(ax, x, y, s, size=12, color=None, weight="normal", ha="left", va="top"):
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color or P["ink"],
        weight=weight,
        ha=ha,
        va=va,
        linespacing=1.5,
        zorder=6,
    )


def box(ax, x, y, w, h, fill="white", edge=None, dashed=False, radius=14, lw=1.4):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge or P["gray"],
        linewidth=lw,
        linestyle=(0, (5, 4)) if dashed else "solid",
        zorder=1,
    )
    ax.add_patch(p)
    return p


def line(ax, pts, color=None, width=2, dashed=False, alpha=1):
    ax.plot(
        *zip(*pts),
        color=color or P["gray"],
        lw=width,
        linestyle=(0, (5, 4)) if dashed else "-",
        alpha=alpha,
        zorder=3,
        solid_capstyle="round",
    )


def arrow(ax, pts, color=None, dashed=False, width=2):
    col = color or P["gray"]
    if len(pts) > 2:
        line(ax, pts[:-1], col, width, dashed)
    ax.add_patch(
        FancyArrowPatch(
            pts[-2],
            pts[-1],
            arrowstyle="-|>",
            mutation_scale=15,
            lw=width,
            color=col,
            linestyle=(0, (5, 4)) if dashed else "solid",
            zorder=4,
        )
    )


def page(title, subtitle, h=1260):
    fig = plt.figure(figsize=(20, h / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 2000), ylim=(h, 0))
    ax.axis("off")
    txt(ax, 30, 20, title, 28, P["ink"], "bold")
    txt(ax, 30, 72, subtitle, 13, P["gray"])
    return fig, ax


def panel(ax, x, y, w, h, title, color, bg):
    box(ax, x, y, w, h, bg, bg, radius=5, lw=0)
    txt(ax, x + 15, y + 13, title, 19, color, "bold")


def note(ax, x, y, s, w=400):
    txt(ax, x, y, s, 13, P["gray"])


def pose(ax, name, x, y, w, h):
    img = plt.imread(ROOT / f"assets/g1_{name}.png")
    ax.imshow(img, extent=(x, x + w, y + h, y), aspect="auto", zorder=3)


def neural(ax, x, y, w, h, title, subtitle, color, bg, frozen=False):
    box(ax, x, y, w, h, "white", color, radius=15, lw=2.2)
    box(ax, x + 2, y + 2, w - 4, 44, bg, bg, radius=12, lw=0)
    txt(ax, x + w / 2, y + 11, title, 17, P["ink"], "bold", ha="center")
    columns = [[y + 69, y + 98, y + 127], [y + 76, y + 112], [y + 93]]
    xpos = [x + w * 0.22, x + w * 0.53, x + w * 0.8]
    for j in range(2):
        for ya in columns[j]:
            for yb in columns[j + 1]:
                line(ax, [(xpos[j], ya), (xpos[j + 1], yb)], "#ABB2B8", 1.3)
    for xx, ys in zip(xpos, columns):
        for yy in ys:
            ax.add_patch(Circle((xx, yy), 8, facecolor=bg, edgecolor=color, lw=2, zorder=5))
    txt(ax, x + w / 2, y + h - 32, subtitle, 12, color, ha="center")
    if frozen:
        txt(ax, x + w - 12, y + 49, "FIXED", 9, P["gray"], "bold", ha="right")


def buffer(ax, x, y, w, h, title, detail, color):
    ax.add_patch(
        Rectangle((x, y + 14), w, h - 28, facecolor="#FFF2DC", edgecolor=color, lw=2, zorder=2)
    )
    ax.add_patch(
        Ellipse(
            (x + w / 2, y + h - 14), w, 28, facecolor="#FFF2DC", edgecolor=color, lw=2, zorder=2
        )
    )
    ax.add_patch(
        Ellipse((x + w / 2, y + 14), w, 28, facecolor="#FFF8EB", edgecolor=color, lw=2, zorder=3)
    )
    txt(ax, x + w / 2, y + 38, title, 17, P["ink"], "bold", ha="center")
    txt(ax, x + w / 2, y + 76, detail, 12, P["gray"], ha="center")


def blocks(ax, x, y, width, labels, colors, height=47):
    gap = 6
    w = (width - gap * (len(labels) - 1)) / len(labels)
    for j, (s, c) in enumerate(zip(labels, colors)):
        box(ax, x + j * (w + gap), y, w, height, c, P["ink"], radius=5, lw=1.4)
        txt(
            ax, x + j * (w + gap) + w / 2, y + height / 2, s, 14, P["ink"], ha="center", va="center"
        )


def save(fig, name):
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(ROOT / f"{name}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = page(
        "Motion2Scene: from motion evidence to scene-and-goal control",
        (
            "Three linked learning problems: infer useful geometry, preserve movement "
            "capability, and generate commands from task context."
        ),
        1050,
    )
    panel(ax, 25, 115, 623, 550, "(a) Learn scenes from motion", P["gold"], P["goldbg"])
    txt(ax, 48, 175, "Full motion → event-conditioned proposer", 17, weight="bold")
    box(ax, 48, 220, 359, 259, "white", P["gold"], radius=9)
    target = [(73, 442), (126, 292), (223, 263), (365, 259)]
    arrow(ax, target, P["green"], width=3)
    arrow(ax, [(73, 442), (365, 259)], P["red"], True, 2)
    ax.add_patch(
        Rectangle((189, 322), 68, 47, facecolor="#D8BE88", edgecolor=P["gold"], lw=2, zorder=3)
    )
    for x, y in [(126, 292), (223, 263), (365, 259)]:
        ax.add_patch(Circle((x, y), 5, color=P["green"]))
    txt(ax, 67, 236, "target path", 13, P["green"])
    txt(ax, 217, 388, "chord probe", 13, P["red"])
    pose(ax, "crouch", 423, 236, 194, 234)
    txt(
        ax,
        48,
        491,
        "Hindsight: preserve the observed movement;\nplace geometry that challenges an alternative.",
        16,
    )
    txt(
        ax,
        48,
        559,
        "225 fixed recipes → learned categorical scores\nClearance mask → sequential scene sampling",
        15,
        P["gold"],
        "bold",
    )
    txt(ax, 48, 627, "Schematic geometry; reference pose, not a rollout.", 12, P["gray"])

    panel(ax, 667, 115, 651, 550, "(b) Recover the full-command motor", P["blue"], P["bluebg"])
    neural(
        ax,
        689,
        218,
        281,
        184,
        "Tracking teacher",
        "future + measured history",
        P["gold"],
        P["goldbg"],
    )
    neural(
        ax,
        1015,
        218,
        281,
        184,
        "Forecaster Fθ",
        "history + current command",
        P["blue"],
        P["bluebg"],
    )
    arrow(ax, [(972, 309), (1013, 309)], P["violet"])
    txt(ax, 689, 429, "Online action / token distillation", 17, P["violet"], "bold")
    box(ax, 689, 480, 606, 97, P["frozen"], P["gray"], radius=10, lw=2)
    txt(ax, 708, 497, "Predicted reference → fixed E → Q → D", 18, weight="bold")
    txt(ax, 708, 536, "Retained motor representation → 29D action", 14)
    txt(ax, 689, 609, "Teacher recovery and scene fitting are separate studies.", 13, P["gray"])

    panel(ax, 1337, 115, 638, 550, "(c) Learn navigation from context", P["green"], P["greenbg"])
    blocks(
        ax,
        1359,
        183,
        593,
        ["history930", "request10", "map5 × 15"],
        [P["bluebg"], P["goldbg"], P["goldbg"]],
        54,
    )
    neural(
        ax,
        1359,
        270,
        321,
        182,
        "Navigation Gη",
        "geometry + goal → command114",
        P["green"],
        P["greenbg"],
    )
    arrow(ax, [(1682, 360), (1731, 360)], P["green"])
    box(ax, 1735, 288, 216, 145, P["frozen"], P["gray"], radius=12, lw=2)
    txt(ax, 1755, 309, "Fixed motor M", 16, weight="bold")
    txt(ax, 1755, 353, "Fθ + E + Q + D", 14)
    txt(ax, 1360, 482, "Full command is completed internally.", 17, weight="bold")
    txt(
        ax,
        1360,
        531,
        "Approach → brake → stable goal hold\nKnown-map control; no reference at inference",
        16,
    )
    txt(ax, 1360, 609, "Camera context and wider tasks remain research branches.", 13, P["gray"])

    arrow(ax, [(335, 666), (335, 728)], P["gold"])
    arrow(ax, [(994, 666), (994, 728)], P["blue"])
    panel(
        ax,
        25,
        730,
        1293,
        211,
        "(d) Convert compatible scenes and motor capability into qualified supervision",
        P["violet"],
        P["violetbg"],
    )
    txt(
        ax,
        48,
        791,
        "Scene + goal → execute continuation → score task → admit supported rows",
        17,
        weight="bold",
    )
    txt(
        ax,
        48,
        837,
        (
            "Fresh 50-tick hold; distance, speed, contact, fall and deadline. Retain failed "
            "attempts and collection cost.\nLearner-state recovery supplies bounded local "
            "support; scene acceptance alone supplies no action label."
        ),
        15,
    )
    arrow(ax, [(1320, 833), (1410, 833), (1410, 667)], P["violet"])
    txt(ax, 1454, 750, "Fit navigation; roll out again", 19, P["violet"], "bold")
    txt(
        ax,
        1454,
        797,
        (
            "Current evidence: 4/8 initial tasks,\n1/8 on evaluation-seed confirmation.\n"
            "End-to-end data utility remains unproven."
        ),
        15,
    )
    txt(
        ax,
        30,
        981,
        (
            "Expanded architecture plates: Figure 2 — proposer and loss; Figure 3 — teacher "
            "and motor; Figure 4 — context student and recovery."
        ),
        14,
        P["gray"],
    )
    save(fig, "fig0_research_overview")


def generator():
    fig, ax = page(
        "Motion2Scene: learning a hindsight obstacle distribution",
        (
            "Dataset-generating branch • 21,185 learned parameters • geometry-derived "
            "supervision • zero dynamics in generator fitting"
        ),
        1390,
    )
    panel(ax, 25, 113, 463, 850, "(a) Motion and event anchors", P["gold"], P["goldbg"])
    txt(ax, 47, 171, "Full reference motion M", 19, weight="bold")
    for name, x in [("walk", 45), ("turn", 190), ("crouch", 333)]:
        pose(ax, name, x, 205, 135, 190)
    arrow(ax, [(69, 405), (447, 405)], P["gold"])
    txt(ax, 170, 420, "offline motion time", 13, P["gold"])
    txt(ax, 47, 462, "Fixed native forward kinematics", 17, weight="bold")
    blocks(ax, 47, 503, 416, ["COM", "hip", "head", "hands", "feet"], [P["bluebg"]] * 5, 42)
    txt(ax, 47, 565, "64 channels: positions, derivatives,\norientation and local path change", 15)
    arrow(ax, [(255, 620), (255, 650)], P["gold"])
    txt(ax, 47, 662, "5 event anchors (heuristic)", 17, weight="bold")
    t = np.linspace(0, 1, 160)
    signal = (
        0.1
        + 0.65 * np.exp(-(((t - 0.28) / 0.06) ** 2))
        + 0.5 * np.exp(-(((t - 0.55) / 0.06) ** 2))
        + 0.8 * np.exp(-(((t - 0.79) / 0.06) ** 2))
    )
    line(ax, list(zip(65 + t * 355, 765 - signal * 56)), P["gold"], 2)
    for f in [0, 0.28, 0.55, 0.79, 1]:
        line(ax, [(65 + f * 355, 706), (65 + f * 355, 778)], P["gold"], 1, True)
        ax.add_patch(Circle((65 + f * 355, 783), 5, color=P["gold"]))
    txt(ax, 47, 800, "Endpoints + 3 separated activity peaks", 13, P["gray"])
    txt(
        ax,
        47,
        837,
        "5 anchors × 5 lateral offsets\n× 3 heights × 3 shape templates\n= 225 candidates per motion",
        17,
        weight="bold",
    )

    panel(
        ax,
        503,
        113,
        966,
        440,
        "(b) Learned conditional categorical proposer",
        P["blue"],
        P["bluebg"],
    )
    txt(ax, 524, 174, "Local summary", 15, weight="bold")
    txt(ax, 524, 209, "mean / std over ±0.5 s\n128D; train-only\nnormalization", 13)
    txt(ax, 524, 372, "Candidate recipe", 16, weight="bold")
    txt(ax, 524, 408, "(time, lateral, height) / 3\n+ shape one-hot → 6D", 12)
    neural(ax, 795, 168, 251, 168, "Motion MLP", "128 → 64 → 64 · SiLU", P["blue"], P["bluebg"])
    neural(ax, 795, 356, 251, 166, "Recipe MLP", "6 → 64 · SiLU", P["blue"], P["bluebg"])
    arrow(ax, [(768, 231), (793, 231)], P["blue"])
    arrow(ax, [(768, 420), (793, 420)], P["blue"])
    neural(
        ax,
        1111,
        260,
        334,
        184,
        "Shared scoring head",
        "concat128 → 64 → 1 · SiLU",
        P["blue"],
        P["bluebg"],
    )
    arrow(ax, [(1048, 249), (1074, 249), (1074, 305), (1109, 305)], P["blue"])
    arrow(ax, [(1048, 438), (1074, 438), (1074, 401), (1109, 401)], P["blue"])
    txt(ax, 1110, 466, "225 logits → softmax\nqψ(j | M)", 16, P["blue"], "bold")
    arrow(ax, [(489, 862), (495, 862), (495, 240), (521, 240)], P["gold"])

    panel(
        ax,
        503,
        568,
        966,
        395,
        "(c) Fixed hindsight teacher → learning objective",
        P["violet"],
        P["violetbg"],
    )
    txt(ax, 525, 628, "All-frame target clearance", 17, weight="bold")
    txt(ax, 525, 667, "Outer capsule bounds → Cj\nNear-target geometry → Nj", 15)
    txt(ax, 1005, 628, "Two local geometric probes", 17, weight="bold")
    txt(ax, 1005, 667, "Root chord / entry-pose hold\nInner-primitive interference → Kj", 15)
    arrow(ax, [(745, 727), (745, 756)], P["violet"])
    arrow(ax, [(1215, 727), (1215, 756)], P["violet"])
    box(ax, 526, 763, 919, 151, "white", P["violet"], radius=12, lw=2)
    txt(ax, 547, 780, "w = 0.1 C + N + 3 K;    p* = w / Σw", 20, P["violet"], "bold")
    txt(ax, 547, 824, "Lgen = CE(p*, qC) − 2 log ZC", 20, P["violet"])
    txt(
        ax,
        547,
        867,
        "qC = q · C / ZC;  ZC = Σj qj Cj.  Update ψ only; no physics gradients.",
        13,
        P["gray"],
    )
    arrow(ax, [(1447, 833), (1460, 833), (1460, 474), (1434, 474)], P["red"], True, 2.5)
    txt(
        ax,
        524,
        928,
        "Training: 100 motions; 200 Adam updates/fit; lr 10⁻³; batch 16; two seeds.",
        13,
        P["gray"],
    )

    panel(ax, 1484, 113, 491, 850, "(d) Sampling → paired scenes", P["green"], P["greenbg"])
    txt(ax, 1504, 177, "Freeze proposer; score candidates", 17, weight="bold")
    data = json.loads((ROOT / "assets/proposal_example.json").read_text())
    q = np.array([x["raw_q"] for x in data])
    mask = np.array([x["reference_clear"] for x in data])
    ax.imshow(
        q.reshape(15, 15), extent=(1520, 1710, 443, 253), cmap="Blues", aspect="auto", zorder=2
    )
    ax.imshow(
        (q * mask / (q * mask).sum()).reshape(15, 15),
        extent=(1743, 1933, 443, 253),
        cmap="Greens",
        aspect="auto",
        zorder=2,
    )
    txt(ax, 1560, 222, "raw q", 16, P["blue"], "bold")
    txt(ax, 1776, 222, "filtered qC", 16, P["green"], "bold")
    arrow(ax, [(1715, 347), (1738, 347)], P["green"])
    txt(
        ax,
        1510,
        459,
        "Recorded example: clip 00916; 225 recipe cells.\nGeometric filtering, not learned collision safety.",
        12,
        P["gray"],
    )
    txt(ax, 1504, 524, "Sequential categorical sampling", 17, weight="bold")
    txt(
        ax,
        1504,
        565,
        (
            "Uniform draw u → inverse CDF\nRe-mask overlap with selected objects\nRenormalize "
            "before every next draw\nEmpty support → recorded abstention"
        ),
        15,
    )
    arrow(ax, [(1725, 694), (1725, 726)], P["green"])
    buffer(
        ax,
        1510,
        735,
        446,
        165,
        "3- or 5-obstacle scenes",
        "120 references → 240 scene records\nSame seed: 3-scene is a 5-scene prefix",
        P["gold"],
    )
    txt(ax, 1505, 923, "Export geometry + lineage + sampling receipt", 13, P["gray"])

    panel(
        ax,
        25,
        980,
        1950,
        302,
        "(e) Why this is hindsight — and where physical teaching begins",
        P["ink"],
        P["frozen"],
    )
    pose(ax, "walk", 49, 1047, 135, 186)
    pose(ax, "crouch", 211, 1047, 135, 186)
    txt(ax, 370, 1050, "Motion supplies its own geometric supervision.", 18, weight="bold")
    txt(
        ax,
        370,
        1094,
        (
            "Offline access to the complete motion allows hindsight labels.\nLearn preference "
            "over fixed recipes, not a unique original room.\nChord / held-pose probes require"
            " separate physical qualification."
        ),
        14,
    )
    arrow(ax, [(1190, 1174), (1250, 1174)], P["green"])
    box(ax, 1270, 1050, 670, 194, "white", P["green"], radius=12, lw=2)
    txt(ax, 1293, 1071, "Separate physical qualification", 19, P["green"], "bold")
    txt(
        ax,
        1293,
        1117,
        (
            "Repair reference → execute compatible scene continuation\nMeasure goal hold, "
            "contact, fall, and termination\nOnly executed task evidence supplies positive "
            "action labels"
        ),
        15,
    )
    txt(
        ax,
        30,
        1310,
        (
            "Blue: learned scorer. Violet: fixed geometry-derived teacher. Green: scene "
            "sampling. Red dashed: optimizer gradient. Robot images: reference poses, not "
            "rollouts."
        ),
        12,
        P["gray"],
    )
    save(fig, "fig1_hindsight_generator")


def motor():
    fig, ax = page(
        "Teacher training and full-command motor distillation",
        (
            "Current proposed full-command architecture: predict missing desired-reference "
            "context while preserving the pretrained SONIC motor representation."
        ),
        1350,
    )
    panel(ax, 25, 115, 467, 781, "(a) Privileged tracking teacher", P["gold"], P["goldbg"])
    txt(ax, 47, 171, "Repaired G1 motion library", 18, weight="bold")
    pose(ax, "walk", 48, 210, 135, 210)
    pose(ax, "crouch", 195, 210, 135, 210)
    txt(ax, 347, 265, "89 train\n20 dev", 16, P["gold"], "bold")
    blocks(ax, 47, 443, 421, ["xpriv", "rtrue", "ht"], [P["goldbg"], P["goldbg"], P["bluebg"]])
    arrow(ax, [(253, 498), (253, 530)], P["gold"])
    neural(
        ax,
        69,
        543,
        378,
        190,
        "SONIC actor + critic",
        "release initialization → tracking PPO",
        P["gold"],
        P["goldbg"],
    )
    arrow(ax, [(447, 624), (477, 624), (477, 801), (426, 801)], P["gold"])
    buffer(
        ax, 74, 765, 352, 110, "Simulator rollouts", "tracking rewards + terminations", P["gold"]
    )
    arrow(ax, [(72, 804), (48, 804), (48, 624), (67, 624)], P["gold"])

    panel(
        ax,
        507,
        115,
        1468,
        265,
        "(b) Public full-command interface — 114 values, 113 available",
        P["green"],
        P["greenbg"],
    )
    labs = [
        "heading\n2",
        "velocity\n3",
        "height\n1",
        "yaw rate\n1",
        "arrival\n1",
        "keypoints\n42",
        "joints\n29",
        "joint vel.\n29",
        "orientation\n6",
    ]
    cols = [P["bluebg"]] * 4 + ["#CDD1D4", P["goldbg"], "#E6BDCE", "#E6BDCE", P["bluebg"]]
    blocks(ax, 531, 180, 1418, labs, cols, 76)
    txt(
        ax,
        533,
        279,
        "Current target only + native measured history h ∈ R⁹³⁰. Arrival coordinate 7 stays masked.",
        17,
    )
    txt(
        ax,
        533,
        323,
        (
            "The 79D masked CVAE and 114D transformer are earlier controls; the recovered "
            "specialist below is a deterministic reference forecaster."
        ),
        13,
        P["gray"],
    )

    panel(
        ax,
        507,
        395,
        1468,
        501,
        "(c) Full-command inference through the retained motor path",
        P["blue"],
        P["bluebg"],
    )
    arrow(ax, [(715, 381), (715, 393)], P["green"])
    neural(
        ax,
        531,
        462,
        365,
        185,
        "Reference forecaster Fθ",
        "1044 → 512 → 512 → 576 · SiLU",
        P["blue"],
        P["bluebg"],
    )
    txt(ax, 548, 672, "h930 + c114 → 9 × 64 residuals", 16, P["blue"], "bold")
    txt(
        ax,
        548,
        711,
        "Current frame r0 comes from c.\nFuture frames: r̂k = r0 + Δrk.\nTrue future appears only in training.",
        14,
    )
    arrow(ax, [(898, 552), (936, 552)], P["green"])
    box(ax, 940, 463, 1008, 183, "white", P["blue"], radius=12, lw=2)
    txt(ax, 960, 481, "Predicted physical frames", 17, weight="bold")
    blocks(
        ax,
        961,
        522,
        963,
        ["r0"] + [f"r̂{k}" for k in range(1, 10)],
        [P["goldbg"]] + [P["bluebg"]] * 9,
        50,
    )
    txt(
        ax,
        960,
        591,
        "Each frame = qtarget29 + q̇target29 + relative orientation6; pack with the native layout.",
        14,
    )
    arrow(ax, [(1098, 648), (1098, 700)], P["green"])
    boxes = [
        (948, 300, "Encoder E", "640 → 2048 → 1024 → 512 → 512 → 64"),
        (1290, 266, "FSQ Q", "reshape 2 × 32; 32 levels"),
        (1599, 349, "Decoder D", "64 tokens + h930 → a29"),
    ]
    for x, w, title, sub in boxes:
        box(ax, x, 702, w, 146, P["frozen"], P["gray"], radius=13, lw=2)
        txt(ax, x + w / 2, 719, title, 21, weight="bold", ha="center")
        txt(ax, x + w / 2, 763, sub, 11, ha="center")
        txt(ax, x + w / 2, 808, "FROZEN PRETRAINED", 10, P["gray"], "bold", ha="center")
    arrow(ax, [(1250, 772), (1288, 772)], P["green"])
    arrow(ax, [(1558, 772), (1597, 772)], P["green"])
    txt(
        ax,
        549,
        825,
        "E, Q, D fixed; train Fθ only.\nNo state-transition forecast is claimed.",
        14,
        P["gray"],
    )

    panel(
        ax,
        25,
        914,
        1950,
        321,
        "(d) Online distillation — teacher actions at the states visited by the student",
        P["violet"],
        P["violetbg"],
    )
    box(ax, 48, 980, 492, 194, "white", P["gold"], radius=12, lw=2)
    txt(ax, 68, 1000, "Query the fixed teacher", 18, P["gold"], "bold")
    txt(
        ax,
        68,
        1043,
        (
            "Student pre-action state + true reference\n→ a*29 and z*64; same-state label\n"
            "Randomized starts / explicit takeovers"
        ),
        15,
    )
    arrow(ax, [(542, 1076), (584, 1076)], P["violet"])
    buffer(
        ax,
        589,
        980,
        423,
        194,
        "Fresh query + nominal replay",
        "Preserve timing, masks and source identity\nAssistance is not autonomous success",
        P["gold"],
    )
    arrow(ax, [(1014, 1076), (1056, 1076)], P["violet"])
    box(ax, 1062, 980, 887, 194, "white", P["violet"], radius=12, lw=2)
    txt(
        ax,
        1083,
        1000,
        "Lmotor = MSE(a, a*) + λz MSE(z, z*) + λr Lreference",
        20,
        P["violet"],
        "bold",
    )
    txt(
        ax,
        1083,
        1047,
        (
            "λr is optional and configuration-bound. Backpropagate through fixed E/Q/D to Fθ."
            "\nThen evaluate the public student without future-reference inputs or teacher "
            "assistance."
        ),
        14,
    )
    arrow(ax, [(1505, 1175), (1505, 1213), (717, 1213), (717, 898)], P["red"], True)
    txt(ax, 1160, 1191, "update forecaster → roll out again", 13, P["red"])
    txt(
        ax,
        30,
        1265,
        (
            "Green: action inference. Amber: privileged training inputs. Red dashed: "
            "gradient/update path. Gray: frozen components. Reference pose renders are "
            "illustrative."
        ),
        12,
        P["gray"],
    )
    txt(
        ax,
        30,
        1300,
        (
            "Do not identify this deterministic motor specialist with BFM's CVAE or "
            "BFM-Zero's forward–backward representation learning."
        ),
        14,
        P["ink"],
        "bold",
    )
    save(fig, "fig2_teacher_full_command")


def navigation():
    fig, ax = page(
        "From full commands to scene-and-goal behavior",
        (
            "The navigation actor completes a rich motor command internally; the public "
            "interface requires no motion ID, reference phase, or external reference "
            "trajectory."
        ),
        1390,
    )
    panel(ax, 25, 113, 481, 790, "(a) Public context at decision t", P["gold"], P["goldbg"])
    pose(ax, "stand", 47, 177, 134, 198)
    txt(ax, 192, 189, "Proprioceptive history\nh ∈ R⁹³⁰", 18, weight="bold")
    blocks(ax, 193, 278, 285, ["q", "q̇", "ω", "g", "a−1"], [P["bluebg"]] * 5, 48)
    txt(ax, 47, 397, "Known map: ≤5 obstacles", 18, weight="bold")
    box(ax, 48, 439, 433, 196, "white", P["gold"], radius=8)
    for xy, wh in [((205, 472), (49, 85)), ((337, 557), (65, 30))]:
        ax.add_patch(Rectangle(xy, *wh, facecolor="#D8BE88", edgecolor=P["gold"], lw=2, zorder=2))
    arrow(ax, [(78, 596), (160, 469), (281, 459), (437, 465)], P["green"], width=2.5)
    ax.add_patch(Circle((443, 465), 17, fill=False, edgecolor=P["green"], lw=2))
    txt(ax, 61, 613, "schematic task geometry", 10, P["gray"])
    txt(ax, 47, 660, "10D navigation request", 18, weight="bold")
    txt(ax, 47, 699, "start3 + goal3 + tolerance1\n+ terminal speed1 + hold1 + map-known1", 14)
    txt(ax, 47, 767, "Optional causal localization: 4D", 16, weight="bold")
    txt(
        ax, 47, 808, "Rᵀ(pt − pt−5) / Δt + valid bit\nIdeal pose stream; 5-step backward window", 14
    )

    panel(
        ax,
        521,
        113,
        947,
        790,
        "(b) Train navigation; preserve the entire motor specialist",
        P["green"],
        P["greenbg"],
    )
    blocks(ax, 544, 184, 454, ["center3", "size3", "rot6", "type3"], [P["goldbg"]] * 4, 57)
    txt(ax, 549, 256, "5 × 15D obstacle rows + 5 validity bits", 14)
    neural(
        ax,
        546,
        309,
        360,
        181,
        "Shared obstacle encoder",
        "15 → 64 → 64 · SiLU",
        P["green"],
        P["greenbg"],
    )
    arrow(ax, [(908, 401), (945, 401)], P["green"])
    box(ax, 948, 331, 492, 144, "white", P["green"], radius=12, lw=2)
    txt(ax, 971, 352, "Masked mean pooling", 20, weight="bold")
    txt(
        ax,
        971,
        404,
        "64D geometry feature; order-invariant\nUnknown map is rejected, not treated as free.",
        14,
    )
    arrow(ax, [(1193, 477), (1193, 520)], P["green"])
    neural(
        ax,
        963,
        535,
        478,
        191,
        "Command predictor Gη",
        "1004 (+4) → 512 → 512 → 114 · SiLU",
        P["green"],
        P["greenbg"],
    )
    arrow(ax, [(508, 736), (931, 736), (931, 630), (961, 630)], P["green"])
    txt(ax, 550, 685, "normalized h930 + request10\n+ pooled map64 (+ localization4)", 14)
    arrow(ax, [(1202, 728), (1202, 763)], P["green"])
    blocks(
        ax,
        544,
        780,
        896,
        ["root / heading", "body targets", "joint pose / velocity", "orientation"],
        [P["bluebg"], P["goldbg"], "#E6BDCE", P["bluebg"]],
        65,
    )
    txt(
        ax,
        547,
        863,
        "ĉ114 is generated internally. Arrival remains masked. Full-command requests bypass Gη.",
        13,
        P["gray"],
    )

    panel(ax, 1483, 113, 492, 790, "(c) Fixed motor → physical task", P["blue"], P["bluebg"])
    box(ax, 1505, 178, 448, 293, P["frozen"], P["gray"], radius=13, lw=2)
    txt(ax, 1526, 197, "M(h, ĉ)", 24, weight="bold")
    for yy, s in [
        (249, "Fθ: desired-reference anticipation"),
        (294, "E: pretrained reference encoder"),
        (339, "Q: native finite scalar quantizer"),
        (384, "D: history-conditioned decoder"),
    ]:
        txt(ax, 1526, yy, s, 16)
    txt(ax, 1526, 433, "All weights + normalizers unchanged", 13, P["gray"], "bold")
    arrow(ax, [(1441, 810), (1475, 810), (1475, 325), (1503, 325)], P["green"])
    arrow(ax, [(1728, 473), (1728, 526)], P["green"])
    txt(ax, 1662, 507, "29 actions", 15, P["green"], "bold")
    pose(ax, "walk", 1510, 542, 130, 212)
    pose(ax, "stand", 1747, 542, 130, 212)
    arrow(ax, [(1648, 685), (1740, 685)], P["green"])
    txt(ax, 1510, 772, "Approach → brake → maintain goal hold", 16, weight="bold")
    txt(
        ax,
        1510,
        816,
        "3D distance ≤0.25 m; speed ≤0.10 m/s\n50 ticks; contact / fall / deadline checked",
        14,
    )

    panel(
        ax,
        25,
        918,
        1950,
        305,
        "(d) Task-qualified supervision and the navigation-induced recovery loop",
        P["violet"],
        P["violetbg"],
    )
    buffer(
        ax,
        48,
        984,
        407,
        181,
        "Task / recovery replay",
        "Successful demonstrations + qualified suffixes\nRejected attempts retain their cost",
        P["gold"],
    )
    arrow(ax, [(457, 1075), (495, 1075)], P["violet"])
    box(ax, 500, 984, 705, 181, "white", P["violet"], radius=12, lw=2)
    txt(ax, 521, 1003, "a* = stopgrad M(h, c*)", 21, P["violet"], "bold")
    txt(ax, 521, 1047, "Lnav = MSE(M(h, ĉ), a*) + 0.1 Lgroup(ĉ, c*)", 18, P["violet"])
    txt(
        ax,
        521,
        1090,
        (
            "Eight normalized command groups; unavailable arrival excluded.\nTarget uses the "
            "same frozen backend at the same measured history."
        ),
        13,
    )
    arrow(ax, [(1207, 1075), (1245, 1075)], P["violet"])
    box(ax, 1250, 984, 702, 181, "white", P["violet"], radius=12, lw=2)
    txt(ax, 1270, 1003, "Learner prefix → physical continuation test", 18, P["violet"], "bold")
    txt(
        ax,
        1270,
        1048,
        (
            "Retain actual state/history; no phase jump or teleport.\nA supported suffix earns"
            " a fresh 50-tick hold on time.\nRefit on supported data + replay; evaluate "
            "without intervention."
        ),
        14,
    )
    arrow(ax, [(853, 1167), (853, 1199), (1199, 1199), (1199, 905)], P["red"], True)
    txt(ax, 920, 1180, "gradients to Gη only", 13, P["red"])
    box(ax, 25, 1245, 1950, 106, P["frozen"], P["gray"], radius=7, lw=1)
    txt(ax, 45, 1262, "Next comparisons", 18, weight="bold")
    txt(
        ax,
        431,
        1265,
        (
            "Direct goal/map → desired-reference head  •  coherent reference-flow chunks  •  "
            "task RL  •  camera belief"
        ),
        16,
    )
    txt(
        ax,
        431,
        1305,
        "Research branches: geometry-conditioned action change alone is not evidence of task success.",
        13,
        P["gray"],
    )
    save(fig, "fig3_navigation_context")


def inverse_appendix():
    fig, ax = page(
        "A separate Motion2Scene branch: differentiable inverse beam learning",
        (
            "This Gaussian beam diagnostic predates the categorical dataset generator. Its "
            "losses and sampling law must not be attributed to hindsight-801."
        ),
        950,
    )
    panel(
        ax,
        25,
        120,
        1950,
        665,
        "(a) Motion-only inference; fixed geometric preference supplies training feedback",
        P["blue"],
        P["bluebg"],
    )
    pose(ax, "crouch", 44, 235, 184, 281)
    txt(ax, 47, 570, "Target capsules\nT × B × 7", 17, weight="bold")
    neural(
        ax,
        307,
        246,
        350,
        203,
        "Temporal encoder",
        "7 → 16; 3 temporal convolutions",
        P["blue"],
        P["bluebg"],
    )
    arrow(ax, [(231, 358), (305, 358)], P["blue"])
    neural(
        ax,
        725,
        246,
        401,
        203,
        "Gaussian mixture (K=4)",
        "weights πk, means μk, Cholesky Lk",
        P["blue"],
        P["bluebg"],
    )
    arrow(ax, [(659, 358), (723, 358)], P["blue"])
    box(ax, 1191, 246, 358, 203, "white", P["green"], radius=12, lw=2)
    txt(ax, 1214, 269, "u = μk + Lk ε", 23, P["green"], "bold")
    txt(
        ax, 1214, 323, "ε ~ N(0,I); k ~ Cat(π)\nBounded sigmoid → (s,h)\nFixed beam sizes / yaw", 15
    )
    arrow(ax, [(1128, 358), (1189, 358)], P["green"])
    box(ax, 1614, 246, 336, 203, P["goldbg"], P["gold"], radius=12, lw=2)
    txt(ax, 1636, 268, "Fixed selector", 18, weight="bold")
    txt(
        ax,
        1636,
        317,
        "Target / competing motions\nClearance → energy\nPreference + feasibility",
        14,
    )
    arrow(ax, [(1551, 358), (1612, 358)], P["green"])
    box(ax, 305, 516, 1645, 176, "white", P["violet"], radius=12, lw=2)
    txt(
        ax,
        329,
        535,
        "Linv = E[Lpreference + 5 Lclearance] + 0.02 KL(qψ(u|M) || N(0,I))",
        24,
        P["violet"],
        "bold",
    )
    txt(
        ax,
        329,
        585,
        (
            "Train with reparameterized samples from every component; use mixture weights and"
            " marginal log density.\nNo target obstacle coordinates enter this loss. Candidate"
            " costs are fixed: upright 0, crouch 1.\nThe loss target is the desired motion "
            "choice. Frozen geometry feedback does not prove obstacle-present execution."
        ),
        15,
    )
    arrow(ax, [(1780, 451), (1780, 514)], P["violet"])
    arrow(
        ax, [(725, 694), (725, 746), (275, 746), (275, 478), (482, 478), (482, 451)], P["red"], True
    )
    txt(
        ax,
        811,
        725,
        "update encoder + mixture head; no learned planner or physics gradients",
        14,
        P["red"],
    )
    txt(
        ax,
        30,
        825,
        (
            "Dataset generator (Figure 2): fixed discrete recipes + geometric target "
            "distribution + coverage/acceptance loss."
        ),
        17,
        weight="bold",
    )
    txt(
        ax,
        30,
        868,
        (
            "This diagnostic: continuous station/height samples + fixed motion-choice energy "
            "+ target barrier + distribution KL."
        ),
        17,
        weight="bold",
    )
    save(fig, "figA_inverse_beam")


if __name__ == "__main__":
    overview()
    generator()
    motor()
    navigation()
    inverse_appendix()
    print("Rendered five model diagrams as PNG, SVG, and PDF.")
