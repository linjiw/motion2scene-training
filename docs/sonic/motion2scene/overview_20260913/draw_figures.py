"""Render editable method figures. All robot/scene drawings are conceptual.

Run from the repository root with .venv_research/bin/python and this file path.
No simulation, learned weights, or running experiment is accessed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
C = dict(
    ink="#172B42",
    muted="#54667B",
    teal="#167D86",
    teal_bg="#EAF7F5",
    purple="#7453AD",
    purple_bg="#F1EDF9",
    blue="#326CC1",
    blue_bg="#ECF3FC",
    amber="#AC6E17",
    amber_bg="#FFF5E5",
    red="#BF5A57",
    gray="#E5EAF0",
    paper="#FFFFFF",
    frozen="#F0F3F7",
)
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 11}
)


def canvas(w, h, title, subtitle, number):
    fig = plt.figure(figsize=(w / 100, h / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, w), ylim=(h, 0))
    ax.axis("off")
    txt(
        ax,
        45,
        34,
        f"MOTION2SCENE  /  METHODS ATLAS                                      FIGURE {number}",
        12,
        C["teal"],
        "bold",
    )
    txt(ax, 45, 78, title, 28, weight="bold")
    txt(ax, 45, 131, subtitle, 13, C["muted"])
    return fig, ax


def txt(ax, x, y, s, size=12, color=None, weight="normal", ha="left", va="top"):
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color or C["ink"],
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
        edgecolor=edge or C["gray"],
        linewidth=lw,
        linestyle=(0, (5, 4)) if dashed else "solid",
        zorder=1,
    )
    ax.add_patch(p)
    return p


def line(ax, pts, color=None, width=2, dashed=False, alpha=1):
    ax.plot(
        *zip(*pts),
        color=color or C["muted"],
        lw=width,
        linestyle=(0, (5, 4)) if dashed else "-",
        alpha=alpha,
        zorder=3,
        solid_capstyle="round",
    )


def arrow(ax, pts, color=None, dashed=False, width=2):
    col = color or C["muted"]
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


def pill(ax, x, y, s, col, fill, w):
    box(ax, x, y, w, 28, fill, fill, radius=7, lw=0)
    txt(ax, x + w / 2, y + 6, s, 9, col, "bold", ha="center")


def band(ax, x, y, w, h, letter, title, col, bg):
    box(ax, x, y, w, h, bg, bg, radius=18)
    box(ax, x + 18, y + 16, 33, 33, col, col, radius=9)
    txt(ax, x + 34.5, y + 22, letter, 13, "white", "bold", ha="center")
    txt(ax, x + 65, y + 22, title, 16, col, "bold")


def robot(ax, x, ground, scale=1, crouch=False, color=None, alpha=1):
    col = color or C["teal"]
    if crouch:
        hip, chest, head = (x - 12, ground - 62), (x + 12, ground - 103), (x + 24, ground - 128)
        legs = [
            [hip, (x + 24, ground - 37), (x - 4, ground)],
            [hip, (x - 40, ground - 30), (x - 22, ground)],
        ]
        arms = [
            [chest, (x + 38, ground - 78), (x + 16, ground - 58)],
            [chest, (x - 9, ground - 83), (x + 5, ground - 65)],
        ]
    else:
        hip, chest, head = (x, ground - 84), (x + 2, ground - 142), (x + 3, ground - 170)
        legs = [
            [hip, (x + 15, ground - 43), (x + 29, ground)],
            [hip, (x - 19, ground - 42), (x - 32, ground)],
        ]
        arms = [
            [chest, (x + 29, ground - 119), (x + 17, ground - 93)],
            [chest, (x - 24, ground - 118), (x - 15, ground - 98)],
        ]

    def trans(p):
        return (x + (p[0] - x) * scale, ground + (p[1] - ground) * scale)

    for limb in legs + arms + [[hip, chest]]:
        line(ax, list(map(trans, limb)), col, 6 * scale, alpha=alpha)
    hx, hy = trans(head)
    ax.add_patch(
        Circle(
            (hx, hy),
            12 * scale,
            facecolor="white",
            edgecolor=col,
            lw=3 * scale,
            alpha=alpha,
            zorder=5,
        )
    )
    for p in [hip, chest] + [limb[1] for limb in legs]:
        ax.add_patch(Circle(trans(p), 4 * scale, color=col, alpha=alpha, zorder=5))


def beam(ax, x, y, w, h=25, col=None):
    col = col or C["amber"]
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#ECDCC2", edgecolor=col, lw=1.7, zorder=2))
    ax.add_patch(
        Polygon(
            [(x, y), (x + 13, y - 10), (x + w + 13, y - 10), (x + w, y)],
            facecolor="#F8EDDB",
            edgecolor=col,
            lw=1.2,
            zorder=2,
        )
    )


def save(fig, name):
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(
            OUT / f"{name}.{ext}",
            dpi=160 if ext == "png" else 100,
            facecolor="white",
            metadata={"Creator": "Motion2Scene methods atlas"},
        )
    plt.close(fig)


def overview():
    fig, ax = canvas(
        2220,
        1480,
        "From motion evidence to scene-conditioned humanoid behavior",
        (
            "A connected research program: construct informative tasks, preserve motor "
            "competence, then learn task-conditioned execution."
        ),
        "1",
    )
    pill(ax, 45, 176, "SOLID: IMPLEMENTED INTERFACE", C["ink"], C["frozen"], 285)
    pill(ax, 345, 176, "DASHED: PROPOSED EXTENSION", C["muted"], C["frozen"], 280)
    txt(
        ax,
        660,
        182,
        (
            "Implemented does not imply task-qualified.  •  Illustration, not a rollout."
            "  •  Repository snapshot: 13 September 2026"
        ),
        11,
        C["muted"],
    )

    band(
        ax,
        40,
        226,
        2140,
        342,
        "A",
        "MOTION2SCENE  ·  Turn motion alternatives into informative environments",
        C["teal"],
        C["teal_bg"],
    )
    for x, w in [(65, 455), (560, 780), (1380, 775)]:
        box(ax, x, 291, w, 246)
    txt(ax, 84, 308, "Motion library + embodiment repair", 15, weight="bold")
    txt(
        ax,
        84,
        349,
        (
            "Generated / recorded / authored references\nG1 retargeting, dynamics and "
            "joint checks\nRecorded ancestry + fixed data splits"
        ),
        12,
    )
    robot(ax, 165, 501, 0.44, color=C["teal"])
    robot(ax, 276, 501, 0.44, crouch=True, color=C["teal"])
    robot(ax, 387, 501, 0.44, crouch=True, color=C["purple"])
    txt(ax, 104, 511, "walk", 9, C["muted"])
    txt(ax, 238, 511, "short tuck", 9, C["muted"])
    txt(ax, 346, 511, "sustained", 9, C["muted"])
    arrow(ax, [(522, 413), (558, 413)], C["teal"])
    txt(ax, 583, 308, "Inverse scene construction", 15, weight="bold")
    txt(ax, 583, 348, "Executed body envelopes\n+ legal entry / return schedules", 12)
    txt(
        ax,
        583,
        410,
        (
            "Learned proposals + analytic search\nTarget clearance + competing response\n"
            "Visibility before the decision deadline"
        ),
        12,
    )
    line(ax, [(1040, 500), (1306, 500)], C["gray"], 2)
    robot(ax, 1100, 500, 0.62, color=C["red"], alpha=0.75)
    robot(ax, 1234, 500, 0.62, crouch=True)
    beam(ax, 1070, 403, 219, 14)
    txt(ax, 1063, 514, "A scene where the choice matters", 9, C["teal"])
    arrow(ax, [(1342, 413), (1378, 413)], C["teal"])
    txt(ax, 1403, 308, "Physics-bound teaching records", 15, weight="bold")
    txt(
        ax,
        1403,
        349,
        (
            "Execute matched continuations in the scene\nMeasure passage, contacts, "
            "timing and termination\nKeep failed attempts; mask missing outcomes"
        ),
        12,
    )
    pill(ax, 1403, 448, "GEOMETRY PROPOSAL", C["muted"], C["frozen"], 218)
    arrow(ax, [(1632, 462), (1668, 462)], C["teal"])
    pill(ax, 1677, 448, "EXECUTED TASK LABEL", C["teal"], C["teal_bg"], 226)
    txt(
        ax,
        1403,
        495,
        "Complementary responses + observable decisions define useful supervision.",
        11,
        C["muted"],
    )
    arrow(ax, [(1690, 538), (1690, 551), (951, 551), (951, 538)], C["teal"])
    txt(
        ax,
        1318,
        534,
        "measured responses → curriculum update → next scene",
        9,
        C["teal"],
        ha="center",
    )

    arrow(ax, [(290, 540), (290, 595)], C["amber"])
    txt(ax, 310, 579, "repaired motion references", 10, C["amber"])
    arrow(
        ax,
        [(1780, 568), (1780, 588), (2210, 588), (2210, 1280), (2148, 1280)],
        C["teal"],
        dashed=True,
    )
    txt(ax, 1877, 573, "broader task-corpus integration", 9, C["teal"])
    band(
        ax,
        40,
        602,
        2140,
        357,
        "B",
        "TEACHER → STUDENT  ·  Learn missing reference context through a preserved motor model",
        C["purple"],
        C["purple_bg"],
    )
    box(ax, 65, 667, 455, 249, C["amber_bg"])
    txt(ax, 86, 683, "Privileged tracking teacher", 15, C["amber"], "bold")
    txt(
        ax,
        86,
        724,
        (
            "SONIC initialization → tracking PPO\nSimulator state + future reference "
            "window\nQualify whole-motion execution"
        ),
        12,
    )
    txt(
        ax,
        86,
        818,
        (
            "Freeze the selected teacher for distillation.\nA scene task needs its own "
            "successful\ncontinuation and independent task score."
        ),
        11,
        C["muted"],
    )
    arrow(ax, [(522, 772), (558, 772)], C["purple"])
    box(ax, 560, 667, 1000, 249)
    txt(
        ax,
        584,
        683,
        "BFM-inspired distillation → recovered anticipatory motor specialist",
        15,
        C["purple"],
        "bold",
    )
    txt(
        ax,
        584,
        724,
        "Measured history + 114D current command → predict 9 missing desired-reference frames",
        12,
    )
    steps = [
        (585, 252, "Reference forecaster", "trainable during motor fit", C["purple_bg"]),
        (880, 274, "SONIC encoder + FSQ", "frozen pretrained weights", C["frozen"]),
        (
            1197,
            337,
            "SONIC decoder → 29 actions",
            "frozen; also reads measured history",
            C["frozen"],
        ),
    ]
    for x, w, title, sub, bg in steps:
        box(ax, x, 770, w, 81, bg)
        txt(ax, x + w / 2, 785, title, 12, weight="bold", ha="center")
        txt(ax, x + w / 2, 815, sub, 9, C["muted"], ha="center")
    arrow(ax, [(839, 808), (878, 808)], C["purple"])
    arrow(ax, [(1156, 808), (1195, 808)], C["purple"])
    txt(
        ax,
        584,
        877,
        "Earlier masked CVAE/token students are comparison models; see Figure 2 for the architecture distinction.",
        10,
        C["muted"],
    )
    arrow(ax, [(1562, 772), (1603, 772)], C["purple"])
    box(ax, 1605, 667, 550, 249)
    txt(ax, 1628, 683, "Student-state learning loop", 15, C["purple"], "bold")
    txt(
        ax,
        1628,
        726,
        (
            "Roll out student → query fixed teacher\nAggregate fresh labels + nominal "
            "replay\nBalance motion / phase coverage\nRecord assisted and autonomous "
            "execution"
        ),
        12,
    )
    arrow(ax, [(1877, 888), (1877, 942), (1070, 942), (1070, 917)], C["purple"])
    txt(ax, 1350, 923, "DAgger: refit on visited states", 10, C["purple"], ha="center")

    arrow(ax, [(1230, 960), (1230, 1001)], C["blue"])
    txt(ax, 1250, 977, "retain the complete motor specialist", 10, C["blue"])
    band(
        ax,
        40,
        1007,
        2140,
        341,
        "C",
        "SCENE + GOAL  ·  Learn public task conditioning while retaining motor behavior",
        C["blue"],
        C["blue_bg"],
    )
    for x, w in [(65, 455), (560, 475), (1075, 485), (1605, 550)]:
        box(ax, x, 1072, w, 238)
    txt(ax, 87, 1089, "Public navigation inputs", 15, C["blue"], "bold")
    txt(
        ax,
        87,
        1130,
        (
            "Measured robot history\nLocalized start / goal + terminal request\nKnown 3D "
            "map + obstacle-validity masks\nOptional causal localization velocity"
        ),
        12,
    )
    txt(ax, 87, 1251, "Camera / partial-map belief: future stage", 10, C["muted"])
    arrow(ax, [(522, 1194), (558, 1194)], C["blue"])
    txt(ax, 582, 1089, "Trainable context branch", 15, C["blue"], "bold")
    txt(
        ax,
        582,
        1130,
        (
            "Goal + obstacle-set encoder\nInternally complete the 114D command\nUse "
            "successful task demonstrations\nQualify recovery from learner states"
        ),
        12,
    )
    txt(ax, 582, 1251, "Direct reference / token branch: comparison", 10, C["muted"])
    arrow(ax, [(1037, 1194), (1073, 1194)], C["blue"])
    txt(ax, 1097, 1089, "Frozen motor specialist", 15, C["blue"], "bold")
    txt(
        ax,
        1097,
        1130,
        (
            "Forecaster → encoder → FSQ → decoder\nNavigation learns through input "
            "gradients\nFull-command specialist stays unchanged\nActions execute in the "
            "native simulator"
        ),
        12,
    )
    txt(ax, 1097, 1251, "Goal/map mode receives no external reference.", 10, C["muted"])
    arrow(ax, [(1562, 1194), (1603, 1194)], C["blue"])
    txt(ax, 1628, 1089, "Independent task evaluation", 15, C["blue"], "bold")
    txt(
        ax,
        1628,
        1130,
        (
            "Reach goal → slow down → maintain hold\nMeasure contacts, falls and deadline"
            "\nTest changed goals / obstacle configurations\nSeparate motor skill from "
            "task selection"
        ),
        12,
    )
    txt(ax, 1628, 1251, "Broader avoidance + transfer remain research targets.", 10, C["muted"])
    arrow(ax, [(1878, 1312), (1878, 1371), (790, 1371), (790, 1312)], C["blue"], dashed=True)
    txt(
        ax,
        1345,
        1353,
        "Task-supported recovery → aggregation → adapter update (bounded study)",
        10,
        C["blue"],
        ha="center",
    )
    txt(
        ax,
        45,
        1420,
        (
            "Central question: can physically verified, decision-relevant experience "
            "teach scene-dependent behavior without losing the inherited motor "
            "competence?"
        ),
        13,
        weight="bold",
    )
    save(fig, "01_research_overview")


def architecture():
    fig, ax = canvas(
        2220,
        1630,
        "What the teacher sees, what the student predicts, what stays frozen",
        (
            "The current motor specialist and navigation adapter are distinct from the "
            "earlier masked CVAE. Each branch has an explicit input contract."
        ),
        "2",
    )
    band(
        ax,
        40,
        185,
        2140,
        340,
        "A",
        "CURRENT INFERENCE GRAPH  ·  A complete desired-reference window is constructed inside the policy",
        C["blue"],
        C["blue_bg"],
    )
    box(ax, 65, 254, 400, 224)
    txt(ax, 87, 273, "Two public request profiles", 15, weight="bold")
    txt(
        ax,
        87,
        317,
        (
            "Motor: history h + current command c\nNavigation: history h + goal g + map S"
            "\nOptional pose-history velocity + validity"
        ),
        12,
    )
    txt(
        ax,
        87,
        418,
        "External future / phase / motion ID\nare absent from the navigation API.",
        11,
        C["muted"],
    )
    arrow(ax, [(467, 365), (502, 365)], C["blue"])
    box(ax, 505, 254, 490, 224)
    txt(ax, 528, 273, "Navigation command completion", 15, C["blue"], "bold")
    txt(
        ax,
        528,
        315,
        (
            "5 × 15D obstacle primitives + validity mask\nShared MLP → masked mean "
            "pooling\n10D task request + 930D history\n512-wide MLP → internal 114D "
            "command"
        ),
        12,
    )
    txt(ax, 528, 445, "Only this branch trains during current navigation fitting.", 10, C["blue"])
    arrow(ax, [(997, 365), (1040, 365)], C["blue"])
    box(ax, 1043, 254, 655, 224, C["purple_bg"])
    txt(ax, 1067, 273, "Anticipatory motor F", 15, C["purple"], "bold")
    txt(
        ax,
        1067,
        315,
        (
            "930D history + 114D current command → 9 × 64D residuals\nCurrent 64D target "
            "frame is reconstructed from command\nConcatenate current + 9 predicted "
            "frames; native packing"
        ),
        12,
    )
    for i in range(10):
        box(ax, 1070 + i * 47, 421, 37, 34, C["purple"] if i else C["teal"], radius=4)
        txt(ax, 1088 + i * 47, 429, "0" if i == 0 else f"+{i}", 9, "white", ha="center")
    txt(ax, 1570, 420, "desired targets\nnot state forecasts", 9, C["muted"])
    arrow(ax, [(1700, 365), (1740, 365)], C["purple"])
    box(ax, 1743, 254, 412, 224, C["frozen"])
    txt(ax, 1767, 273, "Preserved SONIC motor path", 15, weight="bold")
    txt(
        ax,
        1767,
        315,
        (
            "640D reference → frozen encoder E\n64 outputs → finite scalar quantizer Q\n"
            "Frozen decoder D(tokens, history h)\n29 native action coordinates"
        ),
        12,
    )
    txt(ax, 1767, 445, "Encoder, FSQ and decoder retained.", 10, C["muted"])
    arrow(ax, [(190, 480), (190, 509), (1350, 509), (1350, 480)], C["teal"])
    txt(
        ax,
        795,
        490,
        "Full-command profile bypasses the navigation branch and supplies c directly",
        10,
        C["teal"],
        ha="center",
    )

    band(
        ax,
        40,
        562,
        1328,
        439,
        "B",
        "TRAINING  ·  Same measured state, privileged teacher targets",
        C["amber"],
        C["amber_bg"],
    )
    box(ax, 65, 628, 560, 214)
    txt(ax, 86, 646, "Fixed teacher at the student's state", 15, C["amber"], "bold")
    txt(
        ax,
        86,
        687,
        (
            "Privileged state + true future reference\n→ teacher action a* and native "
            "tokens z*\nStudent rollout / nominal replay are distinguished\nPreserve "
            "pre-action timing and continuation identity"
        ),
        12,
    )
    box(ax, 683, 628, 660, 214)
    txt(ax, 705, 646, "Motor imitation objective", 15, C["purple"], "bold")
    txt(
        ax,
        705,
        687,
        (
            "Lmotor = action MSE + λz token MSE + λr reference loss\nTrue future frames "
            "are targets during training only.\nUpdate F; keep E, Q and D fixed.\n"
            "Reference-loss coefficient is configuration-specific."
        ),
        12,
    )
    arrow(ax, [(627, 732), (681, 732)], C["amber"])
    box(ax, 65, 863, 1278, 107, C["frozen"])
    txt(
        ax,
        86,
        880,
        "Navigation fitting uses the exact frozen motor as its action target",
        14,
        C["blue"],
        "bold",
    )
    txt(
        ax,
        86,
        918,
        (
            "Lnav = ||M(h, ĉ) − stopgrad M(h, c*)||² + 0.1 × structured command loss; "
            "gradients update the context branch only."
        ),
        12,
    )
    band(ax, 1403, 562, 777, 439, "C", "CLOSED-LOOP DATA AGGREGATION", C["teal"], C["teal_bg"])
    loop = [
        (1428, 633, "1  Roll out current learner", "Actual states, actions, history and resets"),
        (
            1428,
            737,
            "2  Query / execute a supported continuation",
            "Candidate advice ≠ demonstrated recovery",
        ),
        (
            1428,
            841,
            "3  Refit with fresh data + replay",
            "Keep failed attempts and all interaction cost",
        ),
    ]
    for x, y, title, sub in loop:
        box(ax, x, y, 727, 84)
        txt(ax, x + 20, y + 13, title, 14, C["teal"], "bold")
        txt(ax, x + 20, y + 47, sub, 11, C["muted"])
    arrow(ax, [(1790, 717), (1790, 735)], C["teal"])
    arrow(ax, [(1790, 821), (1790, 839)], C["teal"])
    txt(
        ax,
        1428,
        949,
        "Motor DAgger: executed. Navigation-supported recovery: separate bounded study.",
        10,
        C["muted"],
    )

    band(
        ax,
        40,
        1036,
        1328,
        397,
        "D",
        "EARLIER BFM-STYLE MASKED CVAE  ·  Implemented comparison",
        C["purple"],
        C["purple_bg"],
    )
    box(ax, 65, 1104, 393, 193)
    txt(ax, 86, 1122, "Public prior p(z | h, m ⊙ c, m)", 14, C["purple"], "bold")
    txt(
        ax,
        86,
        1164,
        (
            "79D root / body / joint commands\nExplicit availability masks\nHistory + "
            "command transformer\nGaussian latent parameters"
        ),
        11,
    )
    box(ax, 493, 1104, 391, 193, C["amber_bg"])
    txt(ax, 515, 1122, "Training posterior q(z | ·)", 14, C["amber"], "bold")
    txt(
        ax,
        515,
        1164,
        (
            "Adds privileged simulator state\nand future reference\nResidual posterior "
            "mean\nAction reconstruction + KL(q || p)"
        ),
        11,
    )
    box(ax, 920, 1104, 422, 193, C["frozen"])
    txt(ax, 942, 1122, "Token adapter → frozen decoder", 14, weight="bold")
    txt(
        ax,
        942,
        1164,
        (
            "64 native quantized tokens\nDecoder also receives history\nPublic-prior "
            "action loss supports inference\nPosterior removed at execution"
        ),
        11,
    )
    arrow(ax, [(460, 1200), (491, 1200)], C["purple"], dashed=True)
    arrow(ax, [(886, 1200), (918, 1200)], C["purple"])
    arrow(ax, [(263, 1298), (263, 1321), (1131, 1321), (1131, 1298)], C["purple"])
    txt(
        ax,
        690,
        1302,
        "inference: public prior → token adapter → decoder",
        10,
        C["purple"],
        ha="center",
    )
    txt(
        ax,
        85,
        1352,
        "Masked context-token, direct-action regression and action-flow students were evaluated as alternatives.",
        12,
    )
    txt(
        ax,
        85,
        1390,
        (
            "The recovered deterministic motor F is a later architecture; it is not a "
            "renamed CVAE or a reproduced BFM checkpoint."
        ),
        11,
        C["muted"],
    )
    band(
        ax,
        1403,
        1036,
        777,
        397,
        "E",
        "NEXT COMPARISONS  ·  Not established results",
        C["blue"],
        C["blue_bg"],
    )
    box(ax, 1428, 1104, 727, 279, "white", C["blue"], dashed=True)
    txt(
        ax,
        1450,
        1124,
        (
            "Direct scene / goal → desired-reference prediction\nPersistent flow over "
            "qualified reference chunks\nTask RL or bounded action-residual adaptation\n"
            "Camera / partial-map belief with causal memory"
        ),
        14,
        C["blue"],
    )
    txt(
        ax,
        1450,
        1296,
        (
            "Match data, interaction cost and the fixed motor backend.\nScore autonomous "
            "task behavior and motor retention separately."
        ),
        11,
        C["muted"],
    )

    box(ax, 40, 1471, 2140, 110, C["frozen"])
    txt(ax, 64, 1491, "Information boundary", 14, weight="bold")
    txt(
        ax,
        350,
        1493,
        "Public: measured history, explicit request, registered known map and causal localization.",
        12,
    )
    txt(
        ax,
        350,
        1530,
        (
            "Training only: true future reference, motion identity/clock, teacher "
            "actions and qualification records. Freeze status is stage-specific."
        ),
        12,
        C["muted"],
    )
    save(fig, "02_teacher_student_architecture")


def scene_illustration():
    fig, ax = canvas(
        2220,
        1370,
        "Why the scene must make a behavior choice necessary",
        (
            "Conceptual illustrations of the method and its evaluation questions. Robot "
            "poses, routes and response tables below are schematic, not measured "
            "results."
        ),
        "3",
    )
    widths = [
        (40, 686, "A", "Motion → scene: preserve a useful alternative"),
        (766, 686, "B", "Scene → decision: teach complementary responses"),
        (1492, 686, "C", "Goal → behavior: approach, brake and hold"),
    ]
    for x, w, letter, title in widths:
        band(ax, x, 198, w, 658, letter, title, C["teal"], C["teal_bg"])
    txt(ax, 64, 270, "Begin with executed, legal motion alternatives", 13, weight="bold")
    line(ax, [(92, 451), (666, 451)], C["muted"], 2)
    for x, crouch, col in [
        (170, False, C["muted"]),
        (370, True, C["teal"]),
        (575, True, C["purple"]),
    ]:
        robot(ax, x, 451, 0.74, crouch, col)
    txt(ax, 130, 466, "upright", 11, C["muted"])
    txt(ax, 314, 466, "short adaptation", 11, C["teal"])
    txt(ax, 524, 466, "sustained", 11, C["purple"])
    arrow(ax, [(383, 510), (383, 547)], C["teal"])
    beam(ax, 159, 590, 415, 22)
    line(ax, [(93, 749), (666, 749)], C["muted"], 2)
    robot(ax, 247, 749, 0.91, False, C["red"], 0.8)
    robot(ax, 476, 749, 0.91, True, C["teal"])
    txt(ax, 238, 574, "×", 25, C["red"], "bold")
    txt(ax, 469, 617, "clearance", 10, C["teal"])
    txt(
        ax,
        66,
        787,
        (
            "Place a beam around the achieved body envelope.\nCheck the whole "
            "entry–passage–return sequence in physics."
        ),
        12,
    )

    txt(ax, 790, 270, "Long passages change how long adaptation is needed", 13, weight="bold")
    beam(ax, 864, 335, 154, 20)
    beam(ax, 1118, 335, 269, 20)
    line(ax, [(823, 502), (1397, 502)], C["muted"], 2)
    robot(ax, 940, 502, 0.90, True, C["teal"])
    robot(ax, 1223, 502, 0.90, True, C["purple"])
    txt(ax, 884, 520, "short opening", 11, C["teal"])
    txt(ax, 1180, 520, "extended passage", 11, C["purple"])
    box(ax, 790, 563, 638, 171, "white")
    txt(ax, 809, 580, "Illustrative task", 11, weight="bold")
    txt(ax, 1058, 580, "Walk", 11, weight="bold")
    txt(ax, 1162, 580, "Short", 11, weight="bold")
    txt(ax, 1272, 580, "Sustained", 11, weight="bold")
    rows = [
        (617, "Open / easy", ["pass", "pass", "pass"]),
        (653, "Long low passage", ["fail", "fail", "pass"]),
        (689, "Transition-sensitive", ["pass", "fail", "fail"]),
    ]
    for y, name, vals in rows:
        txt(ax, 809, y, name, 11)
        for x, value in zip([1058, 1162, 1290], vals):
            txt(ax, x, y, value, 11, C["teal"] if value == "pass" else C["red"])
    txt(
        ax,
        790,
        769,
        (
            "Complementarity means one fixed response cannot cover all tasks.\nA decision"
            " also needs an observable cue before its legal deadline.\nWAIT is useful "
            "only if a successful later continuation remains."
        ),
        12,
    )

    txt(ax, 1516, 270, "The scene and request must change successful behavior", 13, weight="bold")
    box(ax, 1517, 321, 638, 365, "white")
    for x in range(1538, 2140, 40):
        line(ax, [(x, 341), (x, 660)], C["gray"], 0.5)
    for y in range(341, 661, 40):
        line(ax, [(1537, y), (2137, y)], C["gray"], 0.5)
    ax.add_patch(
        Rectangle((1774, 419), 100, 151, facecolor="#ECDCC2", edgecolor=C["amber"], lw=2, zorder=2)
    )
    txt(ax, 1778, 478, "obstacle", 10, C["amber"])
    ax.add_patch(Circle((1585, 587), 11, facecolor=C["blue"], zorder=4))
    ax.add_patch(Circle((2070, 383), 37, fill=False, edgecolor=C["teal"], lw=2, zorder=4))
    ax.add_patch(Circle((2070, 383), 7, facecolor=C["teal"], zorder=4))
    arrow(ax, [(1585, 587), (1725, 377), (1960, 377), (2030, 382)], C["blue"], width=3)
    arrow(
        ax, [(1585, 587), (1770, 623), (1985, 600), (2050, 429)], C["purple"], dashed=True, width=3
    )
    line(ax, [(1590, 580), (2050, 393)], C["red"], 1.5, dashed=True)
    txt(ax, 1645, 638, "start", 10, C["blue"])
    txt(ax, 2010, 440, "goal region", 10, C["teal"])
    txt(
        ax,
        1536,
        701,
        "Route alternatives are research targets, not demonstrated paths.",
        10,
        C["muted"],
    )
    txt(
        ax,
        1516,
        753,
        (
            "Reach the goal AND reduce speed AND maintain the hold.\nChanging obstacles "
            "should change the route or posture.\nChanging the goal should change the "
            "destination."
        ),
        12,
    )

    band(
        ax,
        40,
        901,
        2140,
        349,
        "D",
        "THE LEARNING SIGNAL  ·  Geometry proposes; execution establishes what can be taught",
        C["blue"],
        C["blue_bg"],
    )
    for x, w, title, body, bg in [
        (
            65,
            620,
            "1  Construct and screen",
            (
                "Conditional learned scene proposals\nTarget clearance + contrast + "
                "visibility\nKeep motion ancestry and construction cost"
            ),
            C["teal_bg"],
        ),
        (
            800,
            620,
            "2  Execute and bind the label",
            (
                "Same task, same prefix, named continuation\nPassage / contact / goal-hold "
                "measurements\nSuccessful task and recovery support checked"
            ),
            C["amber_bg"],
        ),
        (
            1535,
            620,
            "3  Learn and test in closed loop",
            (
                "Fresh learner states + qualified teaching + replay\nTest scene/goal "
                "dependence and motor retention\nSplit by ancestry and task family, not rows"
            ),
            C["purple_bg"],
        ),
    ]:
        box(ax, x, 981, w, 209, bg)
        txt(ax, x + 22, 1001, title, 16, weight="bold")
        txt(ax, x + 22, 1051, body, 13)
    arrow(ax, [(687, 1080), (798, 1080)], C["blue"])
    arrow(ax, [(1422, 1080), (1533, 1080)], C["blue"])
    txt(
        ax,
        65,
        1210,
        (
            "A useful self-learning loop requires executable targets, adequate public "
            "information, and examples from the learner's own state distribution."
        ),
        13,
        C["blue"],
        "bold",
    )
    txt(
        ax,
        45,
        1299,
        (
            "Reported goal-hold pilot: distance ≤ 0.25 m, speed ≤ 0.10 m/s, 50 "
            "consecutive control ticks; retain contact, fall and deadline failures."
        ),
        12,
        C["muted"],
    )
    save(fig, "03_scene_and_learning_illustration")


if __name__ == "__main__":
    overview()
    architecture()
    scene_illustration()
    print(f"Rendered three figures as SVG, PDF and PNG in {OUT}")
