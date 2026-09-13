#!/usr/bin/env python3
"""Summarize the teacher-8192-500 review: per-motion CSV, paired CSV, summary.json, index.html."""

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_common import (  # noqa: E402
    ARMS, DEFAULT_REVIEW, HISTORY_DEV, ORIGINAL_DATA_QUALIFICATION, PACKET, ledger_by_key,
    load_metrics, metrics_dir, sha256,
)

PAIRS = [("trained500", "release"), ("trained500", "previous8000"), ("previous8000", "release")]


def sign_test(improved, regressed):
    m = improved + regressed
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(improved, regressed) + 1)) / 2**m
    return min(1.0, 2 * tail)


class Record:
    """Native-trajectory metrics for one (arm, split, motion); pose file used only for terms."""

    def __init__(self, review, arm, split, key, metrics_row):
        folder = metrics_dir(review, arm, split)
        native = np.load(folder / f"{key}.npz")
        ref = native["reference"].astype(np.float64)
        trk = native["tracked"].astype(np.float64)
        self.T = len(ref)
        self.completed = not bool(metrics_row["terminated"])
        self.progress = float(metrics_row["progress"])
        pose_path = folder / f"{key}.pose.npz"
        self.failure_terms, self.n = [], self.T + 1
        if pose_path.exists():
            pose = np.load(pose_path)
            self.n = int(pose["motion_num_steps"])
            time_out = {str(t) for t in pose["time_out_terms"]}
            valid = self.T if self.completed else min(int(round(self.progress * self.n)), self.T)
            if not self.completed and valid < self.T:
                self.failure_terms = [f[5:] for f in pose.files if f.startswith("term_")
                                      and f[5:] not in time_out and bool(pose[f][valid])]
        self.valid = self.T if self.completed else min(int(round(self.progress * self.n)), self.T)
        self.eg = np.linalg.norm(trk - ref, axis=-1).mean(1) * 1000
        self.el = np.linalg.norm((trk - trk[:, :1]) - (ref - ref[:, :1]), axis=-1).mean(1) * 1000
        self.xy = np.linalg.norm(trk[:, 0, :2] - ref[:, 0, :2], axis=-1)
        self.native_mpjpe_g = float(metrics_row["mpjpe_g"])

    def row(self):
        v = self.valid
        return {
            "completed": self.completed,
            "native_progress": round(self.progress, 6),
            "valid_frames": v,
            "motion_num_steps": self.n,
            "survival_s": round(v * 0.02, 3),
            "failure_time_s": "" if self.completed else round(v * 0.02, 3),
            "failure_terms": "+".join(self.failure_terms),
            "prefix_mpjpe_g_mm": round(float(self.eg[:v].mean()), 2) if v else "",
            "prefix_mpjpe_l_mm": round(float(self.el[:v].mean()), 2) if v else "",
            "root_xy_end_m": round(float(self.xy[v - 1]), 4) if v else "",
            "root_xy_max_m": round(float(self.xy[:v].max()), 4) if v else "",
            "native_mpjpe_g_all_frames_mm": round(self.native_mpjpe_g, 2),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args()
    review = args.review
    lock = json.loads((review / "evaluation-lock.json").read_text())
    ledger = ledger_by_key()
    featured = {f["motion_key"]: f for f in lock["featured_videos"]}
    render = json.loads((review / "render-receipt.json").read_text()) if (
        review / "render-receipt.json").exists() else {"motions": {}}

    records, runs_present = {}, {}
    for split, spec in lock["splits"].items():
        for arm in ARMS:
            path = metrics_dir(review, arm, split) / "metrics_eval.json"
            if not path.exists():
                continue
            metrics = load_metrics(path)
            runs_present[f"{arm}/{split}"] = {"metrics_sha256": sha256(path)}
            for index, key in enumerate(metrics["motion_keys"]):
                row = {name: metrics[name][index] for name in ("terminated", "progress", "mpjpe_g")}
                records[(arm, split, str(key))] = Record(review, arm, split, str(key), row)

    qualification = {}
    if ORIGINAL_DATA_QUALIFICATION.exists():
        for row in csv.DictReader(ORIGINAL_DATA_QUALIFICATION.open()):
            qualification[(row["arm"], f"hindsight_{row['motion_id']}")] = row["completed"] == "True"
    history = {}
    for label, path in HISTORY_DEV.items():
        if path.exists():
            metrics = load_metrics(path)
            history[label] = {str(k): not t for k, t in zip(metrics["motion_keys"], metrics["terminated"])}

    per_motion = []
    for (arm, split, key), rec in sorted(records.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        meta = ledger.get(key, {})
        per_motion.append({
            "arm": arm, "split": split, "motion_key": key,
            "featured_stratum": featured.get(key, {}).get("stratum", ""),
            "category": meta.get("category", ""), "route": meta.get("route", ""),
            "reference_seconds": round(meta.get("frames", 0) / meta.get("fps", 30.0), 2),
            **rec.row(),
            "video": render["motions"].get(key, {}).get("video", ""),
        })
    fields = list(per_motion[0].keys()) if per_motion else []
    with (review / "per-motion-results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_motion)

    paired_rows, aggregates, paired_summary = [], {}, {}
    for split, spec in lock["splits"].items():
        for arm in ARMS:
            recs = [records[(arm, split, k)] for k in spec["motion_keys"] if (arm, split, k) in records]
            if not recs:
                continue
            frames = sum(r.valid for r in recs)
            aggregates[f"{split}/{arm}"] = {
                "motions": len(recs),
                "expected_motions": len(spec["motion_keys"]),
                "completed": sum(r.completed for r in recs),
                "mean_progress": round(float(np.mean([r.progress for r in recs])), 6),
                "median_survival_s": round(float(np.median([r.valid * 0.02 for r in recs])), 3),
                "prefix_mpjpe_g_mm_frame_weighted": round(float(sum(r.eg[: r.valid].sum() for r in recs) / max(frames, 1)), 2),
                "prefix_mpjpe_l_mm_frame_weighted": round(float(sum(r.el[: r.valid].sum() for r in recs) / max(frames, 1)), 2),
                "mean_root_xy_end_m": round(float(np.mean([r.xy[r.valid - 1] for r in recs if r.valid])), 4),
                "failure_terms": {t: sum(t in r.failure_terms for r in recs)
                                  for t in sorted({t for r in recs for t in r.failure_terms})},
            }
        for new, old in PAIRS:
            keys = [k for k in spec["motion_keys"] if (new, split, k) in records and (old, split, k) in records]
            if not keys:
                continue
            counts = {"both_complete": 0, "improved": 0, "regressed": 0, "neither": 0}
            deltas = []
            for key in keys:
                a, b = records[(old, split, key)], records[(new, split, key)]
                outcome = ("both_complete" if a.completed and b.completed else "improved" if b.completed
                           else "regressed" if a.completed else "neither")
                counts[outcome] += 1
                c = min(a.valid, b.valid)
                ea = float(a.eg[:c].mean()) if c else float("nan")
                eb = float(b.eg[:c].mean()) if c else float("nan")
                if c:
                    deltas.append(eb - ea)
                paired_rows.append({
                    "split": split, "motion_key": key, "new": new, "old": old, "outcome": outcome,
                    "progress_old": round(a.progress, 4), "progress_new": round(b.progress, 4),
                    "common_prefix_frames": c,
                    "common_prefix_mpjpe_g_old_mm": round(ea, 2), "common_prefix_mpjpe_g_new_mm": round(eb, 2),
                })
            paired_summary[f"{split}/{new}_vs_{old}"] = {
                **counts, "motions": len(keys),
                "sign_test_p_two_sided": round(sign_test(counts["improved"], counts["regressed"]), 4),
                "median_common_prefix_mpjpe_g_delta_mm": round(float(np.median(deltas)), 2) if deltas else None,
                "note": "descriptive; single seed; development set inspected repeatedly",
            }
    if paired_rows:
        with (review / "paired-results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
            writer.writeheader()
            writer.writerows(paired_rows)

    gate_path = review / "eval/previous8000/development/check.json"
    gate = json.loads(gate_path.read_text()).get("reproduction_gate") if gate_path.exists() else None
    history_rows = [{"label": label, "completed": sum(v.values()), "motions": len(v),
                     "source": str(HISTORY_DEV[label]), "note": "reused; different host"}
                    for label, v in history.items()]
    checks = {}
    for check in sorted(review.glob("eval/*/*/check.json")):
        data = json.loads(check.read_text())
        rows = data.get("per_motion", {}).values()
        checks[data["run"]] = {
            "accepted": data["accepted"], "errors": len(data["errors"]), "warnings": len(data["warnings"]),
            "robot_fk_max_mm": max((r.get("robot_fk_max_mm") or 0 for r in rows), default=None),
            "ghost_fk_max_mm": max((r.get("ghost_fk_max_mm") or 0 for r in rows), default=None),
            "wall_seconds": data.get("receipt", {}).get("wall_seconds"),
            "gpu_total_mib_peak": data.get("receipt", {}).get("gpu_total_mib_peak"),
        }
    summary = {
        "lock_created_utc": lock["created_utc"], "arms": lock["arms"], "seed": lock["seed"],
        "runs_present": runs_present, "run_checks": checks, "aggregates": aggregates,
        "paired": paired_summary, "reproduction_gate": gate, "history_development": history_rows,
        "featured": list(featured), "videos_rendered": sorted(render["motions"]),
        "caveats": [
            "Single seed (91231); startup domain randomization stays on and depends on env index.",
            "Complete = no native termination; there is no root-XY guard, so check drift.",
            "Prefix metrics stop at the first termination; native all-frame MPJPE includes reset replays.",
            "Train-split tiers come from original-data outcomes; the repaired references differ.",
            "Reference tracking only; not scene or navigation success.",
        ],
    }
    (review / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (review / "index.html").write_text(build_html(summary, per_motion, featured, render, qualification, history, ledger))
    print(json.dumps({"aggregates": aggregates, "paired": paired_summary}, indent=2))


def chip(completed, progress, failure_time, terms):
    if completed:
        return '<span class="chip ok">complete</span>'
    return (f'<span class="chip bad">failed {failure_time}s ({progress:.0%})</span>'
            f'<span class="terms">{html.escape(terms)}</span>')


def build_html(summary, per_motion, featured, render, qualification, history, ledger):
    esc = html.escape
    by_key = {}
    for row in per_motion:
        by_key.setdefault(row["motion_key"], {})[row["arm"]] = row
    arm_headers = "".join(f'<th style="color:{ARMS[a]["color"]}">{esc(ARMS[a]["label"])}</th>' for a in ARMS)
    agg_rows = []
    for name, agg in summary["aggregates"].items():
        split, arm = name.split("/")
        agg_rows.append(
            f'<tr><td>{split}</td><td style="color:{ARMS[arm]["color"]}"><b>{esc(ARMS[arm]["label"])}</b></td>'
            f'<td class="n">{agg["completed"]}/{agg["motions"]}</td><td class="n">{agg["mean_progress"]:.1%}</td>'
            f'<td class="n">{agg["median_survival_s"]:.2f}</td><td class="n">{agg["prefix_mpjpe_g_mm_frame_weighted"]:.0f}</td>'
            f'<td class="n">{agg["prefix_mpjpe_l_mm_frame_weighted"]:.0f}</td><td class="n">{agg["mean_root_xy_end_m"]:.2f}</td>'
            f'<td>{esc(", ".join(f"{k} {v}" for k, v in agg["failure_terms"].items()))}</td></tr>')
    pair_rows = "".join(
        f'<tr><td>{esc(name)}</td><td class="n">{p["both_complete"]}</td><td class="n">{p["improved"]}</td>'
        f'<td class="n">{p["regressed"]}</td><td class="n">{p["neither"]}</td><td class="n">{p["sign_test_p_two_sided"]}</td>'
        f'<td class="n">{p["median_common_prefix_mpjpe_g_delta_mm"]}</td></tr>'
        for name, p in summary["paired"].items())
    gate = summary["reproduction_gate"]
    gate_html = ("<p>Not run yet.</p>" if not gate else
                 f'<p>previous8000 / development on this host: <b>{gate["this_completed"]}/20</b>, '
                 f'mean progress {gate["this_mean_progress"]:.4f}; recorded: {gate["recorded_completed"]}/20, '
                 f'{gate["recorded_mean_progress"]:.4f}. Verdict: <b>{esc(gate["verdict"])}</b> '
                 f'(per-motion outcome agreement {gate["per_motion_outcome_agreement"]}/20).</p>')
    hist = "".join(f'<tr><td>{esc(h["label"])}</td><td class="n">{h["completed"]}/{h["motions"]}</td>'
                   f'<td>{esc(h["note"])}</td></tr>' for h in summary["history_development"])
    cards = []
    for key, info in featured.items():
        rows = by_key.get(key, {})
        video = render["motions"].get(key, {}).get("video")
        media = (f'<video controls preload="none" poster="posters/{key}.jpg" src="{esc(video)}"></video>'
                 if video else '<div class="missing">video not rendered</div>')
        outcomes = "".join(
            f'<div><span class="dot" style="background:{ARMS[a]["color"]}"></span>{esc(ARMS[a]["short"])}: '
            f'{chip(r["completed"], r["native_progress"], r["failure_time_s"], r["failure_terms"])}'
            f' <span class="muted">body err {r["prefix_mpjpe_g_mm"]} mm, end XY {r["root_xy_end_m"]} m</span></div>'
            for a, r in ((a, rows[a]) for a in ARMS if a in rows))
        prompts = " → ".join(info.get("prompts", []))
        cards.append(
            f'<section class="card"><h3>{esc(key)} <span class="badge">{esc(info["split"])}</span> '
            f'<span class="badge">{esc(info["category"])}</span></h3>'
            f'<p class="muted">{esc(info["stratum"])} · {info["seconds"]:.1f} s · route {esc(info["route"])}'
            f'{" · " + esc(", ".join(info["reasons"])) if info["reasons"] else ""}</p>'
            f'<p>{esc(prompts)}</p>{media}{outcomes}</section>')
    table_rows = []
    for key in sorted(by_key, key=lambda k: (by_key[k][next(iter(by_key[k]))]["split"], k)):
        rows = by_key[key]
        first = rows[next(iter(rows))]
        cells = []
        for arm in ARMS:
            r = rows.get(arm)
            cells.append("<td>–</td>" if r is None else
                         f'<td data-v="{r["native_progress"]}">{"✓" if r["completed"] else "✗ " + str(r["failure_time_s"]) + "s"}'
                         f' <span class="muted">{r["prefix_mpjpe_g_mm"]}mm</span></td>')
        if first["split"] == "development":
            context = " ".join(("✓" if done.get(key) else "✗") for done in history.values())
            context = f'<span title="{esc(", ".join(history))}">{context}</span>'
        else:
            context = " / ".join(("✓" if qualification.get((a, key)) else "✗") for a in ("release", "trained")
                                 if (a, key) in qualification)
        video = render["motions"].get(key, {}).get("video")
        link = f'<a href="{esc(video)}">mp4</a>' if video else ""
        star = "★ " if key in featured else ""
        table_rows.append(
            f'<tr><td>{star}{esc(key)}</td><td>{first["split"]}</td><td>{esc(first["category"])}</td>'
            f'<td class="n">{first["reference_seconds"]}</td>{"".join(cells)}<td>{context}</td><td>{link}</td></tr>')
    checks = "".join(
        f'<tr><td>{esc(run)}</td><td>{"yes" if c["accepted"] else "NO"}</td><td class="n">{c["errors"]}/{c["warnings"]}</td>'
        f'<td class="n">{c["robot_fk_max_mm"]}</td><td class="n">{c["ghost_fk_max_mm"]}</td>'
        f'<td class="n">{c["wall_seconds"]}</td><td class="n">{c["gpu_total_mib_peak"]}</td></tr>'
        for run, c in summary["run_checks"].items())
    caveats = "".join(f"<li>{esc(c)}</li>" for c in summary["caveats"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Teacher 8192x500 review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font:14px/1.45 system-ui,sans-serif;margin:0;padding:24px 16px;background:#f6f7f9;color:#1d232b}}
main{{max-width:1280px;margin:0 auto}} h1{{margin:0 0 4px}} h2{{margin-top:32px}}
table{{border-collapse:collapse;background:#fff;width:100%}} th,td{{padding:5px 8px;border-bottom:1px solid #e3e6ea;text-align:left;vertical-align:top}}
th{{cursor:pointer;background:#eef1f5;position:sticky;top:0}} td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.wrap{{overflow-x:auto}} .muted{{color:#66707c;font-size:12px}} .badge{{font-size:12px;background:#e4e9f0;border-radius:4px;padding:1px 6px;font-weight:normal}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:16px}}
.card{{background:#fff;border:1px solid #e3e6ea;border-radius:8px;padding:12px}} .card h3{{margin:0}}
video{{width:100%;background:#000;border-radius:4px}} .chip{{border-radius:4px;padding:0 6px;font-size:12px;color:#fff}}
.ok{{background:#2e8b57}} .bad{{background:#c62828}} .terms{{font-size:12px;margin-left:4px;color:#8a1c1c}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:5px;margin-right:6px}} .missing{{padding:40px;text-align:center;background:#eee}}
@media (max-width:640px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Teacher 8192 × 500 — tracking review</h1>
<p class="muted">Locked {esc(summary["lock_created_utc"])} · seed {summary["seed"]} · native Isaac Lab evaluation, MuJoCo mesh replay of captured states.
Gray robot = simulated policy; translucent blue = reference. Red robot = frozen at the failure step.</p>
<ul>{caveats}</ul>
<h2>Headline</h2><div class="wrap"><table><tr><th>split</th><th>checkpoint</th><th>completed</th><th>mean progress</th><th>median survival (s)</th>
<th>prefix body err (mm)</th><th>prefix local err (mm)</th><th>mean end root XY (m)</th><th>failure terms</th></tr>{"".join(agg_rows)}</table></div>
<h2>Paired outcomes</h2><div class="wrap"><table><tr><th>new vs old</th><th>both</th><th>improved</th><th>regressed</th><th>neither</th><th>sign test p</th><th>median Δ common-prefix err (mm)</th></tr>{pair_rows}</table></div>
<h2>Reproduction check and development history</h2>{gate_html}
<div class="wrap"><table><tr><th>earlier evaluation (reused)</th><th>completed</th><th>note</th></tr>{hist}</table></div>
<h2>Featured motions (fixed before evaluation)</h2><div class="cards">{"".join(cards)}</div>
<h2>All motions</h2><p class="muted">Cell: ✓ complete / ✗ failure time, then prefix body error. Context: development = earlier checkpoints
({esc(", ".join(history))}); train = original-data release / previous-teacher completion.</p>
<div class="wrap"><table id="all"><thead><tr><th>motion</th><th>split</th><th>category</th><th>ref s</th>{arm_headers}<th>context</th><th>video</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div>
<h2>Run checks</h2><div class="wrap"><table><tr><th>run</th><th>accepted</th><th>errors/warnings</th><th>robot FK max mm</th><th>ghost FK max mm</th><th>wall s</th><th>GPU MiB peak (whole GPU)</th></tr>{checks}</table></div>
<p class="muted">Files: evaluation-lock.json, video-selection-lock.json, per-motion-results.csv, paired-results.csv, summary.json, render-receipt.json, eval/&lt;arm&gt;/&lt;split&gt;/.</p>
</main><script>
document.querySelectorAll('#all th').forEach((th,i)=>th.onclick=()=>{{const b=th.closest('table').tBodies[0];
const rows=[...b.rows];const dir=th.dataset.dir=th.dataset.dir==='1'?'-1':'1';
rows.sort((x,y)=>{{const a=x.cells[i].dataset.v??x.cells[i].innerText,c=y.cells[i].dataset.v??y.cells[i].innerText;
const na=parseFloat(a),nc=parseFloat(c);return (isNaN(na)||isNaN(nc)?a.localeCompare(c):na-nc)*dir;}});rows.forEach(r=>b.appendChild(r));}});
</script></body></html>"""


if __name__ == "__main__":
    main()
