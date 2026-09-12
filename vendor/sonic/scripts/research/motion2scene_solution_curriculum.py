#!/usr/bin/env python3
"""Run the revised analytic screen and fit an execution-conditioned proposal.

This is a development construction experiment with real empty-scene execution
inputs. It acquires no new physical labels and cannot establish traversal gains.
"""

import argparse
import json
from pathlib import Path
import shutil
import time

from bundle_motion2scene_sources import closure
from motion2scene_development_bank import DATA
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_transition_construction import BANK, cases
from motion2scene_uncertainty_learning import numpy_perturbed
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_curriculum import (
    solution_preserving_screen,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_execution_proposal import (
    fit_proposal,
    proposal_condition,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)


def run(out, seed=91821):
    out.mkdir(parents=True, exist_ok=False)
    bank_path = BANK / "result.json"
    offsets_path = DATA / "m2s-refinement-v1/registration.json"
    refs = [artifact(bank_path), artifact(offsets_path), artifact(Path(__file__))]
    snapshot = out / "implementation"
    snapshot.mkdir()
    for path in sorted(closure([Path(__file__)])):
        destination = snapshot / path.relative_to(Path(__file__).resolve().parents[2])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        refs.append(artifact(path))
    offsets = np.asarray(json.loads(offsets_path.read_text())["audit_offsets"])
    if not np.array_equal(offsets[0], np.zeros(4)):
        raise ValueError("nominal audit offset must be first")
    grid = np.array(
        [[s, h] for s in np.linspace(0.12, 0.88, 20) for h in np.linspace(1.105, 1.445, 69)]
    )
    write_new(
        out / "registration.json",
        {
            "schema": "motion2scene_solution_curriculum_v2",
            "role": "development construction and amortization, not independent evaluation",
            "references": refs,
            "sources": [41001, 41002, 41003],
            "grid": grid.tolist(),
            "offsets": offsets.tolist(),
            "margin_m": 0.01,
            "proposal_seed": seed,
            "learned_steps": 400,
            "samples_per_source_per_arm": 64,
            "physical_verification_required": True,
        },
    )
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    torch.set_num_threads(2)
    bank = json.loads(bank_path.read_text())
    contexts, exemplars, all_cases, reports = [], [], [], []
    started = time.monotonic()
    for source in (41001, 41002, 41003):
        tick = time.monotonic()
        case, _, packet, rows = cases(source, bank)
        nominal = numpy_perturbed(
            grid, offsets[:1], case["states"], case["route"], case["yaw"].item()
        )
        nominal_ok = solution_preserving_screen(nominal, 1, 0)["eligible"]
        ids = np.flatnonzero(nominal_ok)
        candidates = grid[ids]
        values = numpy_perturbed(
            candidates, offsets, case["states"], case["route"], case["yaw"].item()
        )
        result = solution_preserving_screen(values, 1, 0)
        accepted = candidates[result["eligible"]]
        achieved = load_reset_capture(
            checked(Path(rows[0]["trajectory"]["path"]), rows[0]["trajectory"]["sha256"])
        )
        approach_history = np.concatenate(
            [
                np.asarray(achieved[key])[:15].reshape(15, -1)
                for key in (
                    "dof_pos",
                    "dof_vel",
                    "root_pos_w",
                    "root_quat_w",
                    "applied_joint_action",
                )
            ],
            axis=1,
        )
        # Sensor specification is the actual mounted ray fan geometry and range.
        sensor = np.r_[
            packet["origin"], np.array([r["direction"] for r in packet["rays"]]).ravel(), 3.0
        ]
        context = proposal_condition(
            case["states"]["d040"],
            case["states"]["neutral"],
            approach_history,
            sensor,
        )
        np.savez_compressed(
            out / f"screen_{source}.npz",
            grid=grid,
            nominal_clearance_m=nominal,
            audit_candidate_indices=ids,
            audit_clearance_m=values,
            accepted=accepted,
            proposal_condition=context,
        )
        reports.append(
            {
                "source": source,
                "grid_proposals": len(grid),
                "nominal_contrasts": len(ids),
                "solution_preserving": len(accepted),
                "all_offset_contrast": int(
                    (result["eligible"] & result["negative_all_offsets_blocked"]).sum()
                ),
                "clearance_queries": int(nominal.size + values.size),
                "search_seconds": time.monotonic() - tick,
                "executed_inputs": [r["trajectory"] for r in rows],
                "observation_timing_screened": False,
                "physical_labels_acquired": 0,
            }
        )
        if len(accepted):
            contexts.append(context)
            exemplars.append(accepted)
            all_cases.append((source, case, context, accepted))
    fit_start = time.monotonic()
    if not exemplars:
        write_new(
            out / "result.json",
            {"rows": reports, "fit": None, "seconds": time.monotonic() - started},
        )
        return
    model, history = fit_proposal(contexts, exemplars, [0.1, 1.1], [0.9, 1.45], seed=seed)
    fit_seconds = time.monotonic() - fit_start
    torch.save(
        {
            "state_dict": model.state_dict(),
            "condition_dim": len(contexts[0]),
            "domain": [[0.1, 1.1], [0.9, 1.45]],
        },
        out / "execution_proposal.pt",
    )
    samples = []
    for source, case, context, accepted in all_cases:
        for arm in ("uniform", "analytic_resample", "learned"):
            tick = time.monotonic()
            if arm == "uniform":
                scenes = np.random.default_rng(seed + source).uniform(
                    [0.1, 1.1], [0.9, 1.45], (64, 2)
                )
            elif arm == "analytic_resample":
                rng = np.random.default_rng(seed + source)
                scenes = accepted[rng.integers(len(accepted), size=64)] + rng.uniform(
                    [-0.02, -0.0025], [0.02, 0.0025], (64, 2)
                )
            else:
                with torch.no_grad():
                    scenes = model.sample(
                        torch.as_tensor(context),
                        64,
                        [0.1, 1.1],
                        [0.9, 1.45],
                        torch.Generator().manual_seed(seed + source),
                    ).numpy()
            proposal_seconds = time.monotonic() - tick
            audit_start = time.monotonic()
            values = numpy_perturbed(
                scenes, offsets, case["states"], case["route"], case["yaw"].item()
            )
            result = solution_preserving_screen(values, 1, 0)
            np.savez_compressed(
                out / f"samples_{source}_{arm}.npz", scenes=scenes, clearance_m=values, **result
            )
            samples.append(
                {
                    "source": source,
                    "arm": arm,
                    "proposed": len(scenes),
                    "eligible": int(result["eligible"].sum()),
                    "eligible_unique_002_progress_001m_height_bins": len(
                        set(
                            map(
                                tuple,
                                np.floor(scenes[result["eligible"]] / [0.02, 0.01]).astype(int),
                            )
                        )
                    ),
                    "clearance_queries": int(values.size),
                    "proposal_seconds": proposal_seconds,
                    "audit_seconds": time.monotonic() - audit_start,
                    "scope": (
                        "in-context geometric amortization only; "
                        "fit/search costs excluded here and reported separately"
                    ),
                }
            )
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": reports,
            "sampling": samples,
            "fit_seconds": fit_seconds,
            "fit_history": history,
            "seconds": time.monotonic() - started,
            "physical_rollout_steps": 0,
            "timing_eligibility_measured": False,
            "scope": (
                "same-context geometric proposal pilot; no downstream learning, "
                "held-out transfer, or physical robustness claim"
            ),
        },
    )
    print(json.dumps({"rows": reports, "sampling": samples, "fit_seconds": fit_seconds}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out)
