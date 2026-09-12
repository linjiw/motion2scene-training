#!/usr/bin/env python3
"""Inspect and fit a 114D offline baseline using only a portable dataset.

Requires Python and NumPy plus the repository's pure learning modules. No
simulator, original source path, checkpoint or reference bank is opened. File
hashes and complete recorded WAIT continuations are checked locally; this is
not new qualification of the absent raw motion assets or a traversal test.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.hallucination.motion2scene_integer_prefix import (  # noqa: E402
    PREFIX_KEYS,
    paired_prefix_on_ticks,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_multi_option_imitation import (  # noqa: E402
    physical_regret,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_learning import (  # noqa: E402
    SCHEMA as INPUT_SCHEMA,
    TimedScheduledOutcome,
    fit_timed_schedule_policy,
    schedule_layout,
    timed_schedule_teacher,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_timed_schedule_policy import (  # noqa: E402
    expected_feature_names,
    validate_schedule_policy,
)

DATA_SCHEMA = "motion2scene_timed_schedule_portable_dataset_v1"
INPUT_KEYS = {
    "schema_version",
    "feature_names",
    "option_ids",
    "features",
    "command_ticks",
    "capture_elapsed_s",
    "legal_mask",
    "active_before",
}
PREACTION_KEYS = (
    "tick",
    "state",
    "root_pos_w",
    "root_quat_w",
    "features",
    "legal_mask",
    "active_before",
    "measurements",
    "normal_known_mask",
    "capture_elapsed_s",
    "delivered_capture_elapsed_s",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("portable artifact must use a confined relative path")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("portable artifact escapes package through a link")
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    if any(value.dtype.kind not in "biufcUS" for value in values.values()):
        raise ValueError("portable arrays must be pickle free")
    return values


@dataclass(frozen=True)
class PublicScheduleMetadata:
    """Offline adapter for pure helpers; it cannot load a robot/runtime asset.

    ``online_verified`` is the existing pure helper's compatibility gate. Here
    it means the archive declares previously verified schedules; the consumer
    does not independently requalify the absent motion assets. Runtime loading
    still requires its real verified registry and all original asset checks.
    """

    request: dict
    source_registry_id: str

    @property
    def online_verified(self):
        return True

    @property
    def option_ids(self):
        return ("neutral", *(o["option_id"] for o in self.request["options"]))

    @property
    def frame_count(self):
        return self.request["expected_loaded_frames"]


def public_bank(definition, identifier):
    if definition.get("schema") != "motion2scene_timed_option_registry_v1":
        raise ValueError("public declared schedule registry metadata required")
    request = definition["request"]
    bank = PublicScheduleMetadata(request, identifier)
    if (
        type(bank.frame_count) is not int
        or bank.frame_count != 299
        or request["max_entries_per_episode"] != 1
        or len(bank.option_ids) != 7
        or len(set(bank.option_ids)) != 7
        or any(
            type(o["entry_tick"]) is not int
            or type(o["return_tick"]) is not int
            or not 0 < o["entry_tick"] < o["return_tick"] < bank.frame_count - 1
            for o in request["options"]
        )
    ):
        raise ValueError("declared seven schedules within the finite 299-frame horizon required")
    if not np.array_equal(schedule_layout(bank)[0], [15, 50, 70]):
        raise ValueError("114D dataset requires its three declared decision phases")
    return bank


def verify_inventory(package, expected_manifest_sha256=None):
    package = Path(package).resolve()
    digest = sha256(package / "manifest.json")
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256.removeprefix(
        "sha256:"
    ):
        raise ValueError("dataset manifest differs from the expected external digest")
    manifest = read_json(package / "manifest.json")
    if manifest.get("schema") != DATA_SCHEMA:
        raise ValueError("expected portable timed-schedule dataset schema")
    declared = [r["path"] for r in manifest["files"]]
    # Only the two root receipts are exempt; nested manifests remain inventoried.
    actual = {
        str(p.relative_to(package))
        for p in package.rglob("*")
        if p.is_file() and p not in (package / "manifest.json", package / "audit.json")
    }
    if len(set(declared)) != len(declared) or set(declared) != actual:
        raise ValueError("portable file inventory differs")
    for item in manifest["files"]:
        path = relative_path(package, item["path"])
        if (
            sha256(path) != item["sha256"].removeprefix("sha256:")
            or path.stat().st_size != item["size_bytes"]
        ):
            raise ValueError("portable artifact hash/size differs: " + item["path"])
        if (
            path.suffix in (".pkl", ".pt", ".onnx", ".safetensors")
            or path.name == "loaded_reference_bank.npz"
        ):
            raise ValueError("private model/reference assets are outside this portable baseline")
    return manifest, digest


def load_episode(package, episode, bank):
    folder = relative_path(package, episode["directory"])
    data = arrays(folder / "student_inputs.npz")
    count = len(data.get("features", []))
    if (
        set(data) != INPUT_KEYS
        or str(data["schema_version"]) != INPUT_SCHEMA
        or tuple(data["feature_names"]) != expected_feature_names(7)
        or tuple(data["option_ids"]) != bank.option_ids
        or data["features"].shape != (count, 114)
        or not np.isfinite(data["features"]).all()
        or data["legal_mask"].shape != (count, 7)
        or data["legal_mask"].dtype.kind != "b"
        or not np.array_equal(data["command_ticks"], np.arange(1, count + 1))
        or not np.array_equal(data["features"][:, -7:], data["legal_mask"])
        or count != episode["physical_control_frames"]
    ):
        raise ValueError("exact sensor-only 114D names, clock and legality are required")
    active = [bank.option_ids.index(str(option)) for option in data["active_before"]]
    if not np.array_equal(data["features"][:, 100:107], np.eye(7)[active]):
        raise ValueError("recorded active-option feature differs from public command identity")
    alignment = read_json(folder / "sensor_alignment.json")
    if len(alignment["packet_eligible"]) != count:
        raise ValueError("sensor alignment count differs")
    return dict(episode=episode, folder=folder, inputs=data, alignment=alignment)


def episode_history(loaded):
    folder = loaded["folder"]
    raw = arrays(folder / "trajectory.npz")

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"npz_array"}:
                return raw[value["npz_array"]]
            return {key: decode(child) for key, child in value.items()}
        return value

    payload = decode(read_json(folder / "trajectory_metadata.json"))
    sensors = read_json(folder / "sensor_history.json")
    commands = read_json(folder / "commands.json")["records"]
    if len(sensors) != len(commands) or len(sensors) != len(loaded["inputs"]["features"]):
        raise ValueError("physical/sensor/command episode lengths differ")
    observations = []
    for i, (sensor, command) in enumerate(zip(sensors, commands, strict=True)):
        if sensor["tick"] != i + 1 or command["tick"] != i + 1:
            raise ValueError("causal sensor/command tick mismatch")
        observations.append(
            {**sensor, **command, "features": loaded["inputs"]["features"][i].tolist()}
        )
    return payload, observations


def recorded_history_digest(group, payload, observations, tick):
    paired_prefix_on_ticks(payload, payload, tick / 50)
    rows = observations[:tick]
    if len(rows) != tick or rows[-1]["active_before"] != "neutral":
        raise ValueError("complete actual neutral preaction prefix required")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [group, [{key: r[key] for key in PREACTION_KEYS} for r in rows]], sort_keys=True
        ).encode()
    )
    for key in PREFIX_KEYS:
        value = np.ascontiguousarray(np.asarray(payload[key])[:tick])
        digest.update(json.dumps([key, str(value.dtype), value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def verify_teacher(teacher, episodes, bank):
    ids = teacher["episode_ids"]
    if len(ids) != 7 or len(set(ids)) != 7 or any(i not in episodes for i in ids):
        raise ValueError("purported complete teacher group requires seven distinct actual episodes")
    phases, _, entries = schedule_layout(bank)
    branches, histories = [], []
    for index, identifier in enumerate(ids):
        loaded = episodes[identifier]
        episode = loaded["episode"]
        if (
            episode["configured_forced_option_id"] != bank.option_ids[index]
            or episode["mode"] != "forced"
            or episode["split"] != "development"
            or episode["physics_seed"] != teacher["physics_seed"]
            or episode["registry_id"] != teacher["registry_id"]
            or episode["collection_sha256"] != teacher["collection"]["sha256"]
        ):
            raise ValueError("teacher continuation crosses a source, seed, option or data split")
        payload, observations = episode_history(loaded)
        hashes = {}
        for tick in phases.tolist():
            if entries[index] is None or entries[index] >= tick:
                if not all(loaded["alignment"]["packet_eligible"][:tick]):
                    raise ValueError("teacher uses an ineligible causal sensor prefix")
                hashes[tick] = recorded_history_digest(
                    teacher["recorded_history_group"], payload, observations, tick
                )
        a = episode["assessment"]
        branches.append(
            TimedScheduledOutcome(
                episode["original_cell_id"],
                index,
                episode["physics_seed"],
                a["pass"],
                a["costs"]["passage_time_s"],
                a["measurement_admitted"],
                hashes,
                episode["physics_steps"],
            )
        )
        histories.append(observations)
    if len(teacher["targets"]) != len(phases):
        raise ValueError("one teacher target per declared neutral phase required")
    targets = []
    for tick, target in zip(phases.tolist(), teacher["targets"], strict=True):
        packet = histories[0][tick - 1]
        value = timed_schedule_teacher(
            bank,
            branches,
            tick,
            branches[0].prefix_hash_by_tick[tick],
            teacher["physics_seed"],
            np.asarray(packet["legal_mask"], bool),
        )
        if any(target[key] != expected for key, expected in value.items()):
            raise ValueError("stored teacher differs from actual complete WAIT continuations")
        if (
            target["continuation_episode_ids"]
            != [None if i is None else ids[i] for i in value["continuation_option_indices"]]
            or target["recorded_history_sha256"] != branches[0].prefix_hash_by_tick[tick]
            or target["features"] != packet["features"]
        ):
            raise ValueError("teacher feature or continuation identity differs")
        targets.append(
            dict(**value, features=packet["features"], group=teacher["recorded_history_group"])
        )
    return targets


def inspect_package(package, expected_manifest_sha256=None):
    package = Path(package).resolve()
    manifest, digest = verify_inventory(package, expected_manifest_sha256)
    banks = {
        identifier: public_bank(read_json(relative_path(package, row["portable"])), identifier)
        for identifier, row in manifest["registries"].items()
    }
    if len(banks) != 1:
        raise ValueError("one public schedule registry per baseline fit required")
    bank = next(iter(banks.values()))
    episodes = {}
    sources = set()
    for episode in manifest["episodes"]:
        identifier = episode["episode_id"]
        # Separate actual executions can be byte-identical. Source paths are
        # opaque capture IDs here, never opened or replaced by content hashes.
        source = episode["source_trajectory"]["path"]
        if identifier in episodes or source in sources:
            raise ValueError("duplicate episode or repeated physical capture")
        sources.add(source)
        episodes[identifier] = load_episode(package, episode, banks[episode["registry_id"]])
    teachers = read_json(package / "teachers.json")
    if len({tuple(t["recorded_history_group"]) for t in teachers}) != len(teachers):
        raise ValueError("same teacher encounter appears twice")
    targets = [
        row for t in teachers for row in verify_teacher(t, episodes, banks[t["registry_id"]])
    ]
    counts = manifest["counts"]
    if counts["episodes"] != len(episodes) or counts["teacher_decisions"] != len(targets):
        raise ValueError("declared episode/teacher counts differ")
    report = dict(
        schema="motion2scene_portable_schedule_baseline_inspection_v1",
        dataset_manifest_sha256="sha256:" + digest,
        files_verified=len(manifest["files"]),
        episodes=len(episodes),
        sensor_rows=sum(len(e["inputs"]["features"]) for e in episodes.values()),
        teacher_groups=len(teachers),
        teacher_decisions=len(targets),
        declared_physics_steps=counts["physics_steps"],
        option_ids=list(bank.option_ids),
        phase_ticks=schedule_layout(bank)[0].tolist(),
        feature_dimension=114,
        all_inventory_hashes_valid=True,
        wait_continuations_recomputed=True,
        no_external_source_assets_opened=True,
        motion_assets_independently_qualified=False,
        physical_label_scope=(
            "Archived physical assessments; hashes and recorded continuations checked. "
            "Native contact audit remains the separate exporter audit."
        ),
        new_physics_steps=0,
    )
    return bank, targets, report


def fit_targets(bank, targets, l2, *, allow_measured_tie_initialization=False):
    if type(allow_measured_tie_initialization) is not bool:
        raise ValueError("explicit boolean measured-tie initialization setting required")
    if not targets:
        return None, dict(status="no_complete_teacher_groups", policy=None)
    passed = np.asarray([r["pass_labels"] for r in targets], bool)
    times = np.asarray([r["passage_time_s"] for r in targets], float)
    admitted = np.asarray([r["admitted"] for r in targets], bool)
    legal = np.asarray([r["legal_mask"] for r in targets], bool)
    _, supervised = physical_regret(passed, times, admitted, legal)
    ticks = np.asarray([r["phase_tick"] for r in targets])
    phases = schedule_layout(bank)[0]
    coverage = {str(t): int(np.sum(supervised & (ticks == t))) for t in phases}
    if not all(coverage.values()) and not allow_measured_tie_initialization:
        return None, dict(
            status="insufficient_complete_consequential_phase_targets",
            policy=None,
            supervised_phase_counts=coverage,
            excluded_decision_indices=np.flatnonzero(~supervised).tolist(),
        )
    features = np.asarray([r["features"] for r in targets])
    try:
        model, report = fit_timed_schedule_policy(
            bank,
            features,
            expected_feature_names(7),
            ticks,
            passed,
            times,
            admitted,
            legal,
            l2=l2,
            allow_measured_tie_initialization=allow_measured_tie_initialization,
        )
    except ValueError as error:
        if not allow_measured_tie_initialization:
            raise
        return None, dict(
            status="offline_fit_rejected",
            policy=None,
            reason=str(error),
            supervised_phase_counts=coverage,
            allow_measured_tie_initialization=True,
        )
    validate_schedule_policy(model, bank)
    return model, dict(status="offline_fit_complete", fit=report, supervised_phase_counts=coverage)


def run(args):
    allow_measured_tie_initialization = getattr(args, "allow_measured_tie_initialization", False)
    if type(allow_measured_tie_initialization) is not bool:
        raise ValueError("explicit boolean measured-tie initialization setting required")
    if not np.isfinite(args.l2) or args.l2 < 0:
        raise ValueError("finite nonnegative baseline ridge penalty required")
    bank, targets, inspection = inspect_package(args.dataset, args.manifest_sha256)
    out = args.out.resolve()
    if out.is_relative_to(args.dataset.resolve()):
        raise ValueError("baseline outputs must not mutate the immutable dataset")
    out.mkdir(parents=True, exist_ok=False)
    (out / "registration.json").write_text(
        json.dumps(
            dict(
                dataset=str(args.dataset.resolve()),
                inspection=inspection,
                requested_action=args.action,
                l2=args.l2,
                allow_measured_tie_initialization=allow_measured_tie_initialization,
                implementation=dict(
                    driver_sha256="sha256:" + sha256(__file__),
                    pure_modules={
                        name: "sha256:" + sha256(module.__file__)
                        for name, module in sorted(sys.modules.items())
                        if name.startswith("gear_sonic")
                        and getattr(module, "__file__", None)
                        and str(module.__file__).endswith(".py")
                    },
                ),
                scope="Offline public data baseline; no simulation, new motion qualification or traversal claim",
            ),
            indent=2,
            sort_keys=True,
        )
    )
    result = dict(inspection=inspection, status="inspected", policy=None)
    if args.action == "fit":
        model, report = fit_targets(
            bank,
            targets,
            args.l2,
            allow_measured_tie_initialization=allow_measured_tie_initialization,
        )
        result.update(report)
        if model is not None:
            np.savez_compressed(out / "policy.npz", **model)
            # Validate a pickle-free round trip of the exact fitted artifact.
            validate_schedule_policy(arrays(out / "policy.npz"), bank)
            result["policy"] = dict(
                path=str(out / "policy.npz"), sha256="sha256:" + sha256(out / "policy.npz")
            )
    (out / "public_schedule_metadata.json").write_text(
        json.dumps(
            dict(
                request=bank.request,
                source_registry_id=bank.source_registry_id,
                scope="Public metadata only; not independently requalified robot motion assets",
            ),
            indent=2,
            sort_keys=True,
        )
    )
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "fit"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1e-6)
    parser.add_argument(
        "--allow-measured-tie-initialization",
        action="store_true",
        help="Initialize an otherwise unsupported phase from complete measured equal-cost successes.",
    )
    run(parser.parse_args())
