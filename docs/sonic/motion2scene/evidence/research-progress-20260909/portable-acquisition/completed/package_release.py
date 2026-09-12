"""Package the complete reconstructed M2 release with existing validated tar helpers."""

import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
import time

ROOT = Path("/home/linjiw/groot-wbc-sonic-sim-trackb")
VALIDATION = Path(__file__).resolve().parent
HELPER = VALIDATION.parent / "m2s-deterministic-hardlink-release-v1/repack.py"
spec = importlib.util.spec_from_file_location("validated_repacker", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
if (
    helper.file_hash(HELPER)
    != "d7b393372933675733d51efa32fe30c298c598bd2f922f68038db1d6e42612f6"
):
    raise ValueError("preserve the independently verified tar helper")


def ref(path):
    return dict(
        path=str(path),
        sha256="sha256:" + helper.file_hash(path),
        size_bytes=path.stat().st_size,
    )


def package(release, reconstruction, destination):
    index = json.loads((release / "release.json").read_text())
    verified = json.loads(reconstruction.read_text())
    if (
        len(index["corpora"]) != 12
        or index["total_episodes"] != 276
        or index["total_physics_steps"] != 328992
    ):
        raise ValueError("complete original twelve-corpus comparison required")
    if verified["fits"] != 36 or verified[
        "release_sha256"
    ] != "sha256:" + helper.file_hash(release / "release.json"):
        raise ValueError("complete reconstruction must bind this exact release")
    if destination.exists() or (VALIDATION / "package_result.json").exists():
        raise FileExistsError("retain previous archive and receipt")
    physical_audit = json.loads((release / "source_audit.json").read_text())
    proposal = physical_audit["geometric_cost_sources"]["candidate_pool_result"]
    if "sha256:" + helper.file_hash(Path(proposal["path"])) != proposal["sha256"]:
        raise ValueError("frozen proposal computation receipt changed")
    with (release / "candidate_pool_result.json").open("xb") as stream:
        stream.write(Path(proposal["path"]).read_bytes())
    history = json.loads((VALIDATION / "history_sources/history.json").read_text())
    corpus = next(
        c for c in index["corpora"] if c["run_id"] == history["charged_to_run_id"]
    )
    manifest = json.loads((release / corpus["dataset"] / "manifest.json").read_text())
    recorded = {
        e["source_trajectory"]["path"]: e["source_trajectory"]["sha256"]
        for e in manifest["episodes"]
    }
    if any(
        recorded.get(r["path"]) != r["sha256"]
        for r in history["reused_bootstrap_trajectories"]
    ):
        raise ValueError(
            "historical bootstrap must remain in its original corpus without reacquisition"
        )
    for row in history["sources"].values():
        if (
            "sha256:" + helper.file_hash(VALIDATION / "history_sources" / row["file"])
            != row["source"]["sha256"]
        ):
            raise ValueError("historical source changed")
    shutil.copytree(VALIDATION / "history_sources", release / "historical_startup")
    for source in (ROOT / "legal").glob("*.txt"):
        target = release / "legal" / source.name
        target.parent.mkdir(exist_ok=True)
        with target.open("xb") as f:
            f.write(source.read_bytes())
    readme = """# Motion2Scene M2 acquisition records

This development release contains the completed four-arm, three-corpus M2 acquisition prefixes:
276 recorded episodes, 328,992 measured physics steps, and 36 historical M0/M1/M2 ridge fits.
These are training acquisitions, not newly executed policies or held-out evaluation episodes.
Every corpus has 21 teacher branches and two actual pre-update student visits.
Keep corpus boundaries and source identities when analyzing repeated layouts and executions.

From this directory, with Python and NumPy installed:

    python -m pip install -r tools/portable_baseline/requirements.txt
    export OPENBLAS_NUM_THREADS=1
    python tools/portable_baseline/scripts/research/motion2scene_acquisition_dataset_baseline.py \\
      --release . --out /absolute/path/to/new-reconstruction

Use an output directory outside this immutable release. Reconstruction validates all dataset
hashes, complete WAIT continuations, historical teacher prefixes and generating student policies.
It reconstructs all 36 fits with the recorded l2=10, measured-tie initialization and originally
audited replay weights. This does not regenerate the full historical physical-gap audit.
Python 3.11.16 and NumPy 2.4.6 were used for the separate reader validation; other numerical
libraries are checked against the same saved coefficients and actual recorded argmax choices.
No simulator, original experiment directory, pretrained checkpoint or full motion bank is needed
for the offline reconstruction. Physical reproduction still requires the separately licensed assets.

Raw pretrained weights and full motion/reference banks are excluded. Recorded reference commands
remain execution data. LICENSE and legal/ retain the repository's existing notices. Upstream
source snapshots preserve their notices; this package does not relicense third-party assets.

The compact tar archive uses direct earlier-file hardlinks for byte-identical files while keeping
every distinct episode path and provenance identity. Filesystem metadata is normalized; data and
JSON bytes are unchanged. GNU tar can extract it into a new directory. Treat extracted files as
immutable: editing a hardlinked file in place changes its aliases. An extractor that duplicates
links may use more storage. No recorded failure or unknown was removed for compression.

historical_startup/ retains the original seed93203 analytic-contrast startup assessment (unknown),
its independently recorded zero physics steps, the separate 1,192-step conservative reservation,
and the failed initial fit. Its seven actual bootstrap captures are already included in that
corpus and counted once; their source identities are checked before packaging.

source_audit.json separates the shared geometric proposal computation from physical cost;
candidate_pool_result.json retains its original receipt. Logical standalone arm-equivalent
query counts are not extra measured computation or physical interaction.

See DATA_CARD.md, source_audit.json, source_plan.json, corpora/*/DATA_CARD.md and the corpus
manifest/audit files for measurement definitions, source versions and interpretation limits.
"""
    with (release / "README.md").open("x") as f:
        f.write(readme)
    paths = [release, *sorted(release.rglob("*"))]
    members = []
    for path in paths:
        name = (
            release.name
            if path == release
            else str(Path(release.name) / path.relative_to(release))
        )
        helper.safe_name(name, release.name)
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("only regular files and directories are supported")
        if path.is_dir():
            members.append(dict(path=name, kind="directory"))
        else:
            if (
                path.suffix in {".pkl", ".pt", ".onnx", ".safetensors", ".pyc"}
                or path.name == "loaded_reference_bank.npz"
            ):
                raise ValueError(
                    "excluded raw asset or generated bytecode in release: " + name
                )
            members.append(
                dict(
                    path=name,
                    kind="file",
                    size_bytes=path.stat().st_size,
                    sha256=helper.file_hash(path),
                )
            )
    (VALIDATION / "archive_members.json").write_text(
        json.dumps(members, indent=2, sort_keys=True) + "\n"
    )
    seen = {}
    start = time.monotonic()
    with destination.open("xb") as raw:
        with gzip.GzipFile(
            fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
            ) as archive:
                for source, row in zip(paths, members, strict=True):
                    if row["kind"] == "directory":
                        archive.addfile(helper.header(row["path"], directory=True))
                        continue
                    identity = (row["size_bytes"], row["sha256"])
                    previous = seen.get(identity)
                    if previous is None:
                        with source.open("rb") as data:
                            archive.addfile(
                                helper.header(row["path"], size=row["size_bytes"]), data
                            )
                        seen[identity] = row["path"]
                    else:
                        archive.addfile(helper.header(row["path"], target=previous))
    audit = helper.validate_archive(destination, release.name, expected_members=members)
    for source, row in zip(paths, members, strict=True):
        if row["kind"] == "file" and helper.file_hash(source) != row["sha256"]:
            raise ValueError("release changed while archiving")
    result = dict(
        schema="motion2scene_acquisition_compact_archive_v1",
        status="complete",
        release=ref(release / "release.json"),
        reconstruction=ref(reconstruction),
        implementation=ref(Path(__file__)),
        tar_helper=ref(HELPER),
        archive=ref(destination),
        member_inventory=ref(VALIDATION / "archive_members.json"),
        member_count=audit["member_count"],
        hardlink_members=audit["hardlink_members"],
        logical_file_bytes=audit["logical_file_bytes"],
        stored_regular_bytes=audit["stored_regular_bytes"],
        archive_wall_seconds=time.monotonic() - start,
        all_source_and_archive_bytes_match=True,
        new_physics_steps=0,
        new_fits=0,
    )
    (VALIDATION / "package_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            package(
                args.release.resolve(),
                args.reconstruction.resolve(),
                args.archive.resolve(),
            )
        )
    )
