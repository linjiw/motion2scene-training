"""Native robot dependency omission must not silently produce a complete freeze."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))

from motion2scene_freeze_runtime_assets import native_source_assets  # noqa: E402


def fixture(tmp_path):
    native = tmp_path / "gear_sonic/data/assets/robot_description"
    (native / "urdf/g1").mkdir(parents=True)
    (native / "mjcf").mkdir()
    (native / "meshes/g1").mkdir(parents=True)
    (native / "meshes/g1/body.STL").write_bytes(b"synthetic mesh")
    urdf = native / "urdf/g1/main.urdf"
    urdf.write_text(
        '<robot><mesh filename="package://robot_description/meshes/g1/body.STL" /></robot>'
    )
    (native / "mjcf/g1_29dof_rev_1_0.xml").write_text(
        '<mujoco><compiler meshdir="../meshes/g1/" />'
        '<asset><mesh file="body.STL" /></asset></mujoco>'
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "main.usd").write_bytes(b"synthetic USD")
    (cache / ".asset_hash").write_text("synthetic conversion identity")
    return native, urdf, cache


def test_shared_mesh_and_hidden_conversion_identity_are_both_frozen(tmp_path):
    native, _, cache = fixture(tmp_path)
    paths = native_source_assets(tmp_path, cache)
    assert len(paths) == 5
    assert native / "meshes/g1/body.STL" in paths
    assert cache / ".asset_hash" in paths


def test_missing_native_mesh_is_not_omitted(tmp_path):
    native, _, cache = fixture(tmp_path)
    (native / "meshes/g1/body.STL").unlink()
    with pytest.raises(FileNotFoundError):
        native_source_assets(tmp_path, cache)


@pytest.mark.parametrize(
    "content, error",
    [
        ('<robot><mesh filename="https://example.invalid/body.STL" /></robot>', "remote"),
        ('<robot><include filename="another.urdf" /></robot>', "include"),
    ],
)
def test_unresolved_remote_and_new_include_dependencies_are_rejected(tmp_path, content, error):
    _, urdf, cache = fixture(tmp_path)
    urdf.write_text(content)
    with pytest.raises(ValueError, match=error):
        native_source_assets(tmp_path, cache)


def test_composed_usd_dependencies_include_external_geometry_and_label_mdl(tmp_path):
    from motion2scene_freeze_runtime_assets import validate_usd_dependencies

    _, _, cache = fixture(tmp_path)
    external = tmp_path / "outside_geometry.usda"
    external.write_text("synthetic outside payload")
    receipt = dict(
        layers=[str(cache / "main.usd"), str(external)], assets=[], unresolved=["OmniPBR.mdl"]
    )
    paths, audit = validate_usd_dependencies(cache, receipt)
    assert external in paths and cache / "main.usd" in paths
    assert audit["renderer_mdl_dependencies_outside_scope"] == ["OmniPBR.mdl"]
    assert audit["renderer_reproduction_claim"] is False
    receipt["unresolved"].append("missing_body.usd")
    with pytest.raises(ValueError, match="non-MDL"):
        validate_usd_dependencies(cache, receipt)


def test_composed_dependency_audit_requires_actual_root_layer(tmp_path):
    from motion2scene_freeze_runtime_assets import validate_usd_dependencies

    _, _, cache = fixture(tmp_path)
    with pytest.raises(ValueError, match="root layer"):
        validate_usd_dependencies(cache, dict(layers=[], assets=[], unresolved=[]))


def test_editable_source_identity_binds_commit_modified_new_and_deleted_contents(tmp_path):
    import subprocess

    from motion2scene_freeze_runtime_assets import editable_source_identity

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    source = tmp_path / "src"
    source.mkdir()
    (source / "changed.py").write_text("original")
    (source / "removed.py").write_text("removed")
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    clean = editable_source_identity(source)
    assert clean["git_commit"] and clean["dirty_files"] == []
    (source / "changed.py").write_text("updated")
    (source / "removed.py").unlink()
    (source / "new.py").write_text("new source")
    dirty = editable_source_identity(source)
    assert dirty["git_commit"] == clean["git_commit"]
    entries = {row["relative_path"]: row for row in dirty["dirty_files"]}
    assert set(entries) == {"src/changed.py", "src/new.py", "src/removed.py"}
    assert entries["src/removed.py"] == dict(relative_path="src/removed.py", deleted=True)
    first_hash = entries["src/changed.py"]["sha256"]
    (source / "changed.py").write_text("another update")
    next_identity = editable_source_identity(source)
    assert (
        next(
            r["sha256"]
            for r in next_identity["dirty_files"]
            if r["relative_path"] == "src/changed.py"
        )
        != first_hash
    )


def test_direct_url_keeps_editable_target_instead_of_metadata_location(tmp_path):
    from motion2scene_freeze_runtime_assets import bind_editable_sources

    source = tmp_path / "editable"
    source.mkdir()
    (source / "module.py").write_text("fixture")
    inventory = {
        "packages": [
            dict(
                name="package",
                version="1",
                location="/metadata/site-packages",
                direct_url={"url": source.as_uri(), "dir_info": {"editable": True}},
            )
        ]
    }
    bound = bind_editable_sources(inventory)
    assert bound["packages"][0]["editable_source_identity"] == str(source)
    identity = bound["editable_sources"][str(source)]
    assert identity["git_commit"] is None and len(identity["non_git_source_files"]) == 1
    inventory["packages"][0]["direct_url"]["url"] = "https://example.invalid/source"
    with pytest.raises(ValueError, match="local source"):
        bind_editable_sources(inventory)
