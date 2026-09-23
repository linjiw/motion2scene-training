import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_doc_links.py"
spec = importlib.util.spec_from_file_location("check_doc_links", SCRIPT)
links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(links)


def write(root, rel, text=""):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def repo(tmp_path):
    write(
        tmp_path,
        "docs/a.md",
        '# Title\n\n## Section two\n\n## Section two\n\nIntro\n---\n\n<a id="custom"></a>\n',
    )
    write(tmp_path, "vendor/sonic/gear_sonic/foo.py")
    write(tmp_path, "vendor/sonic/notes.md", "[never scanned](missing-in-vendor.md)\n")
    write(
        tmp_path,
        "docs/sonic/motion2scene/X.md",
        "\n".join(
            [
                "[vendored](../../gear_sonic/foo.py)",
                "[not vendored](../../gear_sonic/bar.py)",
                "[research data](../../../research-data/run/fig.png)",
                "[sibling](../../a.md#section-two-1)",
                "[setext](../../a.md#intro)",
                "[custom anchor](../../a.md#custom)",
                "[bad anchor](../../a.md#section-3)",
            ]
        )
        + "\n",
    )
    write(
        tmp_path,
        "README.md",
        "\n".join(
            [
                "# Top heading",
                "[ok](docs/a.md) [web](https://example.com/x.md) [mail](mailto:a@b.c)",
                "[self](#top-heading) [bad self](#nope) [climb](../outside.md)",
                "[host](/home/linjiw/research-data/x.png)",
                "[root relative](/docs/a.md) [root missing](/docs/nope.md)",
                '[spaces](<docs/a.md> "title")',
                '[![badge](/home/linjiw/fig.png)](docs/a.md) <img src="docs/nope.png">',
                "`[code span](code.md)`",
                "```",
                "[fenced](fenced.md)",
                "```",
                "[^1]: Footnote Author et al. not a link",
                "[ref]: docs/a.md",
                "[ref2]: nope.md",
                "<!-- [commented](commented.md) -->",
            ]
        )
        + "\n",
    )
    return tmp_path


def by_target(report):
    rows = {r["target"]: r["class"] for r in report["broken"]}
    rows.update({r["target"]: "broken_anchor" for r in report["broken_anchors"]})
    return rows


def test_classifies_broken_links_and_anchors(repo):
    report = links.scan(links.Repo(repo, use_git=False))
    assert by_target(report) == {
        "../../gear_sonic/foo.py": "vendor_layout",
        "../../gear_sonic/bar.py": "missing",
        "../../../research-data/run/fig.png": "external_host",
        "../../a.md#section-3": "broken_anchor",
        "#nope": "broken_anchor",
        "/home/linjiw/research-data/x.png": "external_host",
        "../outside.md": "external_host",
        "/docs/nope.md": "missing",
        "/home/linjiw/fig.png": "external_host",
        "docs/nope.png": "missing",
        "nope.md": "missing",
    }
    vendored = next(r for r in report["broken"] if r["class"] == "vendor_layout")
    assert vendored["resolves_to"] == "vendor/sonic/gear_sonic/foo.py"
    assert vendored["file"] == "docs/sonic/motion2scene/X.md" and vendored["line"] == 1
    assert report["counts"] == {
        "external_host": 4,
        "vendor_layout": 1,
        "missing": 4,
        "broken_anchor": 2,
    }
    assert report["files_scanned"] == 3  # vendor/ is skipped


def test_github_slugs():
    assert links.slugify("2. Where each layer stands") == "2-where-each-layer-stands"
    assert (
        links.slugify("Roadmap — September 23, 2026 (revised)")
        == "roadmap--september-23-2026-revised"
    )
    assert links.slugify("`foo_bar` and **bold** [link](x.md)") == "foo_bar-and-bold-link"
    assert links.slugify("Tracker: 8192×500 fine-tune") == "tracker-8192500-fine-tune"
    anchors = links.heading_anchors("# A\n## A\n```\n# not a heading\n```\nB\n===\n| t |\n---\n")
    assert anchors == {"a", "a-1", "b"}


def test_check_mode_only_fails_on_new_breakage(repo, capsys):
    base = repo / "docs/.link-baseline.json"
    assert links.main(["--root", str(repo), "--no-git", "--write-baseline"]) == 0
    saved = json.loads(base.read_text())
    assert saved["schema"] == links.SCHEMA and saved["counts"]["missing"] == 4
    assert links.main(["--root", str(repo), "--no-git", "--check"]) == 0

    # Shifting lines does not count as new breakage.
    doc = repo / "docs/sonic/motion2scene/X.md"
    doc.write_text("New first line.\n\n" + doc.read_text())
    assert links.main(["--root", str(repo), "--no-git", "--check"]) == 0

    # A repeat of an already-baselined broken link is new (multiplicity).
    doc.write_text(doc.read_text() + "[again](../../gear_sonic/bar.py)\n")
    assert links.main(["--root", str(repo), "--no-git", "--check"]) == 1
    assert "NEW missing docs/sonic/motion2scene/X.md:" in capsys.readouterr().out

    # Fixing links passes and reports how many baseline rows are now fixed.
    doc.write_text("[fixed](../../a.md)\n")
    assert links.main(["--root", str(repo), "--no-git", "--check"]) == 0
    assert '"fixed": 4' in capsys.readouterr().out  # X.md's 3 broken links + 1 anchor

    # New anchor breakage also fails.
    write(repo, "docs/b.md", "[x](a.md#missing-section)\n")
    assert links.main(["--root", str(repo), "--no-git", "--check"]) == 1


def test_missing_baseline_fails_check_when_links_are_broken(repo):
    assert links.main(["--root", str(repo), "--no-git", "--check"]) == 1


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_mode_treats_ignored_paths_as_missing(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "workspace/\n")
    write(tmp_path, "workspace/run/results.txt", "x")
    write(tmp_path, "docs/r.md", "[ignored](../workspace/run/results.txt)\n[dir](../docs/)\n")
    assert by_target(links.scan(links.Repo(tmp_path))) == {
        "../workspace/run/results.txt": "missing"
    }
    assert by_target(links.scan(links.Repo(tmp_path, use_git=False))) == {}
