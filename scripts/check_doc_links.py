"""Check relative links and heading anchors in the repository's markdown files.

Scans every ``*.md`` outside ``vendor/`` (README.md, CONTRIBUTING.md, docs/**,
experiments/**, scripts/**, paper/**). Web links (any URL scheme) are ignored. Each
remaining link is resolved against the file that contains it, and a broken one is
put in one class:

  external_host  an absolute host path (/home/linjiw/...) or a link that climbs into
                 the sibling research-data checkout (../../../research-data/...);
  vendor_layout  resolves once the doc is re-rooted to where it lived upstream:
                 docs/sonic/ was the docs/ folder of the GR00T-WholeBodyControl checkout
                 vendored at vendor/sonic/, so docs/sonic/X.md is tried as
                 vendor/sonic/docs/X.md (e.g. ../../gear_sonic/... from motion2scene/);
  missing        anything else.

Fragments (``file.md#section`` and ``#section``) pointing at repository markdown are
checked against GitHub-style heading slugs and explicit ``<a id|name>`` anchors, and
broken ones are reported separately.

A path "exists" when git would publish it: tracked, or untracked but not ignored
(so links into the gitignored workspace/ count as missing). Outside a git checkout
the plain filesystem is used.

Usage:
  python scripts/check_doc_links.py                   # summary
  python scripts/check_doc_links.py --list            # every broken link
  python scripts/check_doc_links.py --write-baseline  # record docs/.link-baseline.json
  python scripts/check_doc_links.py --check           # exit 1 on breakage not in the baseline

The baseline compares (file, target) pairs with multiplicity, not line numbers, so
editing a document does not trip ``--check``; moving one does, and the move's commit
re-records the baseline after confirming the per-class counts did not grow.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

BASELINE = "docs/.link-baseline.json"
SCHEMA = "doc_link_baseline_v1"
CLASSES = ("external_host", "vendor_layout", "missing")
SKIP_DIRS = {"vendor", "workspace", ".git", ".claude", "node_modules", "__pycache__"}
# (repo docs prefix, where that tree sits in the vendored upstream checkout)
REROOTS = (("docs/sonic", "vendor/sonic/docs"),)
EXTERNAL_TOPS = {"research-data"}

SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
INLINE_LINK = re.compile(
    r"!?\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\(\s*"
    r"(<[^<>\n]*>|[^\s()<>]*(?:\([^\s()]*\)[^\s()]*)*)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^()]*\)))?\s*\)"
)
REF_DEF = re.compile(r"^\s{0,3}\[(?!\^)[^\]]+\]:\s*(<[^<>\n]*>|\S+)")
HTML_LINK = re.compile(
    r"<(?:a|img|source|video|link)\b[^>]*?\s(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.I
)
HTML_ANCHOR = re.compile(r"<[a-z][^>]*?\s(?:id|name)\s*=\s*[\"']([^\"']+)[\"']", re.I)
ATX = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
SETEXT = re.compile(r"^\s{0,3}(=+|-+)\s*$")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
COMMENT = re.compile(r"<!--.*?-->", re.S)


def skipped(dirname):
    return dirname in SKIP_DIRS or dirname.startswith(".venv")


def published_paths(root):
    """Files git would publish (tracked + untracked-not-ignored) and their parent dirs.

    Returns None outside a git checkout, in which case the filesystem is used."""
    try:
        cmd = ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others"]
        out = subprocess.run([*cmd, "--exclude-standard"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    files = {f for f in out.decode("utf-8", "surrogateescape").split("\0") if f}
    # Deleted-but-still-indexed files are not published.
    files = {f for f in files if (root / f).exists()}
    dirs = {str(p) for f in files for p in PurePosixPath(f).parents if str(p) != "."}
    return files | dirs


class Repo:
    def __init__(self, root, use_git=True):
        self.root = Path(root).resolve()
        self.paths = published_paths(self.root) if use_git else None
        self._anchors = {}

    def exists(self, rel):
        """rel: normalized POSIX path relative to the root ('' is the root itself)."""
        if rel in ("", "."):
            return True
        if self.paths is not None:
            return rel in self.paths
        return (self.root / rel).exists()

    def markdown_files(self):
        if self.paths is not None:
            files = [f for f in self.paths if f.endswith(".md") and (self.root / f).is_file()]
        else:
            files = []
            for d, dirs, names in os.walk(self.root):
                dirs[:] = [x for x in dirs if not skipped(x)]
                rel = Path(d).relative_to(self.root)
                files += [(rel / n).as_posix() for n in names if n.endswith(".md")]
        return sorted(f for f in files if not any(skipped(x) for x in PurePosixPath(f).parts[:-1]))

    def anchors(self, rel):
        if rel not in self._anchors:
            self._anchors[rel] = heading_anchors((self.root / rel).read_text(errors="replace"))
        return self._anchors[rel]


def normalize(parts):
    """Collapse '.' and '..'; returns None when the path climbs above the root."""
    out = []
    for p in parts:
        if p in ("", "."):
            continue
        if p == "..":
            if not out:
                return None
            out.pop()
        else:
            out.append(p)
    return "/".join(out)


def strip_code(lines, blank_spans=True):
    """Yield (line_number, text) outside fenced code, optionally with code spans blanked."""
    fence = None
    for n, line in enumerate(lines, 1):
        m = FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        yield n, CODE_SPAN.sub(lambda s: " " * len(s.group(0)), line) if blank_spans else line


def extract_links(text):
    """Return [(line, target)] for inline, image, reference-definition and HTML links."""
    text = COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    out = []
    for n, line in strip_code(text.splitlines()):
        found = []
        for m in INLINE_LINK.finditer(line):
            found.append((m.start(), m.group(2)))
            # A badge-style [![alt](image)](target) also links its image.
            found += [
                (m.start(1) + i.start(), i.group(2)) for i in INLINE_LINK.finditer(m.group(1))
            ]
        found += [(m.start(), m.group(1)) for m in HTML_LINK.finditer(line)]
        m = REF_DEF.match(line)
        if m:
            found.append((m.start(), m.group(1)))
        for _, target in sorted(found):
            target = target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1].strip()
            if target:
                out.append((n, target))
    return out


def slugify(heading):
    """GitHub-style slug of a heading's rendered text."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # links/images -> text
    text = re.sub(r"<[^>]+>", "", text)  # inline HTML
    text = text.replace("`", "").replace("*", "")
    text = re.sub(r"(?<!\w)_+|_+(?!\w)", "", text)  # _emphasis_, keep snake_case
    text = unicodedata.normalize("NFC", text).strip().lower()
    text = "".join(c for c in text if c.isalnum() or c in " -_" or unicodedata.category(c) == "Mn")
    return text.replace(" ", "-")


def heading_anchors(text):
    anchors, seen = set(), Counter()
    text = COMMENT.sub("", text)
    prev = None
    for _, line in strip_code(text.splitlines(), blank_spans=False):
        anchors.update(m.group(1) for m in HTML_ANCHOR.finditer(CODE_SPAN.sub("", line)))
        m = ATX.match(line)
        title = m.group(2) if m else None
        if title is None and prev and prev.strip() and SETEXT.match(line) and not ATX.match(prev):
            # '---' under a paragraph is a setext h2; a bare '---' after a blank is a rule.
            if not re.match(r"^\s{0,3}([-*_]\s*){3,}$", prev) and not prev.lstrip().startswith(
                ("|", "-", "*", ">")
            ):
                title = prev.strip()
        prev = line
        if title is None:
            continue
        slug = slugify(title)
        anchors.add(f"{slug}-{seen[slug]}" if seen[slug] else slug)
        seen[slug] += 1
    return anchors


def classify(repo, doc, target):
    """Return (status, detail). status: 'ok', 'skip', a CLASSES member, or 'broken_anchor'."""
    if SCHEME.match(target) or target.startswith("//"):
        return "skip", None
    path, _, frag = target.partition("#")
    path = unquote(path.split("?", 1)[0])
    doc_dir = PurePosixPath(doc).parent
    if not path:
        resolved = doc
    elif path.startswith("/"):
        resolved = normalize(path.split("/"))  # GitHub resolves these from the repo root
        if resolved is None or not repo.exists(resolved):
            top_level = resolved and repo.exists(resolved.split("/")[0])
            return ("missing" if top_level else "external_host"), None
    else:
        resolved = normalize(list(doc_dir.parts) + path.split("/"))
        if resolved is None or not repo.exists(resolved):
            if resolved is None or EXTERNAL_TOPS & set(path.split("/")):
                return "external_host", None
            for prefix, upstream in REROOTS:
                if doc.startswith(prefix + "/"):
                    moved = upstream + str(doc_dir)[len(prefix) :]
                    rerooted = normalize([*moved.split("/"), *path.split("/")])
                    if rerooted and repo.exists(rerooted):
                        return "vendor_layout", rerooted
            return "missing", None
    if frag and resolved.endswith(".md") and (repo.root / resolved).is_file():
        want = unquote(frag).lower()
        if want not in {a.lower() for a in repo.anchors(resolved)}:
            return "broken_anchor", resolved
    return "ok", None


def scan(repo):
    files = repo.markdown_files()
    broken, anchors, checked = [], [], 0
    for doc in files:
        text = (repo.root / doc).read_text(errors="replace")
        for line, target in extract_links(text):
            status, detail = classify(repo, doc, target)
            if status == "skip":
                continue
            checked += 1
            row = {"file": doc, "line": line, "target": target}
            if status == "broken_anchor":
                anchors.append(row)
            elif status != "ok":
                row["class"] = status
                if detail:
                    row["resolves_to"] = detail
                broken.append(row)
    counts = {c: sum(r["class"] == c for r in broken) for c in CLASSES}
    counts["broken_anchor"] = len(anchors)
    return {
        "schema": SCHEMA,
        "files_scanned": len(files),
        "links_checked": checked,
        "counts": counts,
        "broken": broken,
        "broken_anchors": anchors,
    }


def compare(report, baseline):
    """Split breakage against the baseline by (file, target) with multiplicity.

    Returns (rows not covered by the baseline, number of baseline rows now fixed)."""
    new, fixed = [], 0
    for field in ("broken", "broken_anchors"):
        allowed = Counter((r["file"], r["target"]) for r in baseline.get(field, []))
        for r in report[field]:
            k = (r["file"], r["target"])
            if allowed[k] > 0:
                allowed[k] -= 1
            else:
                new.append(r)
        fixed += sum(allowed.values())
    return new, fixed


def dump_baseline(report):
    """JSON with one broken link per line, so baseline diffs stay readable."""
    head = [
        f" {json.dumps(k)}: {json.dumps(v)},"
        for k, v in report.items()
        if k not in ("broken", "broken_anchors")
    ]
    body = []
    for field in ("broken", "broken_anchors"):
        rows = [" " + json.dumps(r, ensure_ascii=False) for r in report[field]]
        body.append(f' "{field}": [' + ("\n" + ",\n".join(rows) + "\n ]" if rows else "]"))
    return "{\n" + "\n".join(head) + "\n" + ",\n".join(body) + "\n}\n"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--baseline", type=Path, help=f"default: <root>/{BASELINE}")
    p.add_argument("--list", action="store_true", help="print every broken link")
    p.add_argument("--no-git", action="store_true", help="use the filesystem, not git's file list")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="exit 1 on breakage not in the baseline")
    g.add_argument("--write-baseline", action="store_true")
    a = p.parse_args(argv)
    repo = Repo(a.root, use_git=not a.no_git)
    baseline_path = a.baseline or repo.root / BASELINE
    report = scan(repo)
    print(json.dumps({k: report[k] for k in ("files_scanned", "links_checked", "counts")}))
    if a.list:
        for r in report["broken"]:
            print(f"{r['class']:14} {r['file']}:{r['line']}: {r['target']}")
        for r in report["broken_anchors"]:
            print(f"{'broken_anchor':14} {r['file']}:{r['line']}: {r['target']}")
    if a.write_baseline:
        report = {
            "schema": SCHEMA,
            "generated_by": "scripts/check_doc_links.py --write-baseline",
            **report,
        }
        baseline_path.write_text(dump_baseline(report))
        print(f"wrote {baseline_path}")
    if a.check:
        baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
        new, fixed = compare(report, baseline)
        for r in new:
            print(f"NEW {r.get('class', 'broken_anchor')} {r['file']}:{r['line']}: {r['target']}")
        print(json.dumps({"baseline": baseline.get("counts", {}), "new": len(new), "fixed": fixed}))
        if fixed:
            print(f"{fixed} baseline links now resolve; --write-baseline tightens the baseline")
        return 1 if new else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
