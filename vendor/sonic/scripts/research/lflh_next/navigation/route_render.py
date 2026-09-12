"""Render new per-motion route candidates without relocating old predictions."""

import json
from pathlib import Path

from scripts.research.lflh_next.navigation import render
from scripts.research.lflh_next.navigation.route_study import OUT


def main():
    render.OUT = OUT
    render.main()
    path = OUT / "viewer/data.js"
    payload = json.loads(path.read_text().removeprefix("const DATA=").removesuffix(";"))
    payload["routeSpec"] = payload.pop("spec")[100:]
    payload["revision"] = "route-relative-v2"
    path.write_text("const DATA=" + json.dumps(payload, separators=(",", ":")) + ";")
    for name in ["index.html", "app.js"]:
        (OUT / "viewer" / name).write_text(Path(__file__).with_name(name).read_text())


if __name__ == "__main__":
    main()
