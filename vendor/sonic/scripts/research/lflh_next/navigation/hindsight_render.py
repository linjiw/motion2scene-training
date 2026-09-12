"""Render measured motion channels and newly scored hindsight candidates."""

import json
from pathlib import Path

import torch

from scripts.research.lflh_next.navigation import render
from scripts.research.lflh_next.navigation.hindsight_study import OUT


def main():
    render.OUT = OUT
    render.main()
    p = OUT / "viewer/data.js"
    payload = json.loads(p.read_text().removeprefix("const DATA=").removesuffix(";"))
    payload["routeSpec"] = payload.pop("spec")[100:]
    payload["cfUpper"] = torch.load(OUT / "labels.pt", weights_only=True)["cf_upper"][100:].tolist()
    payload["revision"] = "hindsight-v3"
    p.write_text("const DATA=" + json.dumps(payload, separators=(",", ":")) + ";")
    for target, source in [("index.html", "hindsight_index.html"), ("app.js", "hindsight_app.js")]:
        (OUT / "viewer" / target).write_text(Path(__file__).with_name(source).read_text())


if __name__ == "__main__":
    main()
