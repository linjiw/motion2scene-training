"""Private local visualization of motion-derived reference geometry."""

import json
from pathlib import Path

from plotly.offline import get_plotlyjs
import torch

OUT = Path("/home/linjiw/research-data/m2s-multimotion-shapes-20260911")
d = torch.load(OUT / "geometry.pt", weights_only=True)
p = torch.load(OUT / "probabilities.pt", weights_only=True)
x = (d["features"][64:] * d["scale"] + d["mean"]).reshape(16, 32, 30, 3)
p["uniform"] = torch.ones(16, 225) / 225
payload = {
    "motion": x.tolist(),
    "gap": d["gaps"][64:].tolist(),
    "spec": d["specs"].tolist(),
    "prob": {k: v.tolist() for k, v in p.items()},
}
js = Path(__file__).with_name("viewer.js").read_text()
html = (
    Path(__file__).with_name("template.html").read_text()
    + get_plotlyjs()
    + """</script><script>const DATA="""
    + json.dumps(payload, separators=(",", ":"))
    + """;</script><script>"""
    + js
    + """</script></html>"""
)
(OUT / "viewer.html").write_text(html)
print(OUT / "viewer.html")
