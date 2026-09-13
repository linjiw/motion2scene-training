"""Build a fully local HTML reading copy and printable PDF, with offline math.

Uses Matplotlib, Markdown, and Playwright with an installed Chromium.
Markdown can be provided via PYTHONPATH without changing project dependencies.
"""

import base64
import io
from pathlib import Path
import re

import markdown
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
source = (ROOT / "methods_draft.md").read_text()
math_fragments = []


def math_html(formula, display):
    buffer = io.BytesIO()
    formula = " ".join(formula.split())
    math_to_image(
        "$" + formula + "$",
        buffer,
        format="svg",
        prop=FontProperties(size=14 if display else 12),
        color="#172b42",
    )
    encoded = base64.b64encode(buffer.getvalue()).decode()
    label = formula.replace('"', "&quot;")
    css = "display-math" if display else "inline-math"
    image = f'<img class="{css}" alt="{label}" src="data:image/svg+xml;base64,{encoded}">'
    if display:
        image = '<div class="equation">' + image + "</div>"
    key = f"MATHPLACEHOLDER{len(math_fragments):04d}END"
    math_fragments.append((key, image))
    return "\n\n" + key + "\n\n" if display else key


source = re.sub(r"\$\$(.*?)\$\$", lambda m: math_html(m.group(1), True), source, flags=re.S)
source = re.sub(r"\$([^$\n]+)\$", lambda m: math_html(m.group(1), False), source)
body = markdown.markdown(source, extensions=["tables", "fenced_code", "toc", "sane_lists"])
for key, fragment in math_fragments:
    body = body.replace("<p>" + key + "</p>", fragment).replace(key, fragment)
# Figures open the vector original; all assets and math remain local.
body = re.sub(
    r'<img alt="([^"]*)" src="(0[123][^"]+)\.png"\s*/?>',
    r'<a class="figure" href="\2.svg" target="_blank"><img alt="\1" src="\2.svg"></a>',
    body,
)
body = body.replace("<p><strong>Figure ", '<p class="caption"><strong>Figure ')
html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion2Scene — Research overview and methods</title>
<style>
:root{color-scheme:light;
--ink:#172b42;
--muted:#54667b;
--teal:#167d86}

*{box-sizing:border-box}
body{margin:0;
background:#edf2f6;
color:var(--ink);
font:17px/1.78 system-ui,sans-serif}

nav{background:var(--ink);
color:white;
padding:16px 5vw;
display:flex;
gap:28px;
align-items:center;
flex-wrap:wrap}

nav b{letter-spacing:.12em;
font-size:13px}
nav a{color:#cbece8;
font-size:14px;
text-decoration:none}

main{background:white;
max-width:1320px;
margin:30px auto 70px;
padding:55px 65px;
box-shadow:0 10px 50px #172b4209}

h1{font-size:40px;
line-height:1.15;
max-width:1060px;
letter-spacing:-.035em;
margin:0 0 25px}

h2{font-size:25px;
line-height:1.3;
margin:62px 0 20px;
border-top:1px solid #dce5ec;
padding-top:23px;
color:var(--teal)}

h3{font-size:20px;
margin-top:35px}
p,li{max-width:1080px}
p{margin:16px 0}
a{color:#2564ac;
text-underline-offset:3px}

.figure{display:block;
margin:26px -30px;
text-decoration:none}
.figure img{display:block;
width:100%;
height:auto;
border:1px solid #e3e9ef}

.caption{background:#f2f6f8;
border-left:4px solid var(--teal);
padding:20px 24px;
font-size:14px;
line-height:1.65;
max-width:none}

.equation{overflow:auto;
text-align:center;
margin:24px 0;
padding:18px;
background:#f5f7fa;
border-radius:8px}

.display-math{max-width:100%;
height:auto}
.inline-math{height:1.2em;
width:auto;
vertical-align:-.19em;
max-width:none}

table{border-collapse:collapse;
width:100%;
font-size:14px;
line-height:1.6;
margin:25px 0}
th{text-align:left;
background:#eaf3f3;
color:var(--teal)}

th,td{border:1px solid #dce5ec;
padding:12px 15px;
vertical-align:top}
tr:nth-child(even){background:#f8fafc}

pre{overflow:auto;
background:#172b42;
color:#eef7ff;
padding:20px;
font-size:13px;
border-radius:8px}
code{font-family:ui-monospace,monospace}

@media(max-width:760px){main{margin:0;
padding:32px 23px}
h1{font-size:30px}
.figure{margin:20px -12px}
table{font-size:12px}
th,td{padding:8px}
}

@page{size:A4;
margin:18mm 16mm 20mm}
 @media print{
body{background:white;
font:10.3pt/1.5 Arial,sans-serif}
nav{display:none}
main{max-width:none;
margin:0;
padding:0;
box-shadow:none}

h1{font-size:26pt}
h2{font-size:16pt;
margin-top:25pt;
padding-top:12pt;
break-after:avoid}
h3{font-size:12pt;
break-after:avoid}

p{orphans:3;
widows:3;
margin:9pt 0}
.figure{margin:12pt 0;
break-inside:avoid}
.figure img{max-height:225mm;
object-fit:contain}

.caption{font-size:9pt;
line-height:1.45;
padding:10pt 12pt;
box-decoration-break:clone}
table{font-size:8pt;
line-height:1.35}

th,td{padding:6pt 7pt}
tr{break-inside:avoid}
a{color:#245985;
text-decoration:none}
.equation{break-inside:avoid;
padding:10pt}

pre{white-space:pre-wrap;
font-size:8pt;
break-inside:avoid}
.inline-math{height:1.13em}
.display-math{max-height:35mm}

}

</style></head><body>
<nav><b>MOTION2SCENE / METHODS ATLAS</b><a href="research_methods.pdf">PDF document</a>
<a href="methods_draft.md">Editable draft</a><a href="01_research_overview.svg">Overview SVG</a>
<a href="02_teacher_student_architecture.svg">Architecture SVG</a>
<a href="03_scene_and_learning_illustration.svg">Scene illustration SVG</a></nav>
<main>""" + body + "</main></body></html>"
path = ROOT / "index.html"
path.write_text(html)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
    page.goto(path.as_uri(), wait_until="networkidle")
    page.evaluate("document.fonts.ready")
    broken = page.locator("img").evaluate_all(
        "xs => xs.filter(x => !x.complete || !x.naturalWidth).map(x => x.src)"
    )
    if broken:
        raise RuntimeError(f"Broken images: {broken}")
    page.screenshot(path=str(ROOT / "reading_copy_preview.png"), full_page=False)
    page.pdf(
        path=str(ROOT / "research_methods.pdf"),
        print_background=True,
        prefer_css_page_size=True,
        display_header_footer=True,
        header_template="<span></span>",
        footer_template=(
            '<div style="font:8px '
            'Arial;width:100%;text-align:center;color:#54667b">Motion2Scene · Methods '
            'draft · 13 September 2026 — <span class="pageNumber"></span> / <span '
            'class="totalPages"></span></div>'
        ),
    )
    print(
        {
            "math_expressions": len(math_fragments),
            "images": page.locator("img").count(),
            "horizontal_overflow": page.evaluate(
                "document.documentElement.scrollWidth > innerWidth"
            ),
            "html": str(path),
            "pdf": str(ROOT / "research_methods.pdf"),
        }
    )
    browser.close()
