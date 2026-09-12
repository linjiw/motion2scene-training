# Primary-source checks

Checked on 8 September 2026. These sources establish context and venue rules;
they do not validate Motion2Scene's records. No draft was uploaded to a third-party
research service. References below were inspected directly at the stated depth.

| Source | Inspection and role in this paper | Boundary |
|---|---|---|
| [LfH](https://arxiv.org/abs/2007.14479) | Primary abstract and bibliographic record: constructs constrained scenes from open-space motions | Motion-to-scene generation alone is not new |
| [LfLH](https://arxiv.org/pdf/2108.09793) | Method and training passages: encoder, planning-based decoder, generated observations and downstream reactive learning | Neither learned hallucination nor downstream utility alone distinguishes this paper |
| [LfH-CP](https://arxiv.org/html/2509.26513) | Introduction, construction method and evaluation: critical points, dynamic obstacle trajectories and trained-navigation evaluation | Motion2Scene's distinction is paired **executed humanoid command outcomes**, their frozen-controller verification, and acceptance/learning diagnostics |
| [HumanoidPF](https://arxiv.org/html/2601.16035) | Primary abstract and method sections: scene generation, potential-field representation, learned whole-body traversal, real-world evaluation | Broader capability and deployment scope; no like-for-like numerical performance comparison |
| [Perceptive Humanoid Parkour](https://php-parkour.github.io/static/images/paper.pdf) | Primary paper and [author project](https://php-parkour.github.io/): motion matching, tracking experts, depth-policy distillation and long-horizon skills; RSS 2026 listing checked | Its real-world skill-chaining result is not a baseline on this finite two-command bank |
| [SONIC](https://arxiv.org/abs/2511.07820) | Primary abstract and updated bibliographic record, including Science Robotics 11(117), eaed4592 (2026) | Inherited tracking capability; not a Motion2Scene contribution or study-specific training cost |
| [Agarwal et al.](https://arxiv.org/abs/2108.13264) | Primary abstract and stated evaluation-uncertainty framing | Few-run uncertainty depends on experimental units; does not license a generic episode bootstrap here |
| [Eliasziw and Donner](https://pubmed.ncbi.nlm.nih.gov/1805322/) | Bibliographic record and primary abstract, not a full-method reproduction: ordinary matched-pair inference assumes independence across pairs; discusses repeated-measures adjustment | We retain descriptive registered calculations and expose clusters; do not claim to implement this paper's adjustment |
| [Lakens](https://journals.sagepub.com/doi/full/10.1177/1948550617697177) | Primary equivalence-testing discussion | Nonsignificance is not equivalence; no equivalence margin was preregistered or tested here |

The manuscript's novelty is a bounded combination of execution-conditioned scene
construction, paired command verification, and measured selector-data utility.
The learned proposal is an ablation. This is not a claim that no earlier work has
ever used motion contrasts, paired outcomes, or geometric acceptance.

## Venue checks

The [official ICRA 2027 CFP](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
lists September 15, 2026, 11:59 PST as the paper deadline. Use the actual portal
clock; target September 14. The PDF must use the conference double-column format
and fit all material, references and acknowledgments within eight pages.
There is no separate textual appendix allowance. At least three keywords are
required, and every author must be entered in the portal.

The same CFP allows accompanying video during August 5–September 9 and
September 17–22: at most 180 seconds and 20 MB, mpeg/mp4/mpg, at least 480-pixel
height and 20 fps, progressive scan. It permits textual URLs, not embedded PDF
links. Reviewers need not inspect external sites. Substantive AI-generated text,
figures or code requires an acknowledgment naming the system, affected sections
and use; grammar editing is distinguished. The current acknowledgment documents
this revision, while earlier usage awaits author reconciliation.

The [RAS double-anonymous rules](https://www.ieee-ras.org/publications/rules-for-the-double-anonymous-review-process/)
require removing identifying authorship, metadata and project links. Essential
robot imagery is allowed. The new PDF/video omit author identity and project
links; local manifests are not anonymous review attachments.

The class was obtained from the [official PaperCept LaTeX support page](https://ras.papercept.net/conferences/support/tex.php)
and its linked archive. [Template provenance](template-provenance.json) records
the downloaded archive URL and SHA-256 values. Body size and class margins remain
standard; no page-limit compression was applied.
