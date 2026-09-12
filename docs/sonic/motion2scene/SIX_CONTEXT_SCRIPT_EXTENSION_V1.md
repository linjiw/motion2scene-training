# Six-context script selection extension

The original ordered 72-setting sensor-script grid was selected again after the
complementary passage became development evidence. The ranking, cost definition,
prefix requirements, unknown handling, and exact tie rules are unchanged. This
gives the script the same added development contrast as the six-context learner.
All 42 physical outcomes were already known when this separate extension was
registered; it is not a prospective held-out or curriculum experiment.

The once-only selection re-audited all 42 forced branches from the original five
contexts and the complementary context: 50,064 previously recorded physics steps,
42 known outcomes, no unknown branches, and no new physics. All 72 settings have
complete known panels. Eleven pass all six recorded-branch proxies. Setting 57
is the unique minimum-cost winner among them:

```json
{
  "immediate_prior_band_count": 1,
  "later_prior_band_count": 2,
  "minimum_observed_free_height_m": 1.4,
  "sustained_minimum_hazard_bands": 1
}
```

The script consumes only actual eligible neutral 114D sensor/state packets at
ticks 15, 50, and 70 until its first commitment. Each chosen complete branch has
an exactly matching physical and causal sensor prefix. Later forced-neutral
packets after commitment are not used. The runtime loader independently accepts
the selected JSON and returns exactly these parameters. The observed-height
quantity is the existing corridor summary, which can combine separately observed
floor and ceiling cells; its threshold is not a claim of dense paired-surface
perception.

| Context | New first commitment | Measured branch passage time | Original selected-script time |
| --- | --- | ---: | ---: |
| Empty | WAIT at all three phases | 5.36 s | 5.36 s |
| Short beam `0187` | WAIT, WAIT, sustained at 1.40 s | 4.06 s | 3.62 s |
| Long beam `0247` | WAIT, WAIT, sustained at 1.40 s | 4.64 s | 4.50 s |
| Early beam `0284` | WAIT, prior at 1.00 s | 2.72 s | 2.72 s |
| Two-beam course | WAIT, prior at 1.00 s | 4.12 s | 4.12 s |
| Complementary beam `0100` | WAIT, WAIT, sustained at 1.40 s | 4.20 s | Failure |

The new script's recorded-branch proxy is 6/6, with mean relative-time regret
0.02380952. The original selected script is 5/6 on this expanded panel, with
regret 0.16738506. The added passage capability comes with measured proxy cost
regressions of 0.44 seconds on the short beam and 0.14 seconds on the long beam.
These are physical branch costs under teacher seed 8731, not newly executed
script episodes or estimates of seed-8732 performance.

All seven constants were ranked on the same six contexts. The original constant
`prior_splice_e015_r265` remains selected: 5/6 passage and mean regret 0.16788321.
Both prior schedules fail the complementary context; both sustained schedules
fail the early context. No qualified constant passes all six contexts. This
finite-panel finding justifies testing selection, without establishing that a
learned policy outperforms the newly selected strong script.

The immutable selection directory is
`/home/linjiw/research-data/groot-wbc/m2s-six-context-script-development-selection-v1`.
Its registration SHA-256 is
`64300ea10d40e42d0fb83982e00da1775ab50e77edc0ea3360df65033736ef87`;
the result SHA-256 is
`3dafd2b0e6a296b3394e8e816272cbc1dc4e816508f5215521d74a82dacb1932`.
Selected runtime `script_parameters.json` has SHA-256
`c92cee106307966dbbc8e2d2799c08de4bd63383de41b6015ae9cb0ddf857470`.
The original five-context selection, model, and actual 20-episode comparison
remain unchanged.

Implementation: `scripts/research/motion2scene_select_six_context_baselines.py`,
using the byte-identical original ranking helper. Fifteen focused tests pass.
The initial pre-freeze wrapper validation error (a nonexistent redundant manifest
field) is retained in `pre_freeze_validation_attempt_01.json`; it scored no
candidates and changed no registered selection rule.

The next assignment-only specification is
`m2s-six-context-policy-extension-plan-v1/specification.json`, SHA-256
`ea1bb61fcaec44858db135964bf45a1fc2cf031e8eeecd0968123144bd1af9b7`.
It fixes six learned6 and six script6 episodes plus the complementary context
for the original four policies, all at seed 8732: 16 new assignments with a
19,072-step budget. Together with the unchanged original 20, these fill a
six-context by six-policy development table. That assignment-only specification
binds no stale execution-source hashes.

Following the separately audited scorer application and native/runtime asset
refresh, all 16 child invocations were prepared in
`m2s-six-context-policy-extension-v1`. Registration SHA-256 is
`38a8f1000c5ead182fa34a1e44bab336cd83b10dd5c980287314a2a93d350839`;
the independent `prelaunch_audit.json` SHA-256 is
`a837eda56e9e551d4837ca27bb29325e575a38c59be80e0e8a84da19c9543bab`.
The audit checks every actual command and model/script binding, all three applied
scorer modules, the common 222-artifact runtime declaration, exact sensor and
horizon settings, all one- or two-beam contact paths, and all 16 unlaunched output
directories. The derived executable specification is in
`m2s-six-context-policy-extension-execution-plan-v1`, SHA-256
`576cdf2fb980857fe4c1ce715e8da2107d67cceff1a50fac5143104f3106fc3f`.
Preparation and this audit launch no physics. The queue owner must execute the
registered batch before any six-context actual policy outcome is claimed.

The registered extension has now completed. The
[six-context actual comparison](SIX_CONTEXT_POLICY_COMPARISON_V1.md) retains all
36 assignments across the original and extended panels: 35 complete captures and
one partial, contact-verified failure. The newly selected script and six-context
learner each pass all six development contexts; both take 0.50 s longer on the
short and long passages than their original five-context versions. These results
describe actual development executions, separately from the offline selection
proxy above.
