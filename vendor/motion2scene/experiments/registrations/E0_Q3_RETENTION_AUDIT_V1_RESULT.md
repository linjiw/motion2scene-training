# E0 Q3 retained-state audit result

Status: completed postregistered secondary audit of the frozen six-cell V8 pilot. This does not
alter the V8 candidates, tracker verdict, or original selection.

## Result

- Q3a tracker survival: 5/6.
- Q3b requested-route retention: 1/6 retained, 3/6 lost after tracking, and 2/6 could not be
  called retained because the reference itself failed the new route gate.
- Q3c paired S0--S4: 0/6 measured. V1 assigned a different Kimodo seed to every prompt and has no
  same-seed, same-route walk null.
- Legacy self-baseline behavior: duck and shoulder-turn were present in both reference and
  achieved trajectories; arm-tuck was present but its rollout failed the tracker gate. These are
  not promoted to S4.
- Q3d whole-body paired envelope retention: not measured. The legacy reductions use torso height
  or body-link-origin width, not exact whole-body height or route-normal collision width.

This audit narrows the motion substrate: none of the six is eligible for a v2 matched critical
ladder on this evidence alone.

## Bound artifacts

| artifact | SHA-256 |
| --- | --- |
| Legacy reference semantic measurements | `sha256:202826396f2b8ab17ea2ab1c68cde2308d4219482e0deb16d74f9f4a0e7f2aa4` |
| Route secondary JSON | `sha256:3326c9958535b862dc7ee1e468f39628fa9cb1199caea962bc099a474cfae70f` |
| Route secondary report | `sha256:64138a7f066ff3e1f3f7ce42b2909f88c9654ea7357b7d68aa0a69928c99fb59` |
| Q3 retention audit JSON | `sha256:b80df9bea8c553bd7b7ed3d1c64fe5a9f0110c3769ce179ed9bebf28692f7da2` |
| Q3 retention audit report | `sha256:f8cdbeff4c81d16f05ab5a96e8b295b0b72a67150ca4e70795ce63d46676b23d` |

External root:
`/home/linjiw/research-data/groot-wbc/cg-wbc-v1-pilot/`

Analysis code: `experiments/e0_route_validity.py` and `experiments/e0_q3_retention.py`.
