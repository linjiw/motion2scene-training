# E0-Q3 Pilot V7 Result

Observed: 2026-09-04T13:31:28-04:00.

The first V7 cell completed one full 199-step SONIC pass, wrote exactly one 199-frame trajectory
at 50 Hz, emitted the success marker, and exited normally. Rendering was disabled and no render
artifact was expected. The Q3 scorer then classified the trajectory as unevaluable because a bare
plane has no pair-resolved support-floor sensors, leaving three required evidence fields absent.

This reproduces repository design delta D2-006. The capture remains valid non-verdict trajectory
evidence, but it is not a scientific acceptance or rejection. The batch stopped at 1/6 cells; the
remaining five were not started.

The next registered batch uses the existing `screen_empty` scene, whose authored floor enables
the required support sensors. Its remote geometry must contribute zero external collision force.

