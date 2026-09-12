# E0-Q3 Pilot V8 Gradeable-Empty Design Delta

Registered: 2026-09-04T13:32:58-04:00, before V8 execution.

V7 completed a valid 199-frame, 50 Hz bare-plane trajectory, but its lack of pair-resolved
support-floor sensors made the Q3 acceptance policy correctly return unevaluable. This is the
exact failure already captured by repository design delta D2-006.

V8 uses the established `screen_empty` scene. Its authored
`/World/ground/terrain/Structure/Floor` supplies left/right ground-force evidence. Its walls are
more than four metres from the route and its single shelf is at 5.05 m height. An accepted Q3 cell
must additionally record zero external collision force; otherwise the scene cannot serve as an
obstacle-absent proxy.

Because the scene/evidence contract changed, V8 has a new prediction record. Its numerical motion
predictions are identical to V2 and were not revised using the ungradeable V7 trajectory. V8 keeps
the six motions, seeds, controller, trajectory-only recorder, memory gate, serial scheduling, and
cost ceiling fixed.
