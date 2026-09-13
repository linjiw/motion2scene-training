# Contributing

Use a new branch and isolated environment. Never change an active training environment in place. Keep data bundles and historical evidence immutable; add new experiment packets and manifests instead.

Install the base test dependencies with `pip install -e '.[test]'` and run `python -m pytest -q tests` (with `PYTHONPATH` unset if your shell sources ROS or another Python distribution). Native tests require the documented Isaac/SONIC dependencies. Format new package code with Black and check it with Ruff. When editing vendored code, record the original snapshot hash, current hash and reason in `manifests/vendor-amendments.json`.

A research PR should state the question, changed behavior, data/teacher bindings, exact executed checks and untested limits. Numerical claims must link to measured receipts. Preserve failures and qualification gates. Sign off commits with `git commit -s`.

Dataset/model contributions need provenance and original asset terms. Do not add credentials or simulator installations. Large assets belong in Git LFS with SHA-256 manifests. Publishing a code change does not grant new rights to the underlying assets.
