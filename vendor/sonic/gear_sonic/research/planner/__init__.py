"""SONIC kinematic planner: deploy-faithful closed-loop runner, controllers and geometry.

Nothing in this package imports ONNX Runtime, a simulator or a GPU library at module load.
``ort_session`` imports ``onnxruntime`` only when a session is constructed.
"""
