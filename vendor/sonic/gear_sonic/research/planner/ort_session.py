"""CPU ONNX Runtime session for ``planner_sonic.onnx`` (batch 1), checked against the contract.

Run it with an interpreter that has ``onnxruntime`` (on this host
``/home/robotixx/GR00T-WholeBodyControl/.venv_sim/bin/python``, ORT 1.23.2). One session per
thread; ``intra_op_threads`` defaults to 4.
"""

from __future__ import annotations

import os
import time

import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_planner_adapter import (
    INPUT_SPEC,
    OUTPUT_SPEC,
)

DEFAULT_PLANNER_ONNX = (
    "/home/robotixx/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/"
    "planner_sonic.onnx"
)
_ORT_TYPES = {"float32": "tensor(float)", "int64": "tensor(int64)", "int32": "tensor(int32)"}


class OrtPlannerSession:
    """Callable ``inputs -> {"mujoco_qpos", "num_pred_frames"}`` with per-call timing."""

    def __init__(
        self,
        model_path: str = DEFAULT_PLANNER_ONNX,
        *,
        intra_op_threads: int = 4,
        inter_op_threads: int = 1,
    ):
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            raise RuntimeError("run with CUDA_VISIBLE_DEVICES='' (CPU-only guard)")
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(intra_op_threads)
        options.inter_op_num_threads = int(inter_op_threads)
        started = time.perf_counter()
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.init_seconds = time.perf_counter() - started
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("unexpected execution provider")
        for actual, spec in (
            (self.session.get_inputs(), INPUT_SPEC),
            (self.session.get_outputs(), OUTPUT_SPEC),
        ):
            if {value.name for value in actual} != set(spec):
                raise ValueError("graph tensor names differ from the audited contract")
            for value in actual:
                dtype, shape = spec[value.name]
                if value.type != _ORT_TYPES[dtype] or tuple(value.shape) != shape:
                    raise ValueError(f"graph tensor differs from the contract: {value.name}")
        self.model_path = model_path
        self.intra_op_threads = int(intra_op_threads)
        self.inter_op_threads = int(inter_op_threads)
        self.ort_version = ort.__version__
        self.call_seconds: list[float] = []

    def __call__(self, inputs):
        started = time.perf_counter()
        qpos, count = self.session.run(list(OUTPUT_SPEC), dict(inputs))
        self.call_seconds.append(time.perf_counter() - started)
        return {"mujoco_qpos": qpos, "num_pred_frames": count}

    def timing_summary(self) -> dict:
        values = np.asarray(self.call_seconds) * 1e3
        if len(values) == 0:
            return {"calls": 0}
        return {
            "calls": int(len(values)),
            "median_ms": float(np.median(values)),
            "p10_ms": float(np.percentile(values, 10)),
            "p90_ms": float(np.percentile(values, 90)),
            "max_ms": float(values.max()),
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
            "onnxruntime": self.ort_version,
            "session_init_s": self.init_seconds,
        }
