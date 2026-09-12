"""Workflow-specific checks without starting Isaac or allocating GPU memory by default."""

import importlib.util
import sys

PROFILES = {
    "base": [],
    "view": ["numpy", "matplotlib"],
    "student": ["numpy", "torch", "hydra", "trl", "transformers", "joblib"],
    "teacher": [
        "numpy",
        "torch",
        "hydra",
        "trl",
        "isaaclab",
        "isaacsim",
        "smpl_sim",
        "mujoco",
    ],
}


def inspect(root, workspace, profile, cuda=False):
    missing = [
        name for name in PROFILES[profile] if importlib.util.find_spec(name) is None
    ]
    issues = [f"Missing module: {name}" for name in missing]
    if profile in ("student", "teacher") and sys.version_info[:2] != (3, 11):
        issues.append("Native workflows require Python 3.11")
    if profile in ("student", "teacher"):
        model = "previous_teacher.pt" if profile == "student" else "sonic_release.pt"
        if not (workspace / "models" / model).is_file():
            issues.append(
                f"Missing models/{model}; unpack checkpoints into this workspace"
            )
    if profile == "teacher":
        assets = root / "vendor/sonic/gear_sonic/data"
        expected = workspace / "vendor/sonic/gear_sonic/data"
        if not assets.exists() or assets.resolve() != expected.resolve():
            issues.append(
                "Robot assets missing or bound to another workspace; unpack robot-assets"
            )
        if not (
            workspace / "m2s-sonic-repaired-v1_1-20260912/screened/train/metadata.pkl"
        ).is_file():
            issues.append("Missing repaired training data; unpack repaired-motions")
    result = {
        "profile": profile,
        "python": sys.version,
        "repository": str(root),
        "workspace": str(workspace),
        "issues": issues,
        "scope": (
            "Dependency discovery and input presence; "
            "not a simulator launch or full hash audit"
        ),
    }
    if cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                issues.append("CUDA unavailable")
            else:
                x = torch.ones(16, 16, device="cuda")
                result["cuda_matmul"] = float((x @ x)[0, 0])
                if result["cuda_matmul"] != 16:
                    issues.append("CUDA numerical probe failed")
        except Exception as error:
            issues.append(f"CUDA probe failed: {error}")
    result["ready"] = not issues
    return result
