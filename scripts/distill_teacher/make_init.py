"""Zero-initialized AnticipatoryMotor bound to the 8192x500 teacher (current-frame repetition)."""

import argparse
import json
from pathlib import Path

import torch

from gear_sonic.research.scene_distillation.motor_training import build_motor


def main(ids_path, output):
    ids = json.loads(Path(ids_path).read_text())
    config = dict(
        motor_architecture="anticipatory",
        current_frame_extension=True,
        reference_layout_version="native_q_then_v_v1",
        teacher_checkpoint=ids["teacher_checkpoint"],
        teacher_sha256=ids["teacher_sha256"],
    )
    model = build_motor(config)
    if bool(model.normalized):
        raise ValueError("Initialization must leave normalization unset")
    torch.save(
        dict(
            stage="motor_foundation",
            model=model.state_dict(),
            config=config,
            teacher_sha256=config["teacher_sha256"],
            step=0,
            qualification="zero_initialized_current_frame_repetition",
        ),
        Path(output),
    )
    print("wrote", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    main(a.ids, a.output)
