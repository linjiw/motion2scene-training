"""Render one frozen USD reference scene through Isaac Lab, with physics stopped.

The GPU preflight fails closed. A render is visualization, never a teacher rollout.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from gear_sonic.research.hindsight_training.runtime import cuda_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.frames <= 60:
        parser.error("frames must be between 1 and 60")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "state": "preflight",
        "scene": args.scene,
        "requested_render_frames": args.frames,
        "rendered_frames": 0,
        "physics_step_events": 0,
        "teacher_rollouts": 0,
        "physical_labels": None,
        "application_launched": False,
    }
    app, subscription = None, None
    try:
        scene_path = args.dataset / "scenes" / (args.scene + ".usdc")
        if not scene_path.is_file():
            raise FileNotFoundError(scene_path)
        gpu = cuda_preflight(minimum_free_bytes=4096 * 1024**2)
        receipt["gpu_preflight"] = gpu
        receipt["gpu_query_exit_code"] = gpu["nvml_exit"]
        (args.output / "gpu-preflight.txt").write_text(gpu["nvml_output"])
        if not gpu["cuda_execution_passed"]:
            receipt.update(state="BLOCKED", reason="Actual CUDA execution failed")
            return 2
        receipt["GPU_prelaunch_MiB"] = {
            "total": gpu["total_bytes"] / 1024**2,
            "free": gpu["free_bytes"] / 1024**2,
        }
        if not gpu["launch_allowed"]:
            receipt.update(
                state="BLOCKED",
                reason="Less than 4096 MiB free for bounded render; other jobs untouched",
            )
            return 2
        if args.preflight_only:
            receipt["state"] = "preflight_passed"
            return 0
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, enable_cameras=True).app
        receipt["application_launched"] = True
        import carb
        import imageio.v3 as iio
        import numpy as np
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Usd

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        carb.settings.get_settings().set("/app/player/playSimulations", False)

        def physics_event(dt):
            receipt["physics_step_events"] += 1

        subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(physics_event)
        context = omni.usd.get_context()
        context.open_stage(str(scene_path.resolve()))
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        assert stage.GetPrimAtPath("/World/Camera")
        product = rep.create.render_product("/World/Camera", (960, 640))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        frames = np.unique(
            np.linspace(stage.GetStartTimeCode(), stage.GetEndTimeCode(), args.frames).round()
        )
        for index, frame in enumerate(frames):
            timeline.set_current_time(float(frame) / stage.GetTimeCodesPerSecond())
            assert not timeline.is_playing()
            rep.orchestrator.step(rt_subframes=4, pause_timeline=True, delta_time=0.0)
            pixels = np.asarray(rgb.get_data())
            if pixels.size == 0:
                raise RuntimeError("Renderer returned an empty image")
            iio.imwrite(args.output / f"frame-{index:03d}.png", pixels)
            receipt["rendered_frames"] += 1
        assert (
            receipt["physics_step_events"] == 0
        ), "Unexpected physics event; images are still not passage evidence"
        receipt.update(
            state="complete",
            usd_version=list(Usd.GetVersion()),
            evidence="Isaac Lab launched reference render, no qualified controller or sensor stream",
        )
        return 0
    except Exception as error:
        receipt.update(state="technical_failure", error=repr(error))
        raise
    finally:
        if subscription is not None:
            subscription = None
        if app is not None:
            app.close()
        receipt["measured_wall_seconds"] = time.perf_counter() - started
        with (args.output / "receipt.json").open("x") as f:
            json.dump(receipt, f, indent=2)
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
