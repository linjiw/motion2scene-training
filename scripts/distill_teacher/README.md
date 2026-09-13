# Anticipatory motor distillation from a local teacher

These scripts produced `workspace/distill-8192` and the [8192-teacher distillation report](../../docs/sonic/motion2scene/DISTILL_8192_TEACHER_20260913.md). They drive the vendored `gear_sonic.research.scene_distillation` modules (`collect`, `motor_training`, `online_motor`, `motor_runtime`) and add no model code.

Set `DISTILL_PACKET` to the packet directory; it defaults to `workspace/distill-8192`. Native launches use `.venv_native` with `PYTHONPATH` unset and a user-owned `TMPDIR`. On shared hosts, run long stages in `tmux`. Python helpers need the vendored package on `PYTHONPATH`:

```bash
cd vendor/sonic && export PY="env -u PYTHONPATH PYTHONPATH=$PWD ../../.venv_native/bin/python"
export DISTILL_PACKET=$PWD/../../workspace/distill-8192 T=../../scripts/distill_teacher
```

## Steps

1. **Stage the packet.**
   ```bash
   $PY $T/stage_packet.py --packet $DISTILL_PACKET \
     --teacher-run ../../workspace/teacher-8192-500/tracking-run-1 \
     --motions ../../workspace/teacher-8192-500/motions
   ```
   This writes `ids.json` and `collect/collection-lock.json`.
2. **Collect teacher data.** Run `bash $T/collect_seeds.sh 91400 91401 ...`: one 89-environment native collection per seed, about 3.5 minutes each.
3. **Broaden support and merge.** For each seed, run
   ```bash
   $PY $T/broaden_support.py --collection $DISTILL_PACKET/collect/seed-S/metrics/collection.json \
     --output $DISTILL_PACKET/collect/broad-S
   ```
   Then merge with `$PY -m gear_sonic.research.scene_distillation.aggregate --inputs .../broad-*/collection.json --output .../broad-all.json`.
4. **Fit offline.**
   ```bash
   $PY $T/make_init.py --ids $DISTILL_PACKET/ids.json --output $DISTILL_PACKET/offline/init-anticipatory.pt
   $PY $T/fit_config.py --manifest .../broad-all.json --init .../init-anticipatory.pt \
     --updates 30000 --save-interval 6000 --out .../fit-config.json
   $PY -m gear_sonic.research.scene_distillation.motor_training --config .../fit-config.json --output .../offline/fit-30k
   ```
5. **Run online DAgger.** Write a stage config with `$PY $T/online_config.py --student ... --replay .../broad-all.json --output .../online/NAME/training --cycles 250 --batch 1024 --config .../online/NAME-config.json`, then launch:
   ```bash
   bash $T/native.sh $DISTILL_PACKET/online/NAME train 2048 91370 \
     gear_sonic.research.scene_distillation.online_motor.OnlineMotorCallback \
     ++callbacks.im_eval.stage_config=.../online/NAME-config.json
   ```
   The callback caps each stage at 7,200 s and 1,000 cycles.
6. **Evaluate.** Run `bash $T/eval_student.sh <checkpoint|teacher> <train|development> <seed> <out>` directly. Alternatively, add `checkpoint split seed out` lines to a queue file and run `eval_queue.sh`; `watch_checkpoints.sh` appends seed-91260 entries for each new online checkpoint. `score.py` and `status.py` tabulate `metrics_eval.json`.

## Notes

- **One process at a time by default.** `native.sh` refuses to start while another of your Isaac processes runs. Set `M2S_ALLOW_CONCURRENT=1` to run evaluations beside online training.
- **Measured GPU memory.** Online at 2048 envs ~7.7 GB; evaluation or collection ~3.8 GB; offline fit ~4.3 GB.
- **Output directories must be new.** The trainers and `native.sh` refuse to overwrite.
