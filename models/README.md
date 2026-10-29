# Local model assets

Open-weight planner checkpoints are staged below this directory and ignored by
git. Their upstream repository, immutable revision, expected files, and hashes
are recorded in `naigos/planner/qwen3_0_6b_base.json`.

The default Qwen planner asset path is:

```text
models/qwen3-0.6b-base/
```

Use `python scripts/qwen_planner.py verify-model` before training. Training and
evaluation use `local_files_only=True`; after staging, they do not access the
network.
