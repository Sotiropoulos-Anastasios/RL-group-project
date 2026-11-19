# Raw DialOp data

This directory is for the downloaded DialOp source files. Raw files are
ignored by Git because they are inputs to the reproducible build, not the
ready-to-use training artifact.

The builder needs:

```text
data/raw/optimization.jsonl
```

The source archive and the other DialOp games may also be kept here for local
reference, but this RL preparation uses only `optimization.jsonl`.

After placing the source file here, run:

```powershell
python scripts/build_dataset.py
```

The source implementation and task definition are preserved in the archived
[DialOp repository](https://github.com/microsoft/dialop). The paper is
[Decision-Oriented Dialogue for Human-AI Collaboration](https://arxiv.org/abs/2305.20076).
