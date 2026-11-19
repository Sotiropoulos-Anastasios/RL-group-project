# RL-group-project

This repository contains the group reinforcement-learning project work.

## DialOp data preparation

The initial DialOp reviewer-assignment data package is in
[`dialop_project/`](./dialop_project/). It converts DialOp's human-human
optimization records into reproducible RL-friendly episodes and normalized
dialogue actions.

The package includes:

- scaled player-specific private views with hidden cells represented as `null`;
- the unchanged oracle table for environment scoring;
- episode-level train/validation/test splits;
- validation of scores, masks, scales, actions, and one-to-one proposals;
- source-to-processed reward consistency checks;
- processed JSONL artifacts and audit reports; and
- documentation describing the data boundary and remaining RL work.

The raw DialOp source data is intentionally excluded from Git. To rebuild the
processed data locally, place `optimization.jsonl` in
`dialop_project/data/raw/` and run:

```powershell
cd dialop_project
python scripts/build_dataset.py
python scripts/audit_dataset.py
python scripts/visualize_dataset.py
```

DialOp's original normalized reward is:

```text
normalized_score = proposal_reward / best_assignment_reward
```

The `best_assignment_reward` is DialOp's pooled-observation reference, not a
new fairness or human-welfare label. Cooperation, fairness, human-centeredness,
and the RL environment remain experiment-layer work.
