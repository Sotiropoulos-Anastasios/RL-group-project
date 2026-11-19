# DialOp reviewer-assignment data for multi-agent RL

This package prepares the DialOp optimization task for the project
**Learning Prosocial Cooperation and Human-Centered Behavior in Multi-Agent AI
Systems Through Deep Reinforcement Learning**.

DialOp provides human-human conversations in which two players assign eight
reviewers to eight papers. Each player sees a different part of the reviewer-
paper score table, and each player's visible scores are independently scaled,
so communication is needed to make a good assignment.

The data preparation here is a foundation for the RL environment. It does not
yet add cooperation, fairness, human-centeredness, conflicting objectives, or
GRPO training. Those are experiment-layer features to be implemented on top of
the processed episodes.

## Repository layout

```text
dialop_project/
├── data/
│   ├── raw/                  # Downloaded DialOp inputs; ignored by Git
│   └── processed/            # Reproducible JSONL artifacts
├── scripts/
│   ├── audit_dataset.py      # Validate processed files and write reports
│   ├── build_dataset.py      # Build processed data from raw JSONL
│   └── visualize_dataset.py  # Write a simple Markdown data overview
├── src/
│   ├── environment.py        # Safe player-specific observation access
│   └── preprocessing.py      # Reusable conversion functions
├── reports/
│   ├── dataset_audit/        # JSON and CSV validation reports
│   └── visualizations/       # Generated Markdown summaries
├── explanation.md            # Plain-language data and proposal guide
└── README.md
```

## Setup

The build and audit scripts use only the Python standard library.

Python 3.10 or newer is recommended. From this directory:

```powershell
python scripts/build_dataset.py --help
```

## Raw data

Place the DialOp optimization file at:

```text
data/raw/optimization.jsonl
```

The current source archive and extracted files are retained locally in
`data/raw/`, but raw inputs are ignored by Git. This follows the same pattern as
the Hand-gesture project: the repository keeps the reproducible processing
code and processed artifact, while downloaded source data remains local.

The converter uses `optimization.jsonl`, not the itinerary or mediation games.

## Build the processed dataset

With the raw source file in place, run:

```powershell
python scripts/build_dataset.py
```

Optional positional arguments are available for a different input or output:

```powershell
python scripts/build_dataset.py path\to\optimization.jsonl path\to\output
```

The builder creates:

- `data/processed/episodes.jsonl` — one complete episode per line;
- `data/processed/actions.jsonl` — one normalized action per line; and
- `data/processed/manifest.json` — counts and split information.

The current processed artifact contains 134 episodes and 2,068 actions, split
deterministically into 94 train, 20 validation, and 20 test episodes.

## DialOp conventions preserved

The original DialOp implementation samples each player scale from
`Uniform(1, 10)` and displays an observed cell as:

```text
int(oracle_table_cell * player_scale)
```

Hidden cells are `null` in the processed private view. The oracle table stays
unchanged for scoring. The authoritative references are the [DialOp repository](https://github.com/microsoft/dialop),
[`optimization.py`](https://github.com/microsoft/dialop/blob/main/dialop/games/optimization.py),
and the [DialOp paper](https://arxiv.org/abs/2305.20076).

## Audit the processed data

Run:

```powershell
python scripts/audit_dataset.py
```

The audit checks:

- episode and action counts;
- unique episode IDs;
- 8x8 numeric oracle tables and private views;
- binary masks and mask/view consistency;
- scale range and exact visible-cell scaling;
- valid action kinds and player IDs;
- valid one-to-one proposal indices;
- processed outcomes against the raw source and reward formula;
- contiguous action steps;
- matching episode-level and flat-action records; and
- absence of oracle, scale, and private-view fields from flat action rows.

Reports are written to `reports/dataset_audit/`:

- `summary.json` — machine-readable validation summary;
- `episodes.csv` — one row per episode; and
- `actions.csv` — counts by action kind.

The command exits with status 0 when the audit passes and status 1 when it
finds an issue.

## Generate a simple overview

Run:

```powershell
python scripts/visualize_dataset.py
```

This writes `reports/visualizations/data_overview.md` with counts, action-type
bars, and score comparisons without requiring plotting dependencies.

## RL privacy boundary

`episodes.jsonl` contains the full `oracle_table` because the simulator and
evaluator need it to calculate assignment quality. The table must not be placed
directly in an agent prompt. The agent-facing helper returns only one player's
scaled private table and mask; it omits scale metadata, the oracle table, and
the other player's view.

Use `src/environment.py` to create a player-specific observation:

```python
from pathlib import Path

from environment import player_observation, read_episodes

episode = next(read_episodes(Path("data/processed/episodes.jsonl")))
observation = player_observation(episode, player=0)
```

The returned observation contains only that player's private table, mask,
workers, papers, and episode metadata. It excludes the full oracle table and
the other player's private view.

## Modeling boundary

DialOp contains human-human demonstrations and terminal assignment outcomes.
It does not contain explicit labels for cooperation, fairness, human utility,
exploitation, or stakeholder welfare. Add those as documented reward functions,
annotations, or synthetic scenario definitions rather than inferring them
from dialogue alone.

A reasonable first reward decomposition is:

```text
normalized_score   = proposal_reward / best_assignment_reward
cooperation        = agreement_progress - unnecessary_rejections
fairness           = stakeholder-specific fairness metric
human_centeredness = author/reviewer/attendee utility metric
communication     = bounded information gain - message cost
```

The normalized score is the original DialOp reward definition. Its reference
is the best matching selected from pooled observations, then scored with the
oracle table; it is not automatically a global full-information optimum. The
other terms are new experiment-layer objectives and are not labels in DialOp.

## Suggested experiment order

1. Evaluate an unmodified language-model agent.
2. Evaluate the same type of agent with cooperation-oriented instructions.
3. Train one agent with task-quality reward while holding the partner fixed.
4. Add cooperation, fairness, and human-centered reward terms.
5. Evaluate self-play only after the reward and metrics are stable.

For the full explanation of the data, processing steps, graphs, and proposal
connection, see [`explanation.md`](./explanation.md).
