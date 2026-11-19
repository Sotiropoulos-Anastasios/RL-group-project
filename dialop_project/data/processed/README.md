# Processed DialOp data

These files are generated from `data/raw/optimization.jsonl`:

- `episodes.jsonl`: complete simulator episodes, including the unchanged oracle
  table, player-scaled private views, actions, and outcomes.
- `actions.jsonl`: flattened actions for dialogue analysis. It intentionally
  omits the oracle table and private views.
- `manifest.json`: counts, split sizes, and the privacy boundary.

Regenerate them with:

```powershell
python scripts/build_dataset.py
```

Visible private cells follow the original DialOp rule:
`int(oracle_table_cell * player_scale)`. Hidden cells are `null`. Do not pass
`oracle_table` directly to a language-model agent. Use the safe helper in
`src/environment.py` to obtain one player's private observation.
