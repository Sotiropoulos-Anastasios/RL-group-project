"""Safe access helpers for DialOp simulator observations.

The processed episode stores the full oracle table for the environment. This
module provides only a player's private observation so callers do not need to
manually select fields and risk leaking the oracle table or the other player's
view into an agent prompt. The returned table is already scaled using DialOp's
player-specific convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_episodes(path: Path) -> Iterator[dict[str, Any]]:
    """Yield complete processed episodes from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            yield record


def player_observation(episode: dict[str, Any], player: int) -> dict[str, Any]:
    """Return the information that one player may observe.

    The returned dictionary deliberately excludes ``oracle_table`` and all
    fields that reveal the other player's private view.
    """

    if player not in (0, 1):
        raise ValueError("player must be 0 or 1")
    private_views = episode.get("private_views")
    if not isinstance(private_views, list) or len(private_views) != 2:
        raise ValueError("episode must contain two private views")
    view = private_views[player]
    if not isinstance(view, dict):
        raise ValueError("private view must be an object")
    required = {"observed_mask", "table", "scale"}
    missing = required - set(view)
    if missing:
        raise ValueError(f"private view is missing fields: {sorted(missing)}")

    return {
        "episode_id": episode["episode_id"],
        "task": episode["task"],
        "split": episode["split"],
        "workers": episode["workers"],
        "papers": episode["papers"],
        "player": player,
        "observed_mask": view["observed_mask"],
        "table": view["table"],
    }
