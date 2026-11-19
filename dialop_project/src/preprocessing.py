"""Convert DialOp optimization records into RL-friendly JSONL files.

The scaling rule in this module follows the original DialOp implementation in
``dialop/games/optimization.py``. For an observed cell, DialOp displays
``int(table_value * player_scale)``. Hidden cells are not displayed. The
oracle table remains unchanged for environment scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


TASKS = [
    "BLEU: a Method for Automatic Evaluation of MT",
    "Electra: Pre-training Text Encoders as Discriminators",
    "GloVe: Global Vectors for Word Representation",
    "GLUE: A Multi-Task Benchmark and Analysis Platform for NLU",
    "LLaMA: Open and Efficient Foundation Language Models",
    "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
    "QuAC: Question Answering in Context",
    "SWAG: An Adversarial Dataset for Commonsense Inference",
]

WORKERS = [
    "Ava Li",
    "Daniel Nguyen",
    "Sofia Patel",
    "Andrei Petrov",
    "Morgan Reed",
    "Joseph Santos",
    "Ethan Smith",
    "Noah Wilson",
]

NUM_ROWS = 8
NUM_COLS = 8
SCORE_MIN = 0
SCORE_MAX_EXCLUSIVE = 100
SCALE_MIN = 1.0
SCALE_MAX_EXCLUSIVE = 10.0
SUPPORTED_ACTIONS = {"message", "proposal", "proposal_response"}


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from a JSONL file with line-aware errors."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            yield record


def validate_score_table(name: str, table: Any) -> list[list[int | float]]:
    """Validate a DialOp 8x8 score table.

    The original generator creates integer scores in [0, 100), including 50
    for cells hidden from both players. This range is checked to catch source
    corruption while preserving the original oracle values exactly.
    """

    if not isinstance(table, list) or len(table) != NUM_ROWS:
        raise ValueError(f"{name} must have {NUM_ROWS} rows")
    for row_index, row in enumerate(table):
        if not isinstance(row, list) or len(row) != NUM_COLS:
            raise ValueError(f"{name}[{row_index}] must have {NUM_COLS} columns")
        for col_index, value in enumerate(row):
            if not _is_finite_number(value):
                raise ValueError(
                    f"{name}[{row_index}][{col_index}] must be a finite number"
                )
            if not SCORE_MIN <= value < SCORE_MAX_EXCLUSIVE:
                raise ValueError(
                    f"{name}[{row_index}][{col_index}] must be in "
                    f"[{SCORE_MIN}, {SCORE_MAX_EXCLUSIVE})"
                )
    return table


def validate_mask(name: str, mask: Any) -> list[list[int | bool]]:
    """Validate an 8x8 binary observation mask."""

    if not isinstance(mask, list) or len(mask) != NUM_ROWS:
        raise ValueError(f"{name} must have {NUM_ROWS} rows")
    for row_index, row in enumerate(mask):
        if not isinstance(row, list) or len(row) != NUM_COLS:
            raise ValueError(f"{name}[{row_index}] must have {NUM_COLS} columns")
        for col_index, value in enumerate(row):
            if value not in (0, 1, False, True):
                raise ValueError(
                    f"{name}[{row_index}][{col_index}] must be binary 0 or 1"
                )
    return mask


def validate_scale(name: str, scale: Any) -> float:
    """Validate a player scale using DialOp's Uniform(1, 10) convention."""

    if not _is_finite_number(scale):
        raise ValueError(f"{name} must be a finite number")
    if not SCALE_MIN <= float(scale) < SCALE_MAX_EXCLUSIVE:
        raise ValueError(
            f"{name} must be in [{SCALE_MIN}, {SCALE_MAX_EXCLUSIVE})"
        )
    return float(scale)


def validate_proposal_ids(
    proposal_ids: Any, context: str
) -> list[list[int]]:
    """Validate a proposal as a partial or full one-to-one matching."""

    if not isinstance(proposal_ids, list) or not proposal_ids:
        raise ValueError(f"{context}.proposal_ids must be a non-empty list")
    if len(proposal_ids) > NUM_ROWS:
        raise ValueError(f"{context}.proposal_ids cannot contain more than 8 pairs")

    pairs: list[list[int]] = []
    reviewers: set[int] = set()
    papers: set[int] = set()
    for pair_index, pair in enumerate(proposal_ids):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"{context}.proposal_ids[{pair_index}] must be [reviewer, paper]")
        reviewer, paper = pair
        if any(not isinstance(value, int) or isinstance(value, bool) for value in pair):
            raise ValueError(
                f"{context}.proposal_ids[{pair_index}] must contain integer indices"
            )
        if not 0 <= reviewer < NUM_ROWS or not 0 <= paper < NUM_COLS:
            raise ValueError(
                f"{context}.proposal_ids[{pair_index}] has an index outside 0..7"
            )
        if reviewer in reviewers:
            raise ValueError(f"{context}.proposal_ids repeats reviewer index {reviewer}")
        if paper in papers:
            raise ValueError(f"{context}.proposal_ids repeats paper index {paper}")
        reviewers.add(reviewer)
        papers.add(paper)
        pairs.append([reviewer, paper])
    return pairs


def _validate_player(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
        raise ValueError(f"{context} must be player ID 0 or 1")
    return value


def validate_source_action(action: Any, step: int, episode_id: str) -> None:
    """Validate one nested raw DialOp action."""

    context = f"{episode_id}.action_log[{step}]"
    if not isinstance(action, dict):
        raise ValueError(f"{context} must be an object")
    kind = action.get("type")
    if kind not in SUPPORTED_ACTIONS:
        raise ValueError(f"{context}.type {kind!r} is not supported")
    actor = _validate_player(action.get("player"), f"{context}.player")
    if "time" in action and (
        not _is_finite_number(action["time"]) or action["time"] < 0
    ):
        raise ValueError(f"{context}.time must be a non-negative finite number")

    if kind == "message":
        message = action.get("message")
        if not isinstance(message, dict):
            raise ValueError(f"{context}.message must be an object")
        message_actor = _validate_player(
            message.get("from_player", actor), f"{context}.message.from_player"
        )
        if message_actor != actor:
            raise ValueError(f"{context}.player disagrees with message.from_player")
        if not isinstance(message.get("data"), str):
            raise ValueError(f"{context}.message.data must be text")
        if not isinstance(message.get("type", "utterance"), str):
            raise ValueError(f"{context}.message.type must be text")
    elif kind == "proposal":
        validate_proposal_ids(action.get("proposal_ids"), context)
        if not isinstance(action.get("proposal"), str):
            raise ValueError(f"{context}.proposal must be text")
    else:
        response = action.get("response")
        if not isinstance(response, dict):
            raise ValueError(f"{context}.response must be an object")
        if not isinstance(response.get("accept"), bool):
            raise ValueError(f"{context}.response.accept must be boolean")
        response_actor = _validate_player(
            response.get("from_player"), f"{context}.response.from_player"
        )
        if response_actor != actor:
            raise ValueError(f"{context}.player disagrees with response.from_player")
        bonus_pay = action.get("bonus_pay", 0)
        if bonus_pay is not None and not _is_finite_number(bonus_pay):
            raise ValueError(f"{context}.bonus_pay must be numeric or null")


def validate_normalized_action(action: Any, context: str) -> None:
    """Validate one processed action row."""

    if not isinstance(action, dict):
        raise ValueError(f"{context} must be an object")
    if not isinstance(action.get("step"), int) or action["step"] < 0:
        raise ValueError(f"{context}.step must be a non-negative integer")
    kind = action.get("kind")
    if kind not in SUPPORTED_ACTIONS:
        raise ValueError(f"{context}.kind {kind!r} is not supported")
    _validate_player(action.get("actor"), f"{context}.actor")
    if kind == "message":
        if not isinstance(action.get("text"), str):
            raise ValueError(f"{context}.text must be text")
    elif kind == "proposal":
        validate_proposal_ids(action.get("proposal_ids"), context)
        if not isinstance(action.get("proposal_html"), str):
            raise ValueError(f"{context}.proposal_html must be text")
    else:
        if not isinstance(action.get("accept"), bool):
            raise ValueError(f"{context}.accept must be boolean")
        _validate_player(action.get("response_from_player"), f"{context}.response_from_player")


def normalize_action(action: dict[str, Any], step: int, episode_id: str) -> dict[str, Any]:
    """Convert one nested DialOp action into a common flat representation."""

    validate_source_action(action, step, episode_id)
    kind = action["type"]
    normalized: dict[str, Any] = {
        "step": step,
        "kind": kind,
        "actor": action["player"],
    }
    if "time" in action:
        normalized["elapsed_seconds"] = action["time"]

    if kind == "message":
        message = action["message"]
        normalized.update(
            {
                "actor": message.get("from_player", action["player"]),
                "text": message["data"],
                "message_type": message.get("type", "utterance"),
            }
        )
    elif kind == "proposal":
        normalized.update(
            {
                "proposal_ids": action["proposal_ids"],
                "proposal_html": action["proposal"],
            }
        )
    else:
        response = action["response"]
        normalized.update(
            {
                "accept": response["accept"],
                "response_from_player": response["from_player"],
                "bonus_pay": action.get("bonus_pay", 0),
            }
        )
    return normalized


def private_view(
    table: list[list[int | float]],
    mask: list[list[int | bool]],
    scale: float,
) -> list[list[int | None]]:
    """Create the exact DialOp-scaled player table with hidden cells as null."""

    validate_scale("scale", scale)
    return [
        [
            int(table[row][col] * scale) if bool(mask[row][col]) else None
            for col in range(NUM_COLS)
        ]
        for row in range(NUM_ROWS)
    ]


def deterministic_episode_splits(
    episode_ids: list[str], validation_fraction: float = 0.15, test_fraction: float = 0.15
) -> dict[str, str]:
    """Assign exact episode-level splits by sorting stable SHA-256 hashes.

    The default target is 70% train, 15% validation, and 15% test. Sorting
    hashes makes the result independent of source-file order while keeping all
    turns from an episode together.
    """

    if not episode_ids or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode_ids must be a non-empty unique list")
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("split fractions must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must leave training data")

    ordered = sorted(
        episode_ids,
        key=lambda episode_id: hashlib.sha256(episode_id.encode()).hexdigest(),
    )
    validation_count = max(1, round(len(ordered) * validation_fraction))
    test_count = max(1, round(len(ordered) * test_fraction))
    if validation_count + test_count >= len(ordered):
        raise ValueError("split fractions leave no training episodes")
    split_map = {episode_id: "train" for episode_id in ordered}
    for episode_id in ordered[-test_count:]:
        split_map[episode_id] = "test"
    validation_end = len(ordered) - test_count
    for episode_id in ordered[validation_end - validation_count : validation_end]:
        split_map[episode_id] = "validation"
    return split_map


def _validate_reward_value(name: str, value: Any) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def convert_record(
    record: dict[str, Any], index: int, split: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Convert one source record into an episode and flat action rows."""

    episode_id = f"optimization-{index:04d}"
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"{episode_id}.split is invalid: {split!r}")
    table = validate_score_table(f"{episode_id}.table", record.get("table"))
    mask1 = validate_mask(f"{episode_id}.mask1", record.get("mask1"))
    mask2 = validate_mask(f"{episode_id}.mask2", record.get("mask2"))
    scale1 = validate_scale(f"{episode_id}.scale1", record.get("scale1"))
    scale2 = validate_scale(f"{episode_id}.scale2", record.get("scale2"))
    action_log = record.get("action_log")
    if not isinstance(action_log, list) or not action_log:
        raise ValueError(f"{episode_id}.action_log must be a non-empty list")
    actions = [
        normalize_action(action, step, episode_id)
        for step, action in enumerate(action_log)
    ]

    final_result = record.get("result")
    if not isinstance(final_result, dict):
        raise ValueError(f"{episode_id}.result must be an object")
    proposal_reward = _validate_reward_value(
        f"{episode_id}.proposal_reward", record.get("proposal_reward")
    )
    final_score = _validate_reward_value(
        f"{episode_id}.result.score", final_result.get("score")
    )
    best_reward = _validate_reward_value(
        f"{episode_id}.best_assignment_reward", record.get("best_assignment_reward")
    )
    best_score = _validate_reward_value(
        f"{episode_id}.result.best", final_result.get("best")
    )
    worst_score = _validate_reward_value(
        f"{episode_id}.result.worst", final_result.get("worst")
    )
    normalized_score = _validate_reward_value(
        f"{episode_id}.result.norm", final_result.get("norm")
    )
    if not math.isclose(proposal_reward, final_score, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"{episode_id}: proposal_reward disagrees with result.score")
    if not math.isclose(best_reward, best_score, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"{episode_id}: best_assignment_reward disagrees with result.best"
        )
    if best_reward <= 0:
        raise ValueError(f"{episode_id}.best_assignment_reward must be positive")
    expected_norm = final_score / best_reward
    if not math.isclose(normalized_score, expected_norm, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"{episode_id}: result.norm must equal result.score / best_assignment_reward"
        )

    response_actions = [
        action for action in actions if action["kind"] == "proposal_response"
    ]
    messages = [action for action in actions if action["kind"] == "message"]
    proposals = [action for action in actions if action["kind"] == "proposal"]
    accepted = sum(1 for action in response_actions if action.get("accept"))

    episode = {
        "episode_id": episode_id,
        "task": "reviewer_assignment",
        "split": split,
        "workers": WORKERS,
        "papers": TASKS,
        "oracle_table": table,
        "private_views": [
            {
                "player": 0,
                "scale": scale1,
                "observed_mask": mask1,
                "table": private_view(table, mask1, scale1),
            },
            {
                "player": 1,
                "scale": scale2,
                "observed_mask": mask2,
                "table": private_view(table, mask2, scale2),
            },
        ],
        "agent_scales": [scale1, scale2],
        "best_assignment_reward": best_reward,
        "human_outcome": {
            "score": final_score,
            "proposal_reward": proposal_reward,
            "best_score": best_score,
            "worst_score": worst_score,
            "normalized_score": normalized_score,
        },
        "actions": actions,
        "metrics": {
            "num_actions": len(actions),
            "num_messages": len(messages),
            "num_proposals": len(proposals),
            "num_responses": len(response_actions),
            "num_accepted_responses": accepted,
            "dialogue_characters": sum(
                len(action.get("text", "")) for action in messages
            ),
        },
    }

    flat_actions = [
        {
            "episode_id": episode_id,
            "task": "reviewer_assignment",
            "split": split,
            **action,
        }
        for action in actions
    ]
    return episode, flat_actions


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write dictionaries as compact UTF-8 JSONL and return the row count."""

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
    return count


def build_dataset(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Build processed episode and flat-action files from DialOp JSONL."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(input_path))
    episode_ids = [f"optimization-{index:04d}" for index in range(len(records))]
    split_map = deterministic_episode_splits(episode_ids)
    episodes: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        episode, flat_actions = convert_record(
            record, index, split_map[episode_ids[index]]
        )
        episodes.append(episode)
        actions.extend(flat_actions)

    write_jsonl(output_dir / "episodes.jsonl", episodes)
    write_jsonl(output_dir / "actions.jsonl", actions)
    splits = {
        split: sum(episode["split"] == split for episode in episodes)
        for split in ("train", "validation", "test")
    }
    try:
        source_name = input_path.relative_to(output_dir.parent.parent).as_posix()
    except ValueError:
        source_name = str(input_path)
    manifest = {
        "source": source_name,
        "task": "reviewer_assignment",
        "num_episodes": len(episodes),
        "num_actions": len(actions),
        "episode_splits": splits,
        "split_method": (
            "Sort episode IDs by SHA-256 digest; assign the final 15% to test, "
            "the preceding 15% to validation, and the remainder to train."
        ),
        "reward_definition": (
            "normalized_score = proposal_reward / best_assignment_reward; "
            "best_assignment_reward is DialOp's pooled-observation reference."
        ),
        "scaling_definition": (
            "visible_player_cell = int(oracle_table_cell * player_scale); "
            "hidden_player_cell = null; oracle_table is unchanged."
        ),
        "privacy_note": (
            "oracle_table is for the environment/evaluator; expose only "
            "one player's scaled private view to an agent"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
