#!/usr/bin/env python3
"""Audit processed DialOp files against the raw source and DialOp rules."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from preprocessing import (  # noqa: E402
    NUM_COLS,
    NUM_ROWS,
    SUPPORTED_ACTIONS,
    private_view,
    read_jsonl,
    validate_mask,
    validate_normalized_action,
    validate_proposal_ids,
    validate_scale,
    validate_score_table,
    validate_source_action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate processed DialOp episodes and action rows."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "processed",
    )
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=PROJECT_DIR / "data" / "raw" / "optimization.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "reports" / "dataset_audit",
    )
    return parser.parse_args()


def add_issue(issues: list[str], message: str) -> None:
    issues.append(message)


def display_input_path(path: Path) -> str:
    """Keep reports portable instead of recording a machine-specific path."""

    try:
        return path.resolve().relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        if path.name == "optimization.jsonl":
            return "data/raw/optimization.jsonl"
        return path.name


def close_enough(left: Any, right: Any) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    )


def check_raw_record(raw: dict[str, Any], index: int, issues: list[str]) -> None:
    """Validate source fields and source reward consistency."""

    episode_id = f"optimization-{index:04d}"
    try:
        validate_score_table(f"{episode_id}.table", raw.get("table"))
        validate_mask(f"{episode_id}.mask1", raw.get("mask1"))
        validate_mask(f"{episode_id}.mask2", raw.get("mask2"))
        validate_scale(f"{episode_id}.scale1", raw.get("scale1"))
        validate_scale(f"{episode_id}.scale2", raw.get("scale2"))
        action_log = raw.get("action_log", [])
        if not isinstance(action_log, list):
            add_issue(issues, f"{episode_id}.action_log must be a list")
            action_log = []
        for step, action in enumerate(action_log):
            validate_source_action(action, step, episode_id)
    except ValueError as exc:
        add_issue(issues, str(exc))

    result = raw.get("result") or {}
    score = result.get("score")
    best = raw.get("best_assignment_reward")
    norm = result.get("norm")
    if close_enough(raw.get("proposal_reward"), score) is False:
        add_issue(issues, f"{episode_id}: proposal_reward != result.score")
    if close_enough(raw.get("best_assignment_reward"), result.get("best")) is False:
        add_issue(issues, f"{episode_id}: best_assignment_reward != result.best")
    if close_enough(score, best) and norm == 1:
        return
    if isinstance(score, (int, float)) and isinstance(best, (int, float)) and best:
        expected_norm = float(score) / float(best)
        if not close_enough(norm, expected_norm):
            add_issue(
                issues,
                f"{episode_id}: result.norm != result.score / best_assignment_reward",
            )


def accepted_proposal_scores(
    raw: dict[str, Any], episode_id: str, issues: list[str]
) -> list[float]:
    """Calculate rewards of accepted raw proposals using the oracle table."""

    table = raw.get("table")
    try:
        validate_score_table(f"{episode_id}.table", table)
    except ValueError:
        return []
    action_log = raw.get("action_log", [])
    if not isinstance(action_log, list):
        return []
    current_proposal: list[list[int]] | None = None
    scores: list[float] = []
    for step, action in enumerate(action_log):
        if not isinstance(action, dict):
            continue
        if action.get("type") == "proposal":
            try:
                current_proposal = validate_proposal_ids(
                    action.get("proposal_ids"), f"{episode_id}.action_log[{step}]"
                )
            except ValueError:
                current_proposal = None
        elif action.get("type") == "proposal_response":
            response = action.get("response") or {}
            if response.get("accept"):
                if current_proposal is None or len(current_proposal) != NUM_ROWS:
                    add_issue(
                        issues,
                        f"{episode_id}.action_log[{step}]: accepted proposal is not a full matching",
                    )
                    continue
                scores.append(
                    float(sum(table[reviewer][paper] for reviewer, paper in current_proposal))
                )
    return scores


def main() -> int:
    args = parse_args()
    processed_paths = {
        "episodes": args.processed_dir / "episodes.jsonl",
        "actions": args.processed_dir / "actions.jsonl",
        "manifest": args.processed_dir / "manifest.json",
    }
    missing = [str(path) for path in (*processed_paths.values(), args.raw_input) if not path.exists()]
    if missing:
        print("Missing audit input files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    episodes = list(read_jsonl(processed_paths["episodes"]))
    actions = list(read_jsonl(processed_paths["actions"]))
    raw_records = list(read_jsonl(args.raw_input))
    manifest = json.loads(processed_paths["manifest"].read_text(encoding="utf-8"))
    issues: list[str] = []

    if len(episodes) != len(raw_records):
        add_issue(issues, f"episode count mismatch: processed={len(episodes)}, raw={len(raw_records)}")
    episode_ids = [episode.get("episode_id") for episode in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        add_issue(issues, "processed episode IDs are not unique")

    split_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    flat_action_counts: Counter[str] = Counter()
    flat_action_episode_ids: set[str] = set()
    flat_action_counts_by_episode: Counter[str] = Counter()
    episode_rows: list[dict[str, Any]] = []

    for index, episode in enumerate(episodes):
        episode_id = episode.get("episode_id", f"<missing-{index}>")
        expected_id = f"optimization-{index:04d}"
        if episode_id != expected_id:
            add_issue(issues, f"{episode_id}: expected episode ID {expected_id}")
        split = episode.get("split")
        if split not in {"train", "validation", "test"}:
            add_issue(issues, f"{episode_id}.split is invalid: {split!r}")
        split_counts[split] += 1

        try:
            oracle = validate_score_table(f"{episode_id}.oracle_table", episode.get("oracle_table"))
        except ValueError as exc:
            add_issue(issues, str(exc))
            oracle = None

        private_views = episode.get("private_views")
        if not isinstance(private_views, list) or len(private_views) != 2:
            add_issue(issues, f"{episode_id}.private_views must contain two views")
            private_views = []
        if not isinstance(episode.get("agent_scales"), list) or len(episode["agent_scales"]) != 2:
            add_issue(issues, f"{episode_id}.agent_scales must contain two scales")

        for player in range(2):
            if player >= len(private_views) or not isinstance(private_views[player], dict):
                continue
            view = private_views[player]
            context = f"{episode_id}.private_views[{player}]"
            try:
                scale = validate_scale(f"{context}.scale", view.get("scale"))
                mask = validate_mask(f"{context}.observed_mask", view.get("observed_mask"))
                expected_table = private_view(oracle, mask, scale) if oracle is not None else None
                if expected_table is not None and view.get("table") != expected_table:
                    add_issue(issues, f"{context}.table does not match oracle, mask, and scale")
                if isinstance(episode.get("agent_scales"), list) and player < len(episode["agent_scales"]):
                    if not close_enough(view.get("scale"), episode["agent_scales"][player]):
                        add_issue(issues, f"{context}.scale disagrees with agent_scales[{player}]")
            except ValueError as exc:
                add_issue(issues, str(exc))

        episode_actions = episode.get("actions")
        if not isinstance(episode_actions, list):
            add_issue(issues, f"{episode_id}.actions must be a list")
            episode_actions = []
        for expected_step, action in enumerate(episode_actions):
            try:
                validate_normalized_action(action, f"{episode_id}.actions[{expected_step}]")
            except ValueError as exc:
                add_issue(issues, str(exc))
                continue
            if action.get("step") != expected_step:
                add_issue(issues, f"{episode_id}.actions[{expected_step}].step is not contiguous")
            if action.get("kind") == "proposal_response" and action.get("actor") != action.get("response_from_player"):
                add_issue(issues, f"{episode_id}.actions[{expected_step}]: response actor mismatch")
            action_counts[action.get("kind")] += 1

        metrics = episode.get("metrics") or {}
        expected_metrics = {
            "num_actions": len(episode_actions),
            "num_messages": sum(action.get("kind") == "message" for action in episode_actions),
            "num_proposals": sum(action.get("kind") == "proposal" for action in episode_actions),
            "num_responses": sum(action.get("kind") == "proposal_response" for action in episode_actions),
            "num_accepted_responses": sum(
                action.get("kind") == "proposal_response" and action.get("accept")
                for action in episode_actions
            ),
            "dialogue_characters": sum(
                len(action.get("text", ""))
                for action in episode_actions
                if action.get("kind") == "message" and isinstance(action.get("text"), str)
            ),
        }
        for metric, expected_value in expected_metrics.items():
            if metrics.get(metric) != expected_value:
                add_issue(issues, f"{episode_id}.metrics.{metric} != computed value")

        if index < len(raw_records):
            raw = raw_records[index]
            check_raw_record(raw, index, issues)
            raw_scores = accepted_proposal_scores(raw, episode_id, issues)
            outcome = episode.get("human_outcome") or {}
            raw_result = raw.get("result") or {}
            comparisons = {
                "score": (outcome.get("score"), raw_result.get("score")),
                "proposal_reward": (outcome.get("proposal_reward"), raw.get("proposal_reward")),
                "best_score": (outcome.get("best_score"), raw_result.get("best")),
                "worst_score": (outcome.get("worst_score"), raw_result.get("worst")),
                "normalized_score": (outcome.get("normalized_score"), raw_result.get("norm")),
                "best_assignment_reward": (episode.get("best_assignment_reward"), raw.get("best_assignment_reward")),
            }
            for field, (processed_value, raw_value) in comparisons.items():
                if not close_enough(processed_value, raw_value):
                    add_issue(issues, f"{episode_id}: processed {field} disagrees with raw source")
            if raw_scores and not close_enough(raw_scores[-1], raw.get("proposal_reward")):
                add_issue(issues, f"{episode_id}: accepted proposal score disagrees with proposal_reward")
            if raw_result.get("best") and not close_enough(
                raw_result.get("norm"), raw_result.get("score") / raw_result.get("best")
            ):
                add_issue(issues, f"{episode_id}: normalized reward formula mismatch")

        outcome = episode.get("human_outcome") or {}
        episode_rows.append(
            {
                "episode_id": episode_id,
                "split": split,
                "num_actions": metrics.get("num_actions"),
                "num_messages": metrics.get("num_messages"),
                "num_proposals": metrics.get("num_proposals"),
                "num_responses": metrics.get("num_responses"),
                "scale1": (episode.get("agent_scales") or [None, None])[0],
                "scale2": (episode.get("agent_scales") or [None, None])[1],
                "score": outcome.get("score"),
                "best_assignment_reward": episode.get("best_assignment_reward"),
                "normalized_score": outcome.get("normalized_score"),
            }
        )

    for action in actions:
        episode_id = action.get("episode_id")
        flat_action_episode_ids.add(episode_id)
        flat_action_counts_by_episode[episode_id] += 1
        kind = action.get("kind")
        flat_action_counts[kind] += 1
        try:
            validate_normalized_action(action, f"flat_action[{episode_id}:{action.get('step')}]")
        except ValueError as exc:
            add_issue(issues, str(exc))
        if action.get("split") not in {"train", "validation", "test"}:
            add_issue(issues, f"flat_action[{episode_id}].split is invalid")
        if "oracle_table" in action or "private_views" in action or "agent_scales" in action:
            add_issue(issues, f"flat_action[{episode_id}]: private data leak")

    if set(episode_ids) != flat_action_episode_ids:
        add_issue(issues, "episode IDs differ between episode and action files")
    for episode in episodes:
        episode_id = episode.get("episode_id")
        expected = (episode.get("metrics") or {}).get("num_actions")
        if flat_action_counts_by_episode[episode_id] != expected:
            add_issue(issues, f"{episode_id}: flat action count mismatch")
        episode_split = episode.get("split")
        mismatched = [
            action
            for action in actions
            if action.get("episode_id") == episode_id and action.get("split") != episode_split
        ]
        if mismatched:
            add_issue(issues, f"{episode_id}: action split differs from episode split")

    normalized_scores = [
        episode["human_outcome"]["normalized_score"]
        for episode in episodes
        if episode.get("human_outcome", {}).get("normalized_score") is not None
    ]
    manifest_splits = manifest.get("episode_splits")
    if manifest_splits != dict(split_counts):
        add_issue(issues, "manifest episode_splits disagrees with processed episodes")
    if manifest.get("num_episodes") != len(episodes):
        add_issue(issues, "manifest num_episodes disagrees with processed episodes")
    if manifest.get("num_actions") != len(actions):
        add_issue(issues, "manifest num_actions disagrees with processed actions")

    summary = {
        "valid": not issues,
        "raw_input": display_input_path(args.raw_input),
        "episodes": len(episodes),
        "actions": len(actions),
        "split_counts": dict(sorted(split_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "flat_action_counts": dict(sorted(flat_action_counts.items())),
        "normalized_score": {
            "mean": round(mean(normalized_scores), 6) if normalized_scores else None,
            "min": min(normalized_scores) if normalized_scores else None,
            "max": max(normalized_scores) if normalized_scores else None,
        },
        "scaling_definition": (
            "DialOp: int(oracle_table_cell * player_scale) for visible cells; "
            "null for hidden cells; oracle_table remains unscaled."
        ),
        "reward_definition": (
            "normalized_score = proposal_reward / best_assignment_reward; "
            "best_assignment_reward is the pooled-observation reference from DialOp."
        ),
        "issues": issues,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if episode_rows:
        with (args.output_dir / "episodes.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
            writer.writeheader()
            writer.writerows(episode_rows)
    with (args.output_dir / "actions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "count"])
        writer.writeheader()
        writer.writerows(
            {"kind": kind, "count": count}
            for kind, count in sorted(flat_action_counts.items())
        )

    print(json.dumps(summary, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
