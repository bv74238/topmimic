"""Prepare PACE motion roles and compare seeded full-clip completion rates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir

import unisupermimic
from unisupermimic.rl.algos.supermimic.stage2 import (
    classify,
    completed_goal,
    manifest_identity,
    read_manifest,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import evaluate, make_eval_env
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import _load_csv


def _motions(additions: list[Path]) -> list[Path]:
    config_dir = Path(unisupermimic.__file__).resolve().parent / "conf" / "supermimic"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["task=g1_super_mimic/mujoco"])
    paths = [Path(path).resolve() for path in cfg.env.commands.motion.motion_files]
    paths.extend(path.expanduser().resolve() for path in additions)
    expected = {"freekick_g1", "marseille_turn", "tennis_g1"}
    if not expected.issubset({path.stem for path in paths}):
        missing = ", ".join(sorted(expected - {path.stem for path in paths}))
        raise ValueError(f"provide --motion for all three custom clips; missing: {missing}")
    names = [path.stem for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("motion CSV basenames must be unique")
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".csv":
            raise ValueError(f"missing motion CSV: {path}")
        data = _load_csv(str(path))
        if (
            data.ndim != 2
            or data.shape[0] < 2
            or data.shape[1] != 36
            or not bool(torch.isfinite(data).all())
        ):
            raise ValueError(f"motion must have at least two finite 36-column frames: {path}")
    return paths


def repeated(
    checkpoint: Path, paths: list[Path], *, backend: str, rollouts: int, seed: int
) -> dict:
    if rollouts < 1:
        raise ValueError("rollouts must be positive")
    env = make_eval_env(checkpoint.parent, backend, paths, random_initial=True, seed=seed)
    try:
        saved = torch.load(checkpoint, map_location=env.device, weights_only=True)
        results = []
        for index in range(rollouts):
            # Reset the command RNG per trial: early failures can otherwise consume
            # a different number of reset draws and shift all subsequent trials.
            env._supermimic_command._rng.manual_seed(seed + index)
            results.append(evaluate(env, saved))
        lengths = dict(
            zip(
                cast(Any, env.supermimic_mdp).motion.names,
                (cast(Any, env.supermimic_mdp).motion.clip_len - 1).tolist(),
                strict=True,
            )
        )
        env._supermimic_command.reset_noise = False
        deterministic = evaluate(env, saved)
    finally:
        env.close()
        cast(Any, env.backend).close()
    per_clip = {}
    for name in results[0]["per_clip"]:
        attempts = [result["per_clip"][name] for result in results]
        successes = sum(not item["failed"] and item["steps"] >= lengths[name] for item in attempts)
        # All runs end at the clip boundary or failure; a completion is a non-failed
        # episode whose length equals the known full reference length.
        per_clip[name] = {
            "completed": successes,
            "rollouts": rollouts,
            "completion_rate": successes / rollouts,
            "norm_return": sum(item["norm_return"] for item in attempts) / rollouts,
        }
    return {
        "checkpoint": str(checkpoint.resolve()),
        "seed": seed,
        "trial_seeds": list(range(seed, seed + rollouts)),
        "rollouts": rollouts,
        "per_clip": per_clip,
        "deterministic": deterministic,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--motion", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--baseline", type=Path, help="Evaluate: baseline report for objective comparison"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("mujoco", "mjwarp", "motrix"), default="mujoco")
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args(argv)
    checkpoint = args.ckpt.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    if args.seed < 0 or (args.rollouts is not None and args.rollouts < 1):
        parser.error("--seed must be nonnegative and --rollouts must be positive")
    if args.mode == "prepare":
        if args.rollouts is not None and args.rollouts != 5:
            parser.error("prepare requires five trials per clip")
        if args.baseline:
            parser.error("--baseline is only valid for evaluate")
        if args.manifest:
            parser.error("--manifest is only valid for evaluate")
        paths = _motions(args.motion)
    else:
        if not args.manifest:
            parser.error("evaluate requires --manifest")
        paths = [Path(item["path"]) for item in read_manifest(args.manifest)["clips"]]
    result = repeated(
        checkpoint,
        paths,
        backend=args.backend,
        rollouts=args.rollouts
        if args.rollouts is not None
        else (5 if args.mode == "prepare" else 20),
        seed=args.seed,
    )
    result["backend"] = args.backend
    if args.mode == "prepare":
        result = {
            "backend": args.backend,
            "base_checkpoint": str(checkpoint),
            "seed": args.seed,
            "rollouts": result["rollouts"],
            "trial_seeds": result["trial_seeds"],
            "threshold": 0.8,
            "deterministic": result["deterministic"],
            "clips": [
                {
                    "path": str(path),
                    "name": path.stem,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "role": classify(
                        result["per_clip"][path.stem]["completed"], result["rollouts"]
                    ),
                    **result["per_clip"][path.stem],
                }
                for path in paths
            ],
        }
        if {item["role"] for item in result["clips"]} != {"acquisition", "consolidation"}:
            raise ValueError("automatic split needs both roles; check motion data and baseline")
    else:
        manifest = read_manifest(args.manifest)
        result["manifest_identity"] = manifest_identity(manifest)
        for item in manifest["clips"]:
            result["per_clip"][item["name"]]["role"] = item["role"]
        result["groups"] = {
            role: sum(
                value["completion_rate"]
                for value in result["per_clip"].values()
                if value["role"] == role
            )
            / sum(value["role"] == role for value in result["per_clip"].values())
            for role in ("acquisition", "consolidation")
        }
        result["group_norm_return"] = {
            role: sum(
                value["norm_return"]
                for value in result["per_clip"].values()
                if value["role"] == role
            )
            / sum(value["role"] == role for value in result["per_clip"].values())
            for role in ("acquisition", "consolidation")
        }
        if args.baseline:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            if (
                baseline.get("manifest_identity") != result["manifest_identity"]
                or baseline.get("backend") != result["backend"]
                or baseline.get("seed") != result["seed"]
                or baseline.get("rollouts") != result["rollouts"]
                or set(baseline.get("per_clip", {})) != set(result["per_clip"])
            ):
                raise ValueError(
                    "baseline and new evaluations must use identical clips, seeds and rollouts"
                )
            result["target_met"] = completed_goal(baseline, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"output": str(args.output), "rollouts": result["rollouts"]}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
