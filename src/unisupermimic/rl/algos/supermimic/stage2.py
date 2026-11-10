"""Optional PACE expansion and STAR-inspired replay selection (not PPO STAR)."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch


def read_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = manifest.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("stage2 manifest requires a nonempty clips list")
    names: set[str] = set()
    roles: set[str] = set()
    for clip in clips:
        file = Path(clip["path"]).expanduser().resolve()
        if file.suffix.lower() != ".csv" or not file.is_file():
            raise ValueError(f"missing or invalid motion CSV: {file}")
        if file.stem in names:
            raise ValueError(f"duplicate motion name: {file.stem}")
        names.add(file.stem)
        if clip.get("role") not in ("acquisition", "consolidation"):
            raise ValueError(f"invalid motion role for {file}")
        if clip.get("name", file.stem) != file.stem:
            raise ValueError(f"motion name differs from CSV: {file}")
        clip["name"] = file.stem
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if "sha256" in clip and clip["sha256"] != digest:
            raise ValueError(f"motion CSV changed after grouping: {file}")
        clip["sha256"] = digest
        roles.add(clip["role"])
        clip["path"] = str(file)
    if roles != {"acquisition", "consolidation"}:
        raise ValueError("stage2 requires both acquisition and consolidation clips")
    return cast(dict[str, Any], manifest)


def manifest_identity(manifest: dict) -> str:
    identity = [(item["path"], item["role"], item["sha256"]) for item in manifest["clips"]]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def completed_goal(baseline: dict, trained: dict) -> bool:
    return bool(
        trained["groups"]["acquisition"] > baseline["groups"]["acquisition"]
        and trained["groups"]["consolidation"] >= baseline["groups"]["consolidation"] - 0.02
    )


def classify(completed: int, rollouts: int, threshold: float = 0.8) -> str:
    if not 0 <= completed <= rollouts or rollouts < 1:
        raise ValueError("invalid completion counts")
    return "consolidation" if completed / rollouts >= threshold else "acquisition"


def consolidation_weight(
    ratio: float,
    ema: float,
    beta: float = 0.99,
    base: float = 0.3,
    gain: float = 5.0,
    reference: float = 0.6,
) -> tuple[float, float]:
    updated = beta * ema + (1 - beta) * ratio
    return updated, min(1.0, base + gain * max(0.0, updated - reference))


def selected_fragments(
    bins: torch.Tensor,
    scores: torch.Tensor,
    done: torch.Tensor,
    difficulty: torch.Tensor,
    *,
    top_fraction: float = 0.05,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select high-value contiguous fragments independently per eligible hard bin.

    Inputs are (window, time). Window boundaries and done rows end a fragment.
    Returns (window, time) indices of complete selected fragments and their
    mean difficulty, usable for weighted replay sampling.
    """
    if bins.shape != scores.shape or bins.shape != done.shape:
        raise ValueError("STAR input shapes differ")
    pairs: dict[int, list[tuple[float, int]]] = defaultdict(list)
    fragments: list[list[tuple[int, int]]] = []
    weights: list[float] = []
    for w in range(bins.shape[0]):
        fragment: list[tuple[int, int]] = []
        for t in range(bins.shape[1]):
            if mask is not None and not bool(mask[w, t]):
                break
            fragment.append((w, t))
            if bool(done[w, t]) or t == bins.shape[1] - 1:
                idx = len(fragments)
                fragments.append(fragment)
                weights.append(
                    sum(float(difficulty[bins[i, j]]) for i, j in fragment) / len(fragment)
                )
                by_bin: dict[int, list[float]] = defaultdict(list)
                for i, j in fragment:
                    b = int(bins[i, j])
                    if float(difficulty[b]) > 1.0:
                        by_bin[b].append(float(scores[i, j]))
                for b, values in by_bin.items():
                    pairs[b].append((sum(values) / len(values), idx))
                fragment = []
    chosen: set[int] = set()
    for candidates in pairs.values():
        count = max(1, math.ceil(top_fraction * len(candidates)))
        chosen.update(idx for _, idx in sorted(candidates, reverse=True)[:count])
    if not chosen:
        return torch.empty((0, 2), dtype=torch.long, device=bins.device), torch.empty(
            0, device=bins.device
        )
    indices = [pair for index in sorted(chosen) for pair in fragments[index]]
    priorities = [weights[index] for index in sorted(chosen) for _ in fragments[index]]
    return torch.tensor(indices, dtype=torch.long, device=bins.device), torch.tensor(
        priorities, device=bins.device
    ).clamp_min(1e-6)


def matching_critic(path: Path, checkpoint: dict) -> dict | None:
    """Use a critic only when its full checkpoint has identical WM/policy tensors."""
    if "critic" in checkpoint:
        return cast(dict[str, Any], checkpoint["critic"])
    candidate = path.with_name("last.pt")
    if not candidate.is_file() or candidate == path:
        return None
    full = torch.load(candidate, map_location="cpu", weights_only=True)
    if full.get("iteration") != checkpoint.get("iteration"):
        return None
    for key in ("wm", "pi"):
        a, b = checkpoint[key], full.get(key, {})
        if a.keys() != b.keys() or any(not torch.equal(a[name].cpu(), b[name]) for name in a):
            return None
    return cast(dict[str, Any] | None, full.get("critic"))
