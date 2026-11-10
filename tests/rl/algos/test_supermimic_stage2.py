"""PACE role/weight and STAR-inspired replay regression tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_supermimic_learner import FakeEnv, _config
from test_supermimic_replay import _transition
from unilab.managers.rng import ManagerRng

from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner
from unisupermimic.rl.algos.supermimic.replay import ReplayBuffer
from unisupermimic.rl.algos.supermimic.stage2 import (
    classify,
    completed_goal,
    consolidation_weight,
    matching_critic,
    read_manifest,
    selected_fragments,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.manager_terms import sample_start_frames


def test_group_threshold_and_goal() -> None:
    assert classify(4, 5) == "consolidation"
    assert classify(3, 5) == "acquisition"
    base = {"groups": {"acquisition": 0.3, "consolidation": 0.9}}
    assert completed_goal(base, {"groups": {"acquisition": 0.31, "consolidation": 0.88}})
    assert not completed_goal(base, {"groups": {"acquisition": 0.31, "consolidation": 0.87}})
    assert not completed_goal(base, {"groups": {"acquisition": 0.3, "consolidation": 0.9}})


def test_consolidation_weight_ema() -> None:
    ema, weight = consolidation_weight(1.0, 0.6)
    assert ema == pytest.approx(0.604)
    assert weight == pytest.approx(0.32)
    assert consolidation_weight(0.0, 0.6)[1] == pytest.approx(0.3)
    assert consolidation_weight(1.0, 1.0)[1] == 1.0


def test_manifest_detects_duplicate_names_missing_and_changed_csv(tmp_path: Path) -> None:
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("1,2\n")
    b.write_text("3,4\n")
    path = tmp_path / "group.json"
    clips = [{"path": str(a), "role": "acquisition"}, {"path": str(b), "role": "consolidation"}]
    path.write_text(json.dumps({"clips": clips}))
    manifest = read_manifest(path)
    assert all("sha256" in clip for clip in manifest["clips"])
    path.write_text(json.dumps(manifest))
    a.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        read_manifest(path)
    clips[1]["path"] = str(tmp_path / "other" / "a.csv")
    path.write_text(json.dumps({"clips": clips}))
    with pytest.raises(ValueError, match="missing"):
        read_manifest(path)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "a.csv").write_text("1,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path)


def test_role_bins_and_uniform_vs_adaptive_sampling() -> None:
    class Motion:
        device = torch.device("cpu")
        num_bins = 4
        bin_start = torch.tensor([0, 50, 100, 150])
        clip_end = torch.tensor([49, 99, 149, 199])
        bin_clip = torch.arange(4)

    motion = Motion()
    rng = ManagerRng.seeded(8)
    values = torch.tensor([2.0, 0.0, 0.0, 2.0])
    eligible = torch.tensor([1, 2])
    for adaptive in (False, True):
        frames, bins = sample_start_frames(
            motion, values, torch.ones(4) / 4, 0.99, 500, rng, eligible, adaptive=adaptive
        )
        assert set(bins.tolist()) == {1, 2}
        assert bool(((frames >= 50) & (frames < 150)).all())
    # Changing mastered-bin statistics cannot affect the acquisition draw.
    left = torch.tensor([100.0, 0.4, 0.8, -100.0])
    right = torch.tensor([-100.0, 0.4, 0.8, 100.0])
    first = sample_start_frames(
        motion, left, torch.ones(4) / 4, 0.99, 64, ManagerRng.seeded(17), eligible, adaptive=True
    )
    second = sample_start_frames(
        motion, right, torch.ones(4) / 4, 0.99, 64, ManagerRng.seeded(17), eligible, adaptive=True
    )
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[0], second[0])
    _, uniform = sample_start_frames(
        motion, values, torch.ones(4) / 4, 0.99, 4000, rng, eligible, adaptive=False
    )
    assert 0.45 < (uniform == 1).float().mean() < 0.55


def test_top_fragments_keep_boundaries_and_bin_selection() -> None:
    bins = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    score = torch.tensor([[1.0, 1.0, 9.0, 9.0], [8.0, 8.0, 2.0, 2.0]])
    done = torch.tensor([[False, True, False, False], [False, True, False, False]])
    selected, weights = selected_fragments(
        bins, score, done, torch.tensor([2.0, 3.0]), top_fraction=0.05
    )
    assert set(map(tuple, selected.tolist())) == {(1, 0), (1, 1), (0, 2), (0, 3)}
    assert set(weights.tolist()) == {2.0, 3.0}
    empty, _ = selected_fragments(bins, score, done, torch.zeros(2))
    assert empty.numel() == 0
    masked, _ = selected_fragments(
        bins,
        score,
        done,
        torch.tensor([2.0, 3.0]),
        mask=torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool),
    )
    assert all(bin_index == 0 for bin_index in bins[masked[:, 0], masked[:, 1]])


def test_star_windows_never_mix_roles_and_preserve_done() -> None:
    replay = ReplayBuffer(4, 2, "cpu", num_joints=2, context_dim=3)
    for step in range(4):
        replay.add(
            {**_transition(step, 2, 2, 3, step, done=step == 1), "bin": torch.tensor([3, 9])}
        )
    windows = replay.star_windows(8, 4, torch.tensor([0]))
    assert bool((windows["env"] == 0).all())
    assert bool((windows["bin"] == 3).all())
    assert bool(windows["done"][:, 1].all())


def test_checkpoint_critic_only_from_identical_base(tmp_path: Path) -> None:
    base = {"iteration": 12000, "wm": {"x": torch.tensor([1.0])}, "pi": {"x": torch.tensor([2.0])}}
    path = tmp_path / "ckpt_012000.pt"
    torch.save(base, path)
    assert matching_critic(path, base) is None
    full = {**base, "critic": {"x": torch.tensor([3.0])}}
    torch.save(full, tmp_path / "last.pt")
    assert matching_critic(path, base) is not None
    full["pi"] = {"x": torch.tensor([4.0])}
    torch.save(full, tmp_path / "last.pt")
    assert matching_critic(path, base) is None


def test_stage2_initializes_without_resuming_counters_and_star_ratio(tmp_path: Path) -> None:
    class Stage2Env(FakeEnv):
        stage2_acquisition_env = torch.tensor([True, False])
        stage2_difficulty = torch.tensor([1.0])

        def __init__(self) -> None:
            super().__init__()
            self.mdp.motion = SimpleNamespace(frame_bin=torch.zeros(100, dtype=torch.long))

    source = SuperMimicLearner(FakeEnv(), _config(), tmp_path / "base")
    base_path = source.save(tmp_path / "base" / "ckpt_012000.pt", include_optimizers=False)
    cfg = replace(
        _config(),
        stage2_enabled=True,
        stage2_base_checkpoint=str(base_path),
        stage2_manifest_sha256="identity",
        stage2_star_fraction=0.25,
    )
    learner = SuperMimicLearner(Stage2Env(), cfg, tmp_path / "stage2")
    learner.initialize_from(base_path)
    assert learner.iteration == 0
    assert learner.ref_pi is not None
    assert not learner._star_critic_ready
    learner.collect(3)
    # A valid selected transition from the acquisition environment only.
    learner.star_pool = (
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([1.0]),
        torch.tensor([1]),
    )
    learner.policy_update()
    assert learner.star_fraction_actual == pytest.approx(0.25)
    learner.star_pool = (
        torch.tensor([-1]),
        torch.tensor([0]),
        torch.tensor([1.0]),
        torch.tensor([1]),
    )
    learner.policy_update()
    assert learner.star_fraction_actual == 0.0


def test_stage2_cpu_short_training_and_resume_guard(tmp_path: Path) -> None:
    class Stage2Env(FakeEnv):
        stage2_acquisition_env = torch.tensor([True, False])
        stage2_difficulty = torch.tensor([1.0])

        def __init__(self) -> None:
            super().__init__()
            self.mdp.motion = SimpleNamespace(frame_bin=torch.zeros(100, dtype=torch.long))

    source = SuperMimicLearner(FakeEnv(), _config(), tmp_path / "source")
    base = source.save(tmp_path / "source" / "ckpt_012000.pt", include_optimizers=False)
    cfg = replace(
        _config(),
        stage2_enabled=True,
        stage2_base_checkpoint=str(base),
        stage2_manifest_sha256="original",
    )
    learner = SuperMimicLearner(Stage2Env(), cfg, tmp_path / "run")
    learner.initialize_from(base)
    learner.train()
    assert (tmp_path / "run" / "stage2_metrics.jsonl").is_file()
    with pytest.raises(ValueError, match="motion library identity"):
        SuperMimicLearner(
            Stage2Env(), replace(cfg, stage2_manifest_sha256="changed"), tmp_path / "another"
        ).load(tmp_path / "run" / "last.pt")


def test_repeated_evaluation_reseeds_each_trial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from unisupermimic.scripts import pace

    class Rng:
        current = -1

        def manual_seed(self, seed: int) -> None:
            self.current = seed

    class Env:
        device = torch.device("cpu")
        supermimic_mdp = SimpleNamespace(
            motion=SimpleNamespace(names=["x"], clip_len=torch.tensor([3]))
        )
        _supermimic_command = SimpleNamespace(_rng=Rng(), reset_noise=True)
        backend = SimpleNamespace(close=lambda: None)

        def close(self) -> None:
            pass

    env = Env()
    monkeypatch.setattr(pace, "make_eval_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(pace.torch, "load", lambda *args, **kwargs: {})
    seen: list[tuple[int, bool]] = []

    def fake_evaluate(env: Env, checkpoint: dict) -> dict:
        del checkpoint
        seen.append((env._supermimic_command._rng.current, env._supermimic_command.reset_noise))
        return {"per_clip": {"x": {"steps": 2, "failed": False, "norm_return": 0.9}}}

    monkeypatch.setattr(pace, "evaluate", fake_evaluate)
    result = pace.repeated(
        tmp_path / "base.pt", [tmp_path / "x.csv"], backend="mujoco", rollouts=3, seed=42
    )
    assert result["trial_seeds"] == [42, 43, 44]
    assert seen == [(42, True), (43, True), (44, True), (44, False)]
    assert result["per_clip"]["x"]["completion_rate"] == 1.0
