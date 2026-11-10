"""SuperMimic training over the injected environment and MDP protocols."""

from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from unilab.rl.contracts.env import EnvProtocol

from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
    State,
    get_supermimic_mdp,
)
from unisupermimic.rl.logging import SuperMimicLogger

from . import networks
from .geometry import unflatten_state
from .networks import Critic, Policy, WorldModel, mlp
from .replay import ReplayBuffer
from .stage2 import consolidation_weight, matching_critic, selected_fragments


@dataclass
class SuperMimicConfig:
    iterations: int = 320_000
    buffer_steps: int = 1000
    utd: float = 0.4
    horizon: int = 32
    goal_frames: int = 10
    hist: int = 10
    z_dim: int = 32
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    policy_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden: tuple[int, ...] = (256, 128)
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    lr: float = 2e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    warmup_steps: int = 64
    wm_only_iters: int = 500
    lr_decay_iters: int = 250_000
    ckpt_every: int = 2000
    log_every: int = 50
    log_backend: str = "tensorboard"
    seed: int = 0
    amp: str = "bf16"
    compile: bool = True
    test_hidden_dims: tuple[int, ...] | None = None
    stage2_enabled: bool = False
    stage2_base_checkpoint: str = ""
    stage2_manifest: str = ""
    stage2_manifest_sha256: str = ""
    stage2_fraction: float = 0.8
    stage2_star_fraction: float = 0.25
    stage2_star_top: float = 0.05
    stage2_star_refresh: int = 50


_LOG_FIELDS = (
    "iteration",
    "wall_h",
    "sim_steps",
    "wm_loss",
    "wm_err8",
    "pi_loss",
    "v_loss",
    "imag_rew",
    "pi_gradnorm",
    "imag_survival",
    "sim_rew",
    "ep_len",
    "ep_ret_norm",
    "term_frac",
    "it_per_s",
    "lr",
)


def _empty_stats() -> dict[str, list[torch.Tensor]]:
    return {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class SuperMimicLearner:
    def __init__(
        self,
        env: EnvProtocol,
        cfg: SuperMimicConfig,
        log_dir: str | Path,
        device: str | torch.device | None = None,
        backend: str = "",
    ):
        if cfg.amp not in ("none", "bf16"):
            raise ValueError(f"amp must be 'none' or 'bf16', got {cfg.amp!r}")
        if cfg.utd <= 0:
            raise ValueError("utd must be positive")
        if cfg.test_hidden_dims is not None and not cfg.test_hidden_dims:
            raise ValueError("test_hidden_dims must be nonempty when provided")
        if cfg.stage2_enabled and (cfg.no_critic or not 0 <= cfg.stage2_star_fraction < 1):
            raise ValueError("stage2 requires a critic and a STAR fraction in [0, 1)")
        if cfg.stage2_enabled and not (
            0 < cfg.stage2_star_top <= 1 and cfg.stage2_star_refresh > 0
        ):
            raise ValueError("stage2 requires a top fraction in (0, 1] and positive refresh")

        self.env = env
        self.cfg = cfg
        self.backend = backend
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_device = torch.device(env.device)
        self.device = self.env_device if device is None else torch.device(device)
        # HOST_BRIDGE backends (e.g. MuJoCo) keep env tensors on CPU while the
        # learner may run on an accelerator (MPS on macOS); when the two differ,
        # the learner works on its own MDP copy and bridges tensors at the env
        # boundary. On CUDA/ROCm the devices always match and nothing changes.
        self._bridged = self.device != self.env_device
        mdp = get_supermimic_mdp(env)
        self.mdp = mdp.to(self.device) if self._bridged else mdp

        # TF32 only pays off for fp32 (amp="none") training: under bf16 autocast
        # the heavy GEMMs already run in bf16, while on ROCm a TF32 bmm over the
        # misaligned strided views produced by unflatten_state triggers per-call
        # rocBLAS kernel selection costing ~100 ms per call.
        torch.backends.cuda.matmul.allow_tf32 = cfg.amp == "none"
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed)
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None

        self.wm = self._make_world_model().to(self.device)
        self.pi = self._make_policy().to(self.device)
        self.critic = Critic(self._critic_input_dim(), hidden=cfg.critic_hidden).to(self.device)
        if cfg.test_hidden_dims is not None:
            self.critic.net = mlp(self.critic.norm.mean.numel(), cfg.test_hidden_dims, 1).to(
                self.device
            )
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=cfg.lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.ref_pi: Policy | None = None
        self.stage2_ema = 0.6
        self.stage2_lambda = 0.3
        self.star_pool: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.star_fraction_actual = 0.0
        self._last_star_sample_count = 0
        self._star_critic_ready = False
        self.consolidation_loss = 0.0
        self.star_observed_completion: float | None = None
        self.regular_observed_completion: float | None = None
        if cfg.stage2_enabled:
            roles = getattr(env, "stage2_acquisition_env", None)
            if roles is None or not bool(roles.any()) or bool(roles.all()):
                raise ValueError("stage2 requires both acquisition and consolidation environments")
            self.acquisition_ids = roles.to(self.device).nonzero().flatten()
            self.consolidation_ids = (~roles.to(self.device)).nonzero().flatten()

        self.buffer = ReplayBuffer(
            cfg.buffer_steps,
            env.num_envs,
            self.device,
            num_joints=self.mdp.num_joints,
            context_dim=self.mdp.context_dim,
        )
        self.obs_hist = torch.zeros(env.num_envs, cfg.hist, self.mdp.obs_dim, device=self.device)
        self.err_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((self.mdp.num_joints,), 3e-3),
                torch.full((self.mdp.num_joints,), 5e-2),
            )
        ).to(self.device)

        self.iteration = 0
        self.sim_steps = 0
        self.wall_time = 0.0
        self.stats = _empty_stats()
        self._wm_rollout_fn = self._compile(self._wm_rollout)
        self._wm_err_fn = self._compile(self._wm_err)
        self._imag_step_fn = self._compile(self._imag_step)
        self._reward_term_fn = self._compile(self._reward_term)
        self._critic_input_fn = self._compile(self._critic_input)

        with (self.log_dir / "args.json").open("w", encoding="utf-8") as stream:
            json.dump(asdict(cfg), stream, indent=2)

    def _compile(self, function):
        if not self.cfg.compile:
            return function
        if self.device.type == "mps":
            # Inductor Metal codegen is not usable on MPS; keep AOT graph
            # capture without kernel generation.
            return torch.compile(function, backend="aot_eager")
        return torch.compile(function)

    def _make_world_model(self) -> WorldModel:
        kwargs = {
            "num_joints": self.mdp.num_joints,
            "context_dim": self.mdp.context_dim,
        }
        if self.cfg.test_hidden_dims is not None:
            return WorldModel(hidden=self.cfg.test_hidden_dims, **kwargs)
        return WorldModel(hidden=self.cfg.wm_hidden, **kwargs)

    def _make_policy(self) -> Policy:
        goal_dim = self.mdp.goal_frame_dim * self.cfg.goal_frames
        policy = Policy(
            self.mdp.obs_dim,
            goal_dim,
            hist=self.cfg.hist,
            z_dim=self.cfg.z_dim,
            num_joints=self.mdp.num_joints,
            hidden=self.cfg.policy_hidden,
        )
        if self.cfg.test_hidden_dims is not None:
            enc_output = policy.enc[-1]
            assert isinstance(enc_output, nn.Linear)
            z_dim = enc_output.out_features
            policy.enc = mlp(goal_dim, self.cfg.test_hidden_dims, z_dim)
            policy.dec = mlp(
                z_dim + self.mdp.obs_dim * self.cfg.hist,
                self.cfg.test_hidden_dims,
                self.mdp.num_joints,
            )
            dec_output = policy.dec[-1]
            assert isinstance(dec_output, nn.Linear)
            with torch.no_grad():
                dec_output.weight.mul_(0.01)
                dec_output.bias.zero_()
        return policy

    def _state_info(self) -> dict[str, torch.Tensor]:
        state = self.env.state
        if state is None:
            state = self.env.init_state()
        info: dict[str, torch.Tensor] = state.info[SUPERMIMIC_STATE_KEY]
        if self._bridged:
            info = {key: value.to(self.device) for key, value in info.items()}
        return info

    def _critic_input_dim(self) -> int:
        info = self._state_info()
        with torch.no_grad():
            features = self.mdp.critic_features(
                unflatten_state(info["s"]),
                info["f"],
                self.mdp.normalize_context(info["xi"]),
            )
        return int(features.shape[-1] + self.mdp.goal_frame_dim * self.cfg.goal_frames)

    def _wm_rollout(
        self, states: torch.Tensor, actions: torch.Tensor, contexts: torch.Tensor
    ) -> State:
        state = unflatten_state(states[:, 0])
        predictions = []
        for index in range(self.cfg.wm_window):
            state = self.wm.step(
                state,
                actions[:, index],
                self.mdp.normalize_context(contexts[:, index]),
            )
            predictions.append(state)
        return (
            torch.stack([prediction[0] for prediction in predictions], 1),
            torch.stack([prediction[1] for prediction in predictions], 1),
            torch.stack([prediction[2] for prediction in predictions], 1),
            torch.stack([prediction[3] for prediction in predictions], 1),
            torch.stack([prediction[4] for prediction in predictions], 1),
            torch.stack([prediction[5] for prediction in predictions], 1),
        )

    def _wm_err(
        self, predicted: State, next_states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.maximum(self.wm.out_norm.std(), self.err_floor)
        error = (self.wm.state_error(predicted, unflatten_state(next_states)) / std) ** 2
        error = error.mean(-1)
        return (error * mask).sum() / next_states.shape[0], error[:, -1]

    def _imag_step(
        self,
        state: State,
        obs_hist: torch.Tensor,
        previous_action: torch.Tensor,
        frames: torch.Tensor,
        context: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> tuple[State, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = self.pi(obs_hist, goal)
        next_state = self.wm.step(state, action, normalized_context)
        next_frames = frames + 1
        obs = self.mdp.add_noise(self.mdp.obs(next_state, action, context, next_frames))
        next_hist = torch.cat((obs_hist[:, 1:], obs.unsqueeze(1)), 1)
        next_goal = self.mdp.goal(next_frames, self.cfg.goal_frames)
        return next_state, action, next_hist, next_frames, next_goal

    def _reward_term(
        self,
        state: State,
        frames: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mdp.reward(state, frames, action, previous_action), self.mdp.soft_termination(
            state, frames
        )

    def _critic_input(
        self,
        state: State,
        frames: torch.Tensor,
        normalized_context: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((self.mdp.critic_features(state, frames, normalized_context), goal), -1)

    @torch.no_grad()
    def collect(self, n_steps: int) -> None:
        for _ in range(n_steps):
            info = self._state_info()
            state = unflatten_state(info["s"])
            obs = self.mdp.add_noise(self.mdp.obs(state, info["a_prev"], info["xi"], info["f"]))
            fresh = info["ep_step"] == 0
            shifted = torch.cat((self.obs_hist[:, 1:], obs.unsqueeze(1)), 1)
            self.obs_hist = torch.where(
                fresh[:, None, None], obs.unsqueeze(1).expand_as(self.obs_hist), shifted
            )
            goal = self.mdp.goal(info["f"], self.cfg.goal_frames)
            self.pi.obs_norm.update(obs)
            self.pi.goal_norm.update(goal)
            action = self.pi(self.obs_hist, goal)
            action = torch.clamp(action + self.cfg.action_noise * torch.randn_like(action), -20, 20)
            if self._bridged:
                step_state = self.env.step(action.to(self.env_device))
            else:
                step_state = self.env.step(action)
            transition = step_state.info[SUPERMIMIC_TRANSITION_KEY]
            if self._bridged:
                transition = {key: value.to(self.device) for key, value in transition.items()}
            if self.cfg.stage2_enabled:
                transition["bin"] = cast(Any, self.mdp).motion.frame_bin[transition["f"]]
                valid = self.mdp.valid_imagination_start(transition["f"], self.cfg.horizon)
                valid &= ~transition["term"]
                roles = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device)
                roles[self.acquisition_ids] = True
                a_count = int((valid & roles).sum())
                c_count = int((valid & ~roles).sum())
                if a_count + c_count:
                    self.stage2_ema, self.stage2_lambda = consolidation_weight(
                        a_count / (a_count + c_count), self.stage2_ema
                    )
            self.buffer.add(transition)
            initial = unflatten_state(transition["s"])
            next_state = unflatten_state(transition["s1"])
            self.wm.in_norm.update(
                self.wm.inputs(
                    initial,
                    transition["a"],
                    self.mdp.normalize_context(transition["xi"]),
                )
            )
            self.wm.out_norm.update(self.wm.targets(initial, next_state))
            self.sim_steps += self.env.num_envs
            self.stats["r"].append(transition["r"].mean() / self.mdp.max_step_reward)
            if "ep_len" in transition:
                self.stats["ep_len"].append(transition["ep_len"].mean())
                self.stats["ep_ret"].append(transition["ep_ret_norm"].mean())
                self.stats["term"].append(transition["term"].float().sum())
                self.stats["done"].append(transition["done"].float().sum())

    def wm_update(self) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
        predicted = self._wm_rollout_fn(window["s"], window["a"], window["xi"])
        loss, last_error = self._wm_err_fn(predicted, window["s1"], window["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        mask = window["mask"][:, -1]
        mean_last_error = (last_error.detach() * mask).sum() / mask.sum().clamp(min=1)
        return loss.detach() / self.cfg.wm_window, mean_last_error

    def _sample_valid_starts(
        self,
        batch_size: int,
        env_ids: torch.Tensor | None = None,
        star_count: int = 0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if batch_size < 1:
            raise ValueError("imagination batch must be positive")
        starts: list[dict[str, torch.Tensor]] = []
        histories: list[dict[str, torch.Tensor]] = []
        count = 0
        selected_count = 0
        attempts = 0
        while count < batch_size:
            attempts += 1
            if attempts > 100:
                raise RuntimeError("no valid imagination starts in the selected role")
            remaining = batch_size - count
            use_star = selected_count < star_count and self.star_pool is not None
            if use_star:
                pool = self.star_pool
                assert pool is not None
                absolute, envs, weights, saved_frames = pool
                ages = absolute - (self.buffer.total - self.buffer.size)
                live = (ages >= 0) & (ages < self.buffer.size)
                if bool(live.any()):
                    eligible = live.nonzero(as_tuple=False).flatten()
                    rows = self.buffer._row(ages[eligible])
                    live[eligible] = (
                        self.buffer.data["f"][rows, envs[eligible]] == saved_frames[eligible]
                    ) & ~self.buffer.data["done"][rows, envs[eligible]]
                if not bool(live.any()):
                    self.star_pool = None
                    continue
                picks = torch.multinomial(weights[live], 2 * remaining, replacement=True)
                chosen = (ages[live][picks], envs[live][picks])
                start, history = self.buffer.sample_starts(
                    2 * remaining, self.cfg.hist, chosen=chosen
                )
            else:
                start, history = self.buffer.sample_starts(
                    2 * remaining, self.cfg.hist, env_ids=env_ids
                )
            valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon)
            indices = valid.nonzero(as_tuple=False).squeeze(-1)[:remaining]
            if len(indices) == 0 and use_star:
                self.star_pool = None
                continue
            if len(indices) == 0:
                continue
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)
            if use_star:
                selected_count += len(indices)
        self._last_star_sample_count = selected_count

        def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        return concatenate(starts), concatenate(histories)

    @torch.no_grad()
    def _refresh_star(self) -> None:
        self.star_pool = None
        self.star_observed_completion = None
        self.regular_observed_completion = None
        if self.buffer.size < self.cfg.horizon or not self._star_critic_ready:
            return
        windows = self.buffer.star_windows(128, self.cfg.horizon, self.acquisition_ids)
        shape = windows["f"].shape
        frames = windows["f"].reshape(-1)
        context = self.mdp.normalize_context(windows["xi"].reshape(-1, windows["xi"].shape[-1]))
        next_frames = torch.minimum(
            frames + 1,
            cast(Any, self.mdp).motion.clip_end[cast(Any, self.mdp).motion.clip_id[frames]],
        )
        goals = self.mdp.goal(frames, self.cfg.goal_frames)
        next_goals = self.mdp.goal(next_frames, self.cfg.goal_frames)
        value = self.critic(
            self._critic_input(
                unflatten_state(windows["s"].reshape(-1, windows["s"].shape[-1])),
                frames,
                context,
                goals,
            )
        ).reshape(shape)
        next_value = self.critic(
            self._critic_input(
                unflatten_state(windows["s1"].reshape(-1, windows["s1"].shape[-1])),
                next_frames,
                context,
                next_goals,
            )
        ).reshape(shape)
        advantage = windows["r"] + self.cfg.gamma * (~windows["done"]) * next_value - value
        before_done = (
            torch.cat(
                (
                    torch.ones((shape[0], 1), device=self.device),
                    (~windows["done"][:, :-1]).float(),
                ),
                dim=1,
            )
            .cumprod(dim=1)
            .bool()
        )
        difficulty = cast(Any, self.env).stage2_difficulty.to(self.device)
        selected, priorities = selected_fragments(
            windows["bin"],
            advantage,
            windows["done"],
            difficulty,
            top_fraction=self.cfg.stage2_star_top,
            mask=before_done,
        )
        selected_mask = torch.zeros_like(windows["done"])
        if selected.numel():
            selected_mask[selected[:, 0], selected[:, 1]] = True
        completed = (
            windows["done"]
            & ~windows["term"]
            & (
                windows["f"] + 1
                >= cast(Any, self.mdp).motion.clip_end[
                    cast(Any, self.mdp).motion.clip_id[windows["f"]]
                ]
            )
        )
        # The window is truncated; these are diagnostic rates among observed
        # terminal transitions, not estimates of full-motion success rates.
        for pool, name in (
            (selected_mask, "star_observed_completion"),
            (before_done & ~selected_mask, "regular_observed_completion"),
        ):
            terminal = pool & windows["done"]
            if bool(terminal.any()):
                setattr(self, name, float(completed[terminal].float().mean()))
        if selected.numel():
            ages = windows["age"][selected[:, 0], selected[:, 1]]
            envs = windows["env"][selected[:, 0], selected[:, 1]]
            frames = windows["f"][selected[:, 0], selected[:, 1]]
            valid = self.mdp.valid_imagination_start(frames, self.cfg.horizon)
            if bool(valid.any()):
                absolute = ages[valid] + (self.buffer.total - self.buffer.size)
                self.star_pool = absolute, envs[valid], priorities[valid], frames[valid]

    def policy_update(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        star_count = int(self.cfg.pi_batch * self.cfg.stage2_star_fraction) if self.star_pool else 0
        self.star_fraction_actual = 0.0
        if self.cfg.stage2_enabled:
            regular, regular_history = self._sample_valid_starts(
                self.cfg.pi_batch - star_count, self.acquisition_ids
            )
            if star_count:
                selected, selected_history = self._sample_valid_starts(
                    star_count, self.acquisition_ids, star_count=star_count
                )
                start = {key: torch.cat((regular[key], selected[key])) for key in regular}
                history = {
                    key: torch.cat((regular_history[key], selected_history[key]))
                    for key in regular_history
                }
                self.star_fraction_actual = self._last_star_sample_count / self.cfg.pi_batch
            else:
                start, history = regular, regular_history
        else:
            start, history = self._sample_valid_starts(self.cfg.pi_batch)
        state = unflatten_state(start["s"])
        context, frames, previous_action = start["xi"], start["f"], start["a_prev"]
        normalized_context = self.mdp.normalize_context(context)
        with torch.no_grad():
            obs_hist = self.mdp.obs(
                unflatten_state(history["s"]),
                history["a_prev"],
                history["xi"],
                history["f"],
            )
        obs_hist = self.mdp.add_noise(obs_hist)
        _set_requires_grad(self.wm, False)
        _set_requires_grad(self.critic, False)

        batch_size = self.cfg.pi_batch
        initial_goal = self.mdp.goal(frames, self.cfg.goal_frames)
        goal = initial_goal
        initial_state = state
        states, actions, previous_actions, all_frames, goals = [], [], [], [], []
        try:
            for _ in range(self.cfg.horizon):
                previous_actions.append(previous_action)
                state, previous_action, obs_hist, frames, goal = self._imag_step_fn(
                    state,
                    obs_hist,
                    previous_action,
                    frames,
                    context,
                    normalized_context,
                    goal,
                )
                states.append(state)
                actions.append(previous_action)
                all_frames.append(frames)
                goals.append(goal)

            def flatten(parts: list[torch.Tensor]) -> torch.Tensor:
                return torch.stack(parts).flatten(0, 1)

            all_states = cast(
                State, tuple(flatten([item[index] for item in states]) for index in range(6))
            )
            flat_frames = flatten(all_frames)
            rewards, dones = self._reward_term_fn(
                all_states, flat_frames, flatten(actions), flatten(previous_actions)
            )
            rewards = rewards.view(self.cfg.horizon, batch_size)
            dones = dones.view(self.cfg.horizon, batch_size)
            alive = torch.cumprod(1 - dones, 0)
            continuation = torch.cat((torch.ones(1, batch_size, device=self.device), alive[:-1]), 0)
            discount = self.cfg.gamma ** torch.arange(
                self.cfg.horizon, device=self.device, dtype=torch.float32
            ).unsqueeze(-1)
            objective = (discount * continuation * rewards).sum(0)
            survival = alive[-1]
            reward_sum = (continuation * rewards).detach().sum(0)
            detached_rewards, detached_dones = rewards.detach(), dones.detach()
            if not self.cfg.no_critic:
                objective = objective + (self.cfg.gamma**self.cfg.horizon) * survival * self.critic(
                    self._critic_input_fn(state, frames, normalized_context, goal)
                )
            loss = -objective.mean()
            if self.ref_pi is not None:
                master, master_history = self._sample_valid_starts(
                    max(1, batch_size // 4), self.consolidation_ids
                )
                master_obs = self.mdp.obs(
                    unflatten_state(master_history["s"]),
                    master_history["a_prev"],
                    master_history["xi"],
                    master_history["f"],
                )
                master_goal = self.mdp.goal(master["f"], self.cfg.goal_frames)
                with torch.no_grad():
                    reference_action = self.ref_pi(master_obs, master_goal)
                imitation = (
                    (self.pi(master_obs, master_goal) - reference_action).square().sum(-1).mean()
                )
                self.consolidation_loss = float(imitation.detach())
                loss = loss + self.stage2_lambda * imitation
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            if torch.isfinite(grad_norm):
                self.opt_pi.step()
        finally:
            _set_requires_grad(self.wm, True)
            _set_requires_grad(self.critic, True)

        value_loss = torch.zeros((), device=self.device)
        if not self.cfg.no_critic:
            with torch.no_grad():
                critic_states = cast(
                    State,
                    tuple(
                        torch.cat(
                            (
                                initial_state[index].unsqueeze(0),
                                torch.stack([item[index].detach() for item in states]),
                            ),
                            0,
                        ).flatten(0, 1)
                        for index in range(6)
                    ),
                )
                critic_frames = torch.cat(
                    (start["f"].unsqueeze(0), torch.stack(all_frames)), 0
                ).flatten()
                critic_goals = torch.cat(
                    (initial_goal.unsqueeze(0), torch.stack(goals)), 0
                ).flatten(0, 1)
                critic_context = (
                    normalized_context.unsqueeze(0)
                    .expand(self.cfg.horizon + 1, -1, -1)
                    .flatten(0, 1)
                )
                critic_input = self._critic_input_fn(
                    critic_states, critic_frames, critic_context, critic_goals
                ).view(self.cfg.horizon + 1, batch_size, -1)
            self.critic.norm.update(critic_input)
            with torch.no_grad():
                values = self.critic(critic_input)
                target = values[-1]
                targets = []
                for index in range(self.cfg.horizon - 1, -1, -1):
                    target = detached_rewards[index] + self.cfg.gamma * (
                        1 - detached_dones[index]
                    ) * ((1 - self.cfg.lam) * values[index + 1] + self.cfg.lam * target)
                    targets.append(target)
                targets = torch.stack(targets[::-1])
            flat_input = critic_input[:-1].reshape(-1, critic_input.shape[-1])
            flat_targets = targets.reshape(-1)
            num_samples = flat_input.shape[0]
            minibatch_size = num_samples // self.cfg.critic_minibatches
            if minibatch_size == 0:
                raise ValueError("critic_minibatches exceeds the number of imagined transitions")
            for _ in range(self.cfg.critic_epochs):
                permutation = torch.randperm(num_samples, device=self.device)
                for index in range(self.cfg.critic_minibatches):
                    indices = permutation[index * minibatch_size : (index + 1) * minibatch_size]
                    value_loss = (
                        (self.critic(flat_input[indices]) - flat_targets[indices]) ** 2
                    ).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()

        imagined_reward = (reward_sum / self.cfg.horizon / self.mdp.max_step_reward).mean()
        return (
            loss.detach(),
            value_loss.detach(),
            imagined_reward,
            grad_norm.detach(),
            survival.mean().detach(),
        )

    def lr_at(self, iteration: int) -> float:
        if iteration >= self.cfg.lr_decay_iters:
            return self.cfg.lr * 0.03
        return self.cfg.lr * (
            0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / self.cfg.lr_decay_iters))
        )

    def _checkpoint(self, include_optimizers: bool) -> dict[str, Any]:
        checkpoint = {
            "wm": self.wm.state_dict(),
            "pi": self.pi.state_dict(),
            "iteration": self.iteration,
            "wall_time": self.wall_time,
            "sim_steps": self.sim_steps,
            "args": asdict(self.cfg),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if self.cfg.stage2_enabled:
            checkpoint["stage2_ema"] = self.stage2_ema
            checkpoint["stage2_base_checkpoint"] = self.cfg.stage2_base_checkpoint
            checkpoint["stage2_manifest_sha256"] = self.cfg.stage2_manifest_sha256
        if include_optimizers:
            checkpoint.update(
                {
                    "critic": self.critic.state_dict(),
                    "opt_wm": self.opt_wm.state_dict(),
                    "opt_pi": self.opt_pi.state_dict(),
                    "opt_v": self.opt_v.state_dict(),
                }
            )
        return checkpoint

    def save(self, path: str | Path | None = None, include_optimizers: bool = True) -> Path:
        destination = Path(path) if path is not None else self.log_dir / "last.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        torch.save(self._checkpoint(include_optimizers), temporary)
        os.replace(temporary, destination)
        return destination

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        required = {"critic", "opt_wm", "opt_pi", "opt_v"}
        missing = sorted(required - checkpoint.keys())
        if missing:
            raise ValueError(
                f"SuperMimic resume requires a full last.pt checkpoint; missing {missing} in {path}"
            )
        self.wm.load_state_dict(checkpoint["wm"])
        self.pi.load_state_dict(checkpoint["pi"])
        self.critic.load_state_dict(checkpoint["critic"])
        self._star_critic_ready = True
        self.opt_wm.load_state_dict(checkpoint["opt_wm"])
        self.opt_pi.load_state_dict(checkpoint["opt_pi"])
        self.opt_v.load_state_dict(checkpoint["opt_v"])
        self.iteration = checkpoint["iteration"]
        self.wall_time = checkpoint["wall_time"]
        self.sim_steps = checkpoint["sim_steps"]
        if self.cfg.stage2_enabled:
            if checkpoint.get("stage2_manifest_sha256") != self.cfg.stage2_manifest_sha256:
                raise ValueError("stage2 resume motion library identity mismatch")
            if checkpoint.get("stage2_base_checkpoint") != self.cfg.stage2_base_checkpoint:
                raise ValueError("stage2 resume base checkpoint mismatch")
            self.stage2_ema = float(checkpoint.get("stage2_ema", 0.6))
            self.stage2_lambda = min(1.0, 0.3 + 5.0 * max(0.0, self.stage2_ema - 0.6))
            self._load_reference(self.cfg.stage2_base_checkpoint)

    def _load_reference(self, path: str | Path, checkpoint: dict | None = None) -> None:
        loaded = (
            checkpoint
            if checkpoint is not None
            else torch.load(path, map_location=self.device, weights_only=True)
        )
        assert loaded is not None
        self.ref_pi = deepcopy(self.pi)
        self.ref_pi.load_state_dict(loaded["pi"])
        self.ref_pi.eval()
        _set_requires_grad(self.ref_pi, False)

    def initialize_from(self, path: str | Path) -> None:
        """Transfer weights and normalizers, but not run counters or optimizer moments."""
        if not self.cfg.stage2_enabled:
            raise ValueError("base checkpoint initialization is reserved for stage2")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.wm.load_state_dict(checkpoint["wm"])
        self.pi.load_state_dict(checkpoint["pi"])
        critic = matching_critic(Path(path), checkpoint)
        if critic is not None:
            self.critic.load_state_dict(critic)
            self._star_critic_ready = True
        else:
            print(
                "[stage2] no matching critic: start with a fresh critic; STAR waits for its first update"
            )
        self._load_reference(path, checkpoint)

    @staticmethod
    def _mean(values: list[torch.Tensor]) -> float:
        return float(sum(float(value) for value in values) / max(len(values), 1))

    def train(self, resume: str | Path | None = None) -> dict[str, float | int | bool]:
        if resume is not None:
            self.load(resume)
        start_iteration = self.iteration
        self.stats = _empty_stats()
        logger = SuperMimicLogger(
            env_name=type(self.env).__name__,
            backend=self.backend,
            max_iterations=self.cfg.iterations,
            wm_only_iters=self.cfg.wm_only_iters,
            start_iteration=start_iteration,
            num_envs=self.env.num_envs,
            log_dir=str(self.log_dir),
            log_backend=self.cfg.log_backend,
        )
        logger.start(status="Warming up...")
        try:
            if start_iteration == 0:
                self.collect(self.cfg.warmup_steps)
            required_steps = max(self.cfg.hist, self.cfg.wm_window)
            if start_iteration > 0:
                required_steps = max(required_steps, self.cfg.warmup_steps)
            if self.buffer.size < required_steps:
                self.collect(required_steps - self.buffer.size)
            logger.log_buffer_fill(self.buffer.size, self.cfg.buffer_steps)
            logger.start_training_timer()

            log_path = self.log_dir / "train_log.csv"
            with log_path.open("a", encoding="utf-8", newline="") as log_file:
                if start_iteration == 0:
                    log_file.write(",".join(_LOG_FIELDS) + "\n")
                train_start = time.time() - self.wall_time
                log_start = time.time()
                step_accumulator = 0.0
                aggregate: dict[str, list[torch.Tensor]] = {
                    key: [] for key in ("wm", "wm8", "pi", "v", "ir", "gn", "surv")
                }
                last_metrics: dict[str, float | int] = {}
                for iteration in range(start_iteration, self.cfg.iterations):
                    if self.cfg.compile and iteration == start_iteration:
                        logger.log_status("Compiling world model (one-off, slow on first run)")
                    if self.cfg.compile and iteration == self.cfg.wm_only_iters:
                        logger.log_status("Compiling policy update (one-off, slow on first run)")
                    iteration_start = time.time()
                    learning_rate = self.lr_at(iteration)
                    for optimizer in (self.opt_wm, self.opt_pi, self.opt_v):
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rate
                    step_accumulator += 1.0 / self.cfg.utd
                    collect_steps = int(step_accumulator)
                    step_accumulator -= collect_steps
                    stats_mark = len(self.stats["r"])
                    self.collect(collect_steps)
                    train_phase_start = time.time()
                    wm_loss = torch.zeros((), device=self.device)
                    wm_error = torch.zeros((), device=self.device)
                    for _ in range(self.cfg.wm_batches):
                        wm_loss, wm_error = self.wm_update()
                    if (
                        self.cfg.stage2_enabled
                        and iteration >= self.cfg.wm_only_iters
                        and iteration % self.cfg.stage2_star_refresh == 0
                    ):
                        self._refresh_star()
                    aggregate["wm"].append(wm_loss)
                    aggregate["wm8"].append(wm_error)
                    pi_loss: float | None = None
                    v_loss: float | None = None
                    imag_rew: float | None = None
                    pi_gradnorm: float | None = None
                    imag_survival: float | None = None
                    if iteration >= self.cfg.wm_only_iters:
                        policy_loss, value_loss, imagined_reward, grad_norm, survival = (
                            self.policy_update()
                        )
                        aggregate["pi"].append(policy_loss)
                        aggregate["v"].append(value_loss)
                        self._star_critic_ready = True
                        aggregate["ir"].append(imagined_reward)
                        aggregate["gn"].append(grad_norm)
                        aggregate["surv"].append(survival)
                        pi_loss = float(policy_loss)
                        v_loss = float(value_loss)
                        imag_rew = float(imagined_reward)
                        pi_gradnorm = float(grad_norm)
                        imag_survival = float(survival)
                    train_time = time.time() - train_phase_start
                    self.iteration = iteration + 1
                    self.wall_time = time.time() - train_start
                    iteration_time = time.time() - iteration_start

                    iter_rewards = self.stats["r"][stats_mark:]
                    mean_reward = self._mean(iter_rewards) if iter_rewards else None
                    ep_len = self._mean(self.stats["ep_len"]) if self.stats["ep_len"] else None
                    ep_ret_norm = self._mean(self.stats["ep_ret"]) if self.stats["ep_ret"] else None
                    total_done = sum(float(value) for value in self.stats["done"])
                    term_frac = (
                        sum(float(value) for value in self.stats["term"]) / total_done
                        if total_done > 0
                        else None
                    )
                    logger.log_step(
                        self.iteration,
                        wm_loss=float(wm_loss),
                        wm_err8=float(wm_error),
                        pi_loss=pi_loss,
                        v_loss=v_loss,
                        imag_rew=imag_rew,
                        imag_survival=imag_survival,
                        pi_gradnorm=pi_gradnorm,
                        lr=learning_rate,
                        sim_rew=mean_reward,
                        ep_len=ep_len,
                        ep_ret_norm=ep_ret_norm,
                        term_frac=term_frac,
                        iter_sim_steps=collect_steps * self.env.num_envs,
                        collect_time=train_phase_start - iteration_start,
                        learn_time=train_time,
                        iteration_time=iteration_time,
                    )

                    if self.iteration % self.cfg.log_every == 0:
                        if self.cfg.stage2_enabled:
                            with (self.log_dir / "stage2_metrics.jsonl").open(
                                "a", encoding="utf-8"
                            ) as metrics:
                                metrics.write(
                                    json.dumps(
                                        {
                                            "iteration": self.iteration,
                                            "valid_acquisition_ema": self.stage2_ema,
                                            "consolidation_weight": self.stage2_lambda,
                                            "consolidation_loss": self.consolidation_loss,
                                            "star_fraction": self.star_fraction_actual,
                                            "normal_fraction": 1.0 - self.star_fraction_actual,
                                            "star_observed_completion": self.star_observed_completion,
                                            "regular_observed_completion": self.regular_observed_completion,
                                            "star_pool_size": 0
                                            if self.star_pool is None
                                            else len(self.star_pool[0]),
                                        }
                                    )
                                    + "\n"
                                )
                        iterations_per_second = self.cfg.log_every / (time.time() - log_start)
                        log_start = time.time()
                        term_frac = float(
                            sum(float(value) for value in self.stats["term"])
                            / max(sum(float(value) for value in self.stats["done"]), 1)
                        )
                        row = [
                            self.iteration,
                            self.wall_time / 3600,
                            self.sim_steps,
                            self._mean(aggregate["wm"]),
                            self._mean(aggregate["wm8"]),
                            self._mean(aggregate["pi"]),
                            self._mean(aggregate["v"]),
                            self._mean(aggregate["ir"]),
                            self._mean(aggregate["gn"]),
                            self._mean(aggregate["surv"]),
                            self._mean(self.stats["r"]),
                            self._mean(self.stats["ep_len"]),
                            self._mean(self.stats["ep_ret"]),
                            term_frac,
                            iterations_per_second,
                            learning_rate,
                        ]
                        log_file.write(
                            ",".join(
                                f"{value:.6g}" if isinstance(value, float) else str(value)
                                for value in row
                            )
                            + "\n"
                        )
                        log_file.flush()
                        last_metrics = dict(zip(_LOG_FIELDS, row, strict=True))
                        self.stats = _empty_stats()
                        aggregate = {key: [] for key in aggregate}

                    if (
                        self.iteration % self.cfg.ckpt_every == 0
                        or self.iteration == self.cfg.iterations
                    ):
                        self.wall_time = time.time() - train_start
                        self.save()
                        numbered = self.save(
                            self.log_dir / f"ckpt_{self.iteration:06d}.pt",
                            include_optimizers=False,
                        )
                        logger.log_save(str(numbered))

            self.wall_time = time.time() - train_start
            logger.finish()
            return {
                "completed": self.iteration >= self.cfg.iterations,
                "iteration": self.iteration,
                "sim_steps": self.sim_steps,
                "wall_time": self.wall_time,
                **last_metrics,
            }
        finally:
            logger.close()


__all__ = ["SuperMimicConfig", "SuperMimicLearner"]
