"""Render a policy rollout with a translucent reference-motion ghost."""

from __future__ import annotations

import argparse
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat, unflatten_state
from unisupermimic.rl.contracts.supermimic import (
    SUPERMIMIC_STATE_KEY,
    SUPERMIMIC_TRANSITION_KEY,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SUPPORTED_BACKENDS
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import (
    load_policy,
    make_eval_env,
)
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP

if TYPE_CHECKING:
    from unisupermimic.tasks.motion_tracking.g1.supermimic.env import SuperMimicManagerBasedRlEnv

CONTROL_FPS = 50
DEFAULT_SCENE = Path("src/unisupermimic/assets/robots/g1/scene_supermimic.xml")


def flattened_qpos(state: torch.Tensor) -> torch.Tensor:
    num_joints = (state.shape[-1] - 13) // 2
    return torch.cat((state[..., :7], state[..., 13 : 13 + num_joints]), dim=-1)


def reference_qpos(motion: Any, frames: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (motion.root_p[frames], mat_to_quat(motion.root_R[frames]), motion.q[frames]),
        dim=-1,
    )


@torch.inference_mode()
def rollout_qpos(
    env: SuperMimicManagerBasedRlEnv,
    checkpoint: dict,
    clip_name: str,
    seconds: float,
    start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    mdp = cast(MDP, env.supermimic_mdp)
    motion = mdp.motion
    if clip_name not in motion.names:
        raise ValueError(f"Unknown motion clip: {clip_name}")
    clip_index = motion.names.index(clip_name)
    max_frames = max(1, round((start + seconds) * CONTROL_FPS))
    args = checkpoint["args"]
    hist = int(args["hist"])
    goal_frames = int(args["goal_frames"])
    policy = load_policy(env, checkpoint)

    _, reset_info = env.reset_to_frames(motion.clip_start.clone())
    current = reset_info[SUPERMIMIC_STATE_KEY]
    history = None
    states = [current["s"][clip_index].cpu()]
    frames = [int(current["f"][clip_index].item())]
    while len(states) < max_frames:
        obs = mdp.obs(
            unflatten_state(current["s"]),
            current["a_prev"],
            current["xi"],
            current["f"],
        )
        if history is None:
            history = obs.unsqueeze(1).repeat(1, hist, 1)
        else:
            history = torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)
        action = policy(history, mdp.goal(current["f"], goal_frames))
        state = env.step(action)
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        if bool(transition["done"][clip_index].item()):
            break
        current = state.info[SUPERMIMIC_STATE_KEY]
        states.append(current["s"][clip_index].cpu())
        frames.append(int(current["f"][clip_index].item()))

    start_frame = round(start * CONTROL_FPS)
    if start_frame >= len(states):
        raise ValueError("start exceeds the completed rollout")
    policy_qpos = flattened_qpos(torch.stack(states)).numpy()
    frame_ids = torch.tensor(frames, dtype=torch.long, device=motion.root_p.device)
    reference = reference_qpos(motion, frame_ids).cpu().numpy()
    return reference[start_frame:], policy_qpos[start_frame:]


def comparison_xml(scene: Path) -> str:
    xml = scene.read_text(encoding="utf-8")
    robot_xml = scene.with_name("supermimic.xml").read_text(encoding="utf-8")
    body = robot_xml[robot_xml.index('<body name="pelvis"') : robot_xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r'rgba="[^"]*"', 'rgba="0.35 0.65 1.0 0.25"', ghost)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="1" ')
    xml = xml.replace("</worldbody>", ghost + "</worldbody>")
    return re.sub(r"\s*<keyframe>.*?</keyframe>", "", xml, flags=re.DOTALL)


def load_comparison_model(scene: Path) -> Any:
    import mujoco

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=scene.parent,
            prefix=".render_",
            suffix=".xml",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(comparison_xml(scene))
        return mujoco.MjModel.from_xml_path(str(temporary))  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def label_font() -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        return ImageFont.load_default()


def make_comparison_frame(image: np.ndarray, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have HxWx3 shape")
    frame = Image.fromarray(image)
    draw = ImageDraw.Draw(frame)
    labels = (
        ((16, 12), label),
        ((16, image.shape[0] - 38), "solid: SuperMimic   blue ghost: Reference"),
    )
    for xy, text in labels:
        bounds = draw.textbbox(xy, text, font=label_font())
        draw.rectangle(
            (bounds[0] - 6, bounds[1] - 4, bounds[2] + 6, bounds[3] + 4),
            fill=(255, 255, 255),
        )
        draw.text(xy, text, fill=(20, 20, 20), font=label_font())
    return np.asarray(frame)


def render_comparison(
    reference: np.ndarray,
    policy: np.ndarray,
    output: Path,
    scene: Path = DEFAULT_SCENE,
    *,
    clip_name: str,
    width: int = 960,
    height: int = 540,
    fps: int = 25,
    global_ghost: bool = False,
) -> int:
    import imageio.v2 as imageio
    import mujoco

    mujoco = cast(Any, mujoco)
    if reference.shape != policy.shape or reference.ndim != 2:
        raise ValueError("reference and policy qpos must have matching frame x qpos shapes")
    if output.suffix.lower() != ".mp4":
        raise ValueError("output must be an .mp4 path")
    if fps <= 0 or CONTROL_FPS % fps != 0:
        raise ValueError(f"fps must be a positive divisor of {CONTROL_FPS}")
    model = load_comparison_model(scene)
    if reference.shape[1] * 2 != model.nq:
        raise ValueError(f"comparison model expects {model.nq // 2} qpos columns")
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 3.6
    camera.azimuth = 135
    camera.elevation = -12
    renderer = mujoco.Renderer(model, height=height, width=width)
    output.parent.mkdir(parents=True, exist_ok=True)
    stride = CONTROL_FPS // fps
    written = 0
    try:
        with imageio.get_writer(
            output,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        ) as writer:
            writer = cast(Any, writer)
            for index in range(0, len(reference), stride):
                data.qpos[:36] = policy[index]
                data.qpos[36:] = reference[index]
                if not global_ghost:
                    data.qpos[36:38] = policy[index, :2]
                mujoco.mj_kinematics(model, data)
                camera.lookat[:] = [policy[index, 0], policy[index, 1], 0.75]
                renderer.update_scene(data, camera=camera)
                label = f"{clip_name}  t={index / CONTROL_FPS:4.1f}s"
                writer.append_data(make_comparison_frame(renderer.render(), label))
                written += 1
    finally:
        renderer.close()
    return written


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=SUPPORTED_BACKENDS, default="mjwarp", help="Simulation backend"
    )
    parser.add_argument(
        "--motion",
        type=Path,
        action="append",
        help="Custom G1 reference-motion CSV (repeat to load multiple clips)",
    )
    parser.add_argument(
        "--clip",
        help="Clip basename to render; defaults to the custom CSV basename or dance1_subject1",
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/videos/supermimic_comparison.mp4")
    )
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--global-ghost", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    from unisupermimic.sim2sim import policy_load_dim_guard

    args = _argument_parser().parse_args(argv)
    if args.motion:
        clip_names = [path.stem for path in args.motion]
        if len(clip_names) != len(set(clip_names)):
            raise ValueError("Custom motion CSV files must have unique basenames")
        if args.clip is None:
            if len(clip_names) != 1:
                raise ValueError("Specify --clip when loading more than one --motion CSV")
            clip_name = clip_names[0]
        else:
            clip_name = args.clip
        env = make_eval_env(args.ckpt.parent, backend=args.backend, motion_files=args.motion)
    else:
        clip_name = args.clip or "dance1_subject1"
        env = make_eval_env(args.ckpt.parent, backend=args.backend)
    try:
        device = torch.device(env.device)
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
        with policy_load_dim_guard(
            env_obs_dim=env.supermimic_mdp.obs_dim,
            env_action_dim=env.action_manager.total_action_dim,
            algo_name="supermimic",
        ):
            reference, policy = rollout_qpos(
                env, checkpoint, clip_name, args.seconds, start=args.start
            )
        written = render_comparison(
            reference,
            policy,
            args.output,
            args.scene,
            clip_name=clip_name,
            width=args.width,
            height=args.height,
            fps=args.fps,
            global_ghost=args.global_ghost,
        )
        print(f"Wrote {written} frames to {args.output}")
    finally:
        env.close()
        cast(Any, env.backend).close()


__all__ = [
    "comparison_xml",
    "flattened_qpos",
    "load_comparison_model",
    "main",
    "make_comparison_frame",
    "reference_qpos",
    "render_comparison",
    "rollout_qpos",
]

if __name__ == "__main__":
    main()
