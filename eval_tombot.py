"""Deterministic recovery and step-climbing benchmarks for a TombotRecovery checkpoint.

    python eval_tombot.py logs/<run>/checkpoints
    python eval_tombot.py logs/<run>/checkpoints --tilt_bands 60:85,85:110 --step_heights_cm 2,4,6

Recovery: tipped starts (no obstacle) in each tilt band; success = level (up_z > 0.95)
for the final second of a 4 s rollout.
Climbing: upright starts 6-12 cm in front of a step of each height; success = the
episode's obstacle-clear event fires within 12 s (hub past the step while upright).
The starts are fixed by seed, so numbers are comparable across checkpoints.
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks

sys.path.insert(0, str(Path(__file__).resolve().parent / "sandbox" / "Tom"))
import tombot_gnn  # noqa: E402
import twmr.tombot_env as tombot_env  # noqa: E402


def latest_checkpoint(path: Path) -> Path:
    path = path.resolve()
    steps = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(steps, key=lambda p: int(p.name)) if steps else path


def run(env, params, n, seed0, steps, metric_names):
    nets = tombot_gnn.make_ppo_networks(env.observation_size, env.action_size)
    policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    keys = jp.stack([jax.random.PRNGKey(seed0 + k) for k in range(n)])
    state = jax.jit(jax.vmap(env.reset))(keys)

    @jax.jit
    def rollout(state):
        def body(s, _):
            a, _ = jax.vmap(policy)(s.obs, keys)
            s = jax.vmap(env.step)(s, a)
            return s, {k: s.metrics[k] for k in metric_names}
        return jax.lax.scan(body, state, None, length=steps)[1]

    return state, {k: np.asarray(v) for k, v in rollout(state).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tilt_bands", default="40:60,60:85,85:110")
    ap.add_argument("--step_heights_cm", default="1,2,3,4,5,6")
    ap.add_argument("--n", type=int, default=32, help="starts per band / height")
    args = ap.parse_args()

    ckpt = latest_checkpoint(Path(args.checkpoint))
    print(f"checkpoint: {ckpt}")
    params = ppo_checkpoint.load(ckpt)
    dt = tombot_env.default_config().ctrl_dt

    for band in filter(None, args.tilt_bands.split(",")):
        lo, hi = map(float, band.split(":"))
        cfg = tombot_env.default_config()
        cfg.update(reset_upright_probability=0.0, obstacle_probability=0.0,
                   reset_tipped_roll_min=np.radians(lo), reset_tipped_roll_max=np.radians(hi))
        env = tombot_env.TransformableWheelMobileRobot(config=cfg)
        steps = int(round(4.0 / dt))
        state, m = run(env, params, args.n, 1000, steps, ["task/up_z"])
        ok = (m["task/up_z"][-int(round(1.0 / dt)):] > 0.95).all(0)
        nose_down = np.asarray(state.data.qpos[:, 4]) > 0
        print(f"recovery  tilt {lo:3.0f}-{hi:3.0f} deg: {ok.mean():4.0%}  "
              f"(nose-down {ok[nose_down].mean():.0%}, nose-up {ok[~nose_down].mean():.0%})")

    for h_cm in filter(None, args.step_heights_cm.split(",")):
        h = float(h_cm) / 100.0
        cfg = tombot_env.default_config()
        cfg.update(reset_upright_probability=1.0, obstacle_probability=1.0,
                   obstacle_height_min=h, obstacle_height_max=h + 1e-4,
                   obstacle_distance_min=0.06, obstacle_distance_max=0.12)
        env = tombot_env.TransformableWheelMobileRobot(config=cfg)
        steps = int(round(12.0 / dt))
        _, m = run(env, params, args.n, 2000, steps,
                   ["task/obstacle_clear_event", "task/up_z", "task/stall_s"])
        cleared = m["task/obstacle_clear_event"].sum(0) > 0.5
        fell = (m["task/up_z"] < 0.0).any(0)
        print(f"climbing  step {float(h_cm):4.1f} cm: cleared {cleared.mean():4.0%}  "
              f"(fell over {fell.mean():.0%}, max stall {m['task/stall_s'].max():.1f} s)")


if __name__ == "__main__":
    main()
