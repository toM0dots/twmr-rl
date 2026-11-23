"""TomBot recovery + traversal environment (MJX).

Task, in priority order:
  1. Recover to a flat, upright posture from a random (often tipped-over) start.
  2. Drive forward (world -y) steadily while level.
  3. Use the transform arms only when needed: when the robot is blocked by a step
     (or while tipped over), and keep them near neutral otherwise.

The actor is "blind": it only sees proprioception (IMU, motor-current proxies, wheel
and arm encoders), the previous action, and a short history of those signals. There is
no forward-looking sensor, so the policy has to infer a step from the history (impact,
pitch, wheel and current patterns). The critic is privileged and sees the obstacle.

Episodes always use a fresh random reset. Train with
`wrapper.wrap_for_brax_training(..., full_reset=True)`; with the default cached reset,
`info` (stall timer, obstacle flags, observation history) would leak between episodes.

Curriculum: set `obstacle_probability=0` for a recovery + flat-driving stage, then
continue training from that checkpoint with obstacles enabled (the default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import Array as JaxArray
from ml_collections import config_dict
from mujoco import MjModel, mjx  # type: ignore
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv, State, dm_control_suite
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import common


ConfigOverridesDict = dict[str, str | int | float | bool | list]
_XML_PATH = Path(__file__).parent.parent.parent / "assets" / "tombot.xml"

# ---------------------------------------------------------------------------
# Observation layout. Must match sandbox/Tom/tombot_gnn.py.
# ---------------------------------------------------------------------------
# One proprioceptive frame (24):
#   0:3    gravity direction in the body frame (from the IMU orientation)
#   3:6    body gyro / 10
#   6:9    body accelerometer / 9.81
#   9:15   signed motor-effort (current proxy) / torque limit
#          [rear-right, rear-left, front-right, front-left wheel, right arm, left arm]
#   15:20  wheel speeds / 100 rpm [rear-right, rear-left, front-right, front-left, middle]
#   20:22  arm angles / joint limit [right, left]
#   22:24  arm speeds / 10 rpm [right, left]
# Actor observation (102):
#   0:24    current frame
#   24:30   previous action
#   30:102  frames from HIST_OFFSETS control steps ago (3 x 24)
FRAME_SIZE = 24
HIST_OFFSETS = (5, 10, 20)  # 0.1 s, 0.2 s, 0.4 s at 50 Hz
HIST_LEN = max(HIST_OFFSETS)
POLICY_ACTION_SIZE = 6
ACTOR_OBS_SIZE = FRAME_SIZE + POLICY_ACTION_SIZE + FRAME_SIZE * len(HIST_OFFSETS)
# Privileged critic observation (39):
#   0:3    root position relative to start [x, y] and height
#   3:5    arm angles / joint limit
#   5:11   first two columns of the body rotation matrix
#   11:24  qvel (scaled)
#   24:30  previous action
#   30:39  task state: obstacle present, height, distance ahead, cleared, stall time,
#          blocked, level, heading alignment, lateral offset
PRIV_OBS_SIZE = 39

# Exact actuator order in the XML and therefore in the policy action.
ACTUATOR_NAMES = (
    "rear_right_drive",
    "rear_left_drive",
    "front_right_drive",
    "front_left_drive",
    "front_right_transform",
    "front_left_transform",
)

OBSTACLE_HALF_DEPTH = 0.025   # must match the step_obstacle geom size in the XML
OBSTACLE_HALF_HEIGHT = 0.05   # 10 cm box sunk into the floor: steps up to 10 cm
FRONT_REACH = 0.14            # hub origin to front of the front wheels, m


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,       # 50 Hz policy/control loop
        sim_dt=0.002,       # 10 physics steps per action
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="jax",
        naconmax=16384,
        njmax=700,

        # --- Reset: orientation -------------------------------------------
        # Upright starts get small noise; the rest are tipped about the wheel axle
        # (body x, nose-up or tail-up) with random arm angles.
        # The tipped range is limited to what the hardware can recover from. With the
        # +/-120 deg arm limit, pushing the arms to the limit from upside down only props
        # the body at ~118 deg, and open-loop arm/wheel patterns did not right any start
        # beyond ~110 deg (doubling the arm torque does not help; +/-180 deg arms reach
        # vertical). Below ~65 deg most starts fall back on their own; 65-110 deg is where
        # the arms are needed. Raise reset_tipped_roll_max only if the arm range grows.
        reset_upright_probability=0.35,
        reset_upright_tilt_noise=0.05,     # rad, roll and pitch
        reset_tipped_roll_min=40.0 * np.pi / 180.0,
        reset_tipped_roll_max=110.0 * np.pi / 180.0,
        reset_tipped_pitch_range=0.35,     # rad
        reset_yaw_range=0.10,              # rad around facing -y
        reset_xy_range=0.02,               # m
        reset_clearance=0.002,             # m between lowest robot geom and floor

        # --- Reset: arms ---------------------------------------------------
        reset_upright_arm_range=0.30,      # rad, coupled, plus mismatch below
        reset_upright_arm_mismatch=0.10,
        reset_tipped_arm_range=2.0,        # rad, independent per arm

        # --- Obstacle --------------------------------------------------------
        obstacle_probability=0.6,
        obstacle_height_min=0.010,         # m (1 cm already blocks the middle wheel)
        # Open-loop trajectory optimisation clears 4 cm steps but no 5-8 cm step (it
        # stalls ~70% of the way, with the arm-less rear end against the face), so the
        # default range stops at 4.5 cm. Train 1-3 cm first, then widen.
        obstacle_height_max=0.045,
        obstacle_distance_min=0.15,        # m from the front wheels to the step face
        obstacle_distance_max=0.45,
        # Fraction of upright obstacle episodes that start right at the step
        # (0.5-3 cm from the face), so the climb itself gets practised directly.
        obstacle_near_start_probability=0.3,
        obstacle_near_distance_max=0.03,
    )


def _quat_up_z(q: JaxArray) -> JaxArray:
    # MuJoCo quaternion ordering: [qw, qx, qy, qz]. Body z-axis . world z-axis.
    _, qx, qy, _ = q
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def _rpy_to_quaternion(roll: JaxArray, pitch: JaxArray, yaw: JaxArray) -> JaxArray:
    """Quaternion [w, x, y, z] for Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = jp.cos(0.5 * roll), jp.sin(0.5 * roll)
    cp, sp = jp.cos(0.5 * pitch), jp.sin(0.5 * pitch)
    cy, sy = jp.cos(0.5 * yaw), jp.sin(0.5 * yaw)
    q = jp.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ])
    return q / (jp.linalg.norm(q) + 1.0e-8)


class TransformableWheelMobileRobot(MjxEnv):
    """MJX environment for the six-actuator TomBot XML.

    Policy action in [-1, 1]^6, in XML actuator order:
      0 rear-right wheel, 1 rear-left wheel, 2 front-right wheel, 3 front-left wheel,
      4 front-right transform (arm) velocity, 5 front-left transform velocity.
    Each action is scaled to the XML command range [-200, 200].
    Forward is world -y (the robot's local -y at a zero-yaw reset).
    """

    # ------------------------------------------------------------------
    # Reward tuning panel (per control step, before PPO reward_scaling).
    # Positive terms are O(1) per step at most so returns stay ~O(100).
    # ------------------------------------------------------------------
    # "Level" gate: 0 below level_min_up_z, 1 above level_full_up_z.
    # Driving tilted earns nothing: locomotion rewards are multiplied by it.
    level_min_up_z = 0.85    # ~32 deg tilt
    level_full_up_z = 0.95   # ~18 deg tilt

    target_speed = 0.08      # m/s forward (wheel no-load limit is ~0.13 m/s)
    reward_speed = 1.0       # level * heading * clip(v_forward / target, -1, 1)
    reward_level = 0.2       # level
    reward_recovery = 3.0    # potential shaping on up_z: up_z(t+1) - up_z(t)
    reward_clear_obstacle = 5.0  # one-time, when the whole robot is past the step
    # Climbing progress: paid once per unit of new progress through the step zone
    # (from just before the front wheels touch the face to the clear line), whenever
    # the robot is not tipped over. It is NOT level-gated, so the pitched climb phase
    # is rewarded, and it tracks the best progress so far, so it cannot be farmed.
    # The episode total is at most reward_climb_progress.
    reward_climb_progress = 6.0
    climb_zone_lead = 0.02       # m: the zone starts this far before front-wheel contact
    climb_min_up_z = 0.5         # progress only counts while up_z is above this

    # Arm usage. "free" = level * (1 - blocked) * (1 - in_step_zone): penalties apply
    # while the robot is level and driving freely, and are lifted while it is tipped
    # over, blocked, or in the step zone (until the step is cleared). Tilting does NOT
    # switch them off for free, because tilting also removes the speed and level rewards.
    penalty_arm_pose = 0.3        # free * mean((arm_angle / 30 deg)^2), ratio capped at 2
    penalty_arm_speed = 0.02      # free * mean((arm_speed / 10 rpm)^2), ratio capped at 2
    arm_angle_scale_deg = 30.0
    arm_speed_scale = 1.04719755  # rad/s (10 rpm)
    arm_ratio_cap = 2.0

    # Stall / blocked detection (privileged; used only for the reward gate).
    stall_command_min = 0.3       # mean wheel action that counts as "trying to drive"
    stall_speed_fraction = 0.25   # forward speed below this fraction of target = stalled
    blocked_ramp_s = 0.4          # seconds of stall to fully lift the arm penalties

    # Smoothness and stability (all bounded).
    penalty_action = 0.005        # mean(a^2)
    penalty_action_rate = 0.05    # mean((a - a_prev)^2)
    penalty_lateral_vel = 0.1     # level * min((v_lateral / target)^2, 4)
    penalty_body_rate = 0.02      # level * min(|world angular velocity|^2, 25)
    penalty_heading = 0.1         # level * (1 - heading alignment)

    # Termination.
    lateral_limit = 0.6           # m from the start line
    penalty_failure = 5.0
    reward_clip = 5.0             # per-step reward is clipped to +/- this

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: ConfigOverridesDict | None = None,
    ):
        super().__init__(config, config_overrides)

        self._xml_path = _XML_PATH.as_posix()
        self._model_assets = common.get_assets()
        self._mj_model: MjModel = MjModel.from_xml_string(
            _XML_PATH.read_text(), self._model_assets
        )
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # type: ignore

        m = self._mj_model
        if int(m.nu) != POLICY_ACTION_SIZE:
            raise ValueError(f"Expected 6 actuators, got nu={m.nu}.")
        if int(m.nq) != 14 or int(m.nv) != 13:
            raise ValueError(f"Expected nq=14, nv=13; got nq={m.nq}, nv={m.nv}.")
        for i, name in enumerate(ACTUATOR_NAMES):
            if m.actuator(name).id != i:
                raise ValueError(f"Actuator {name!r} is not at index {i}.")

        self.policy_action_size = POLICY_ACTION_SIZE
        self.hub_body_id = m.body("central_hub").id
        self.obstacle_mocap_id = int(m.body("step_obstacle").mocapid[0])

        def sensor_adr(name: str) -> int:
            return int(m.sensor_adr[m.sensor(name).id])

        def qpos_adr(name: str) -> int:
            return int(m.jnt_qposadr[m.joint(name).id])

        def qvel_adr(name: str) -> int:
            return int(m.jnt_dofadr[m.joint(name).id])

        self.body_quat_adr = sensor_adr("body_quat")
        self.body_gyro_adr = sensor_adr("body_gyro")
        self.body_acc_adr = sensor_adr("body_acc")
        self.wheel_vel_adrs = np.array([sensor_adr(n) for n in (
            "rear_right_wheel_vel", "rear_left_wheel_vel",
            "front_right_wheel_vel", "front_left_wheel_vel",
            "middle_support_wheel_vel",
        )])
        self.arm_angle_adrs = np.array([sensor_adr("front_right_arm_angle"), sensor_adr("front_left_arm_angle")])
        self.arm_vel_adrs = np.array([sensor_adr("front_right_arm_vel"), sensor_adr("front_left_arm_vel")])

        self.arm_qpos_adrs = np.array([qpos_adr("front_right_arm_hinge"), qpos_adr("front_left_arm_hinge")])
        self.arm_qvel_adrs = np.array([qvel_adr("front_right_arm_hinge"), qvel_adr("front_left_arm_hinge")])
        self.wheel_qpos_adrs = np.array([qpos_adr(n) for n in (
            "middle_support_wheel_spin", "rear_right_wheel_spin", "rear_left_wheel_spin",
            "front_right_wheel_spin", "front_left_wheel_spin",
        )])
        self.actuated_dof_adrs = np.array([qvel_adr(n) for n in (
            "rear_right_wheel_spin", "rear_left_wheel_spin",
            "front_right_wheel_spin", "front_left_wheel_spin",
            "front_right_arm_hinge", "front_left_arm_hinge",
        )])

        self.ctrl_scale = jp.full((POLICY_ACTION_SIZE,), 200.0, dtype=jp.float32)
        # Joint-torque limits implied by the XML gears/force limits.
        self.motor_torque_limits = jp.array(
            [0.093163, 0.093163, 0.093163, 0.093163, 0.588399, 0.588399], dtype=jp.float32)
        self.arm_limit = 2.09439510
        self.wheel_speed_scale = 10.4719755  # 100 rpm
        self.arm_speed_scale_obs = 1.04719755  # 10 rpm

        # qvel scales for the critic, in dof order.
        qvel_scale = np.ones(m.nv, dtype=np.float32)
        qvel_scale[0:3] = 1.0
        qvel_scale[3:6] = 10.0
        for name in ("middle_support_wheel_spin", "rear_right_wheel_spin", "rear_left_wheel_spin",
                     "front_right_wheel_spin", "front_left_wheel_spin"):
            qvel_scale[qvel_adr(name)] = self.wheel_speed_scale
        for name in ("front_right_arm_hinge", "front_left_arm_hinge"):
            qvel_scale[qvel_adr(name)] = self.arm_speed_scale_obs
        self.qvel_scale = jp.asarray(qvel_scale)

        # Robot geoms that can touch the floor, for placing resets just above it.
        obstacle_body = m.body("step_obstacle").id
        ids, types, sizes = [], [], []
        for g in range(m.ngeom):
            body = m.geom_bodyid[g]
            if body in (0, obstacle_body):
                continue
            if not (m.geom_contype[g] & 2 or m.geom_conaffinity[g] & 2):
                continue  # does not collide with the floor
            if m.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                      mujoco.mjtGeom.mjGEOM_SPHERE):
                raise ValueError(f"Unsupported geom type for reset placement: {m.geom(g).name}")
            ids.append(g)
            types.append(int(m.geom_type[g]))
            sizes.append(m.geom_size[g].copy())
        self._floor_geom_ids = np.array(ids)
        self._floor_geom_is_box = jp.asarray(np.array(types) == mujoco.mjtGeom.mjGEOM_BOX)
        self._floor_geom_is_cyl = jp.asarray(np.array(types) == mujoco.mjtGeom.mjGEOM_CYLINDER)
        self._floor_geom_size = jp.asarray(np.array(sizes, dtype=np.float32))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _lowest_point_z(self, data: mjx.Data) -> JaxArray:
        """Exact world-z of the lowest point over floor-colliding robot geoms."""
        pos_z = data.geom_xpos[self._floor_geom_ids, 2]
        rz = data.geom_xmat[self._floor_geom_ids, 2, :]   # z-row: local axes' world-z components
        size = self._floor_geom_size
        box_extent = jp.sum(jp.abs(rz) * size, axis=-1)
        a = jp.abs(rz[:, 2])                                # cylinder axis is local z
        cyl_extent = size[:, 1] * a + size[:, 0] * jp.sqrt(jp.maximum(1.0 - a * a, 0.0))
        sphere_extent = size[:, 0]
        extent = jp.where(self._floor_geom_is_box, box_extent,
                          jp.where(self._floor_geom_is_cyl, cyl_extent, sphere_extent))
        return jp.min(pos_z - extent)

    def reset(self, rng: JaxArray) -> State:
        cfg = self._config
        keys = jax.random.split(rng, 16)
        u = lambda k, lo, hi, shape=(): jax.random.uniform(k, shape, minval=lo, maxval=hi)

        # Orientation.
        tipped = jax.random.uniform(keys[1]) >= cfg.reset_upright_probability
        side = jp.where(jax.random.bernoulli(keys[2]), 1.0, -1.0)
        n = cfg.reset_upright_tilt_noise
        roll = jp.where(tipped, side * u(keys[3], cfg.reset_tipped_roll_min, cfg.reset_tipped_roll_max),
                        u(keys[3], -n, n))
        pitch = jp.where(tipped, u(keys[4], -cfg.reset_tipped_pitch_range, cfg.reset_tipped_pitch_range),
                         u(keys[4], -n, n))
        yaw = u(keys[5], -cfg.reset_yaw_range, cfg.reset_yaw_range)
        quat = _rpy_to_quaternion(roll, pitch, yaw)

        # Arms.
        common_angle = u(keys[6], -cfg.reset_upright_arm_range, cfg.reset_upright_arm_range)
        mismatch = u(keys[7], -cfg.reset_upright_arm_mismatch, cfg.reset_upright_arm_mismatch)
        tipped_arms = u(keys[8], -cfg.reset_tipped_arm_range, cfg.reset_tipped_arm_range, (2,))
        arms = jp.where(tipped, tipped_arms, jp.stack([common_angle + mismatch, common_angle - mismatch]))
        arms = jp.clip(arms, -0.97 * self.arm_limit, 0.97 * self.arm_limit)

        start_xy = u(keys[9], -cfg.reset_xy_range, cfg.reset_xy_range, (2,))

        # Obstacle: a 10 cm box sunk into the floor so that its top is at `height`;
        # sunk completely when this episode has no obstacle.
        has_obstacle = jax.random.uniform(keys[10]) < cfg.obstacle_probability
        height = u(keys[11], cfg.obstacle_height_min, cfg.obstacle_height_max)
        distance = u(keys[12], cfg.obstacle_distance_min, cfg.obstacle_distance_max)
        near = (~tipped) & (jax.random.uniform(keys[13]) < cfg.obstacle_near_start_probability)
        distance = jp.where(near, u(keys[14], 0.005, cfg.obstacle_near_distance_max), distance)
        face_y = start_xy[1] - FRONT_REACH - distance
        obstacle_pos = jp.array([
            0.0,
            face_y - OBSTACLE_HALF_DEPTH,
            jp.where(has_obstacle, height - OBSTACLE_HALF_HEIGHT, -2.0 * OBSTACLE_HALF_HEIGHT - 0.01),
        ])

        qpos = jp.asarray(self.mjx_model.qpos0)
        qpos = qpos.at[0:2].set(start_xy)
        qpos = qpos.at[2].set(0.3)
        qpos = qpos.at[3:7].set(quat)
        qpos = qpos.at[self.arm_qpos_adrs].set(arms)
        qpos = qpos.at[self.wheel_qpos_adrs].set(0.0)

        data = mjx_env.make_data(
            self.mj_model, qpos=qpos, qvel=jp.zeros(self.mjx_model.nv),
            impl=self.mjx_model.impl.value, naconmax=cfg.naconmax, njmax=cfg.njmax,
        )
        data = data.replace(
            ctrl=jp.zeros(self.mjx_model.nu),
            mocap_pos=data.mocap_pos.at[self.obstacle_mocap_id].set(obstacle_pos),
        )
        # Place the robot so its lowest point is just above the floor (no drop, no penetration).
        data = mjx.kinematics(self.mjx_model, data)
        qpos = qpos.at[2].add(cfg.reset_clearance - self._lowest_point_z(data))
        data = data.replace(qpos=qpos)
        data = mjx.forward(self.mjx_model, data)

        frame = self._proprio_frame(data)
        info: dict[str, Any] = {
            "rng": keys[0],
            "start_xy": start_xy,
            "last_action": jp.zeros(POLICY_ACTION_SIZE),
            "frame_hist": jp.tile(frame[None], (HIST_LEN, 1)),
            "stall_s": jp.zeros(()),
            "cleared": jp.zeros(()),
            "best_progress": jp.zeros(()),
            "has_obstacle": has_obstacle.astype(jp.float32),
            "obstacle_height": jp.where(has_obstacle, height, 0.0),
            "obstacle_face_y": face_y,
            "initial_up_z": _quat_up_z(quat),
        }
        obs = self._get_obs(data, info, frame)
        metrics = {k: jp.zeros(()) for k in self._metric_names()}
        return mjx_env.State(data=data, obs=obs, reward=jp.zeros(()), done=jp.zeros(()),
                             metrics=metrics, info=info)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, state: State, action: JaxArray) -> State:
        info = dict(state.info)
        action = jp.clip(jp.reshape(action, (POLICY_ACTION_SIZE,)), -1.0, 1.0)
        data = mjx_env.step(self.mjx_model, state.data, action * self.ctrl_scale, self.n_substeps)

        terms, task = self._reward_terms(state.data, data, action, info)
        reward = sum(terms.values())

        lateral = data.qpos[0] - info["start_xy"][0]
        bad = ~(jp.all(jp.isfinite(data.qpos)) & jp.all(jp.isfinite(data.qvel)) & jp.isfinite(reward))
        failure = bad | (jp.abs(lateral) > self.lateral_limit)
        reward = jp.where(failure, -self.penalty_failure, jp.clip(reward, -self.reward_clip, self.reward_clip))
        reward = jp.where(bad, -self.penalty_failure, reward)

        info["stall_s"] = task["stall_s"]
        info["cleared"] = jp.maximum(info["cleared"], task["clear_event"])
        info["best_progress"] = task["best_progress"]
        info["last_action"] = action

        frame = self._proprio_frame(data)
        obs = self._get_obs(data, info, frame)
        info["frame_hist"] = jp.concatenate([frame[None], info["frame_hist"][:-1]], axis=0)
        obs = jax.tree.map(lambda x: jp.where(bad, 0.0, jp.nan_to_num(x)), obs)

        saturated = (jp.abs(action) > 0.99).astype(jp.float32)
        metrics = {f"reward/{k}" if not k.startswith("penalty") else k.replace("_", "/", 1): v
                   for k, v in terms.items()}
        metrics.update({
            "reward": reward,
            "reward/total": reward,
            "task/up_z": task["up_z"],
            "task/level": task["level"],
            "task/heading_alignment": task["align"],
            "task/forward_position": task["forward_position"],
            "task/forward_velocity": task["v_forward"],
            "task/lateral_position": lateral,
            "task/stall_s": task["stall_s"],
            "task/blocked": task["blocked"],
            "task/arm_free_gate": task["free"],
            "task/right_arm_angle": data.qpos[self.arm_qpos_adrs[0]],
            "task/left_arm_angle": data.qpos[self.arm_qpos_adrs[1]],
            "task/obstacle_present": info["has_obstacle"],
            # One-time event: its episode sum is 1 if the step was cleared.
            "task/obstacle_clear_event": task["clear_event"],
            "task/in_step_zone": task["in_zone"],
            "task/climb_progress": task["best_progress"],
            "task/failure": failure.astype(jp.float32),
            "diagnostic/wheel_saturation": jp.mean(saturated[:4]),
            "diagnostic/arm_saturation": jp.mean(saturated[4:]),
        })
        return mjx_env.State(data=data, obs=obs, reward=reward,
                             done=failure.astype(jp.float32), metrics=metrics, info=info)

    def _reward_terms(self, prev: mjx.Data, data: mjx.Data, action: JaxArray,
                      info: dict[str, Any]) -> tuple[dict[str, JaxArray], dict[str, JaxArray]]:
        up_prev = _quat_up_z(prev.qpos[3:7])
        up = _quat_up_z(data.qpos[3:7])
        level = jp.clip((up - self.level_min_up_z) / (self.level_full_up_z - self.level_min_up_z), 0.0, 1.0)

        rot = data.xmat[self.hub_body_id]
        heading = -rot[:2, 1]                       # world xy of the robot's forward (local -y)
        heading_norm = jp.linalg.norm(heading)
        align = jp.where(heading_norm > 0.1, jp.clip(-heading[1] / (heading_norm + 1e-8), 0.0, 1.0), 0.0)

        v_forward = -data.qvel[1]
        v_lateral = data.qvel[0]
        omega_world = rot @ data.qvel[3:6]           # free-joint angular velocity is body-frame

        # Stall timer and arm-penalty gate.
        trying = jp.mean(action[:4]) > self.stall_command_min
        stalled = trying & (v_forward < self.stall_speed_fraction * self.target_speed) & (level > 0.5)
        stall_s = jp.where(stalled, info["stall_s"] + self.dt, jp.maximum(info["stall_s"] - 2.0 * self.dt, 0.0))
        blocked = jp.clip(stall_s / self.blocked_ramp_s, 0.0, 1.0)

        # Step zone and climbing progress (privileged; reward only).
        has_obstacle = info["has_obstacle"] > 0.5
        far_face_y = info["obstacle_face_y"] - 2.0 * OBSTACLE_HALF_DEPTH
        clear_y = far_face_y - 0.17
        zone_start_y = info["obstacle_face_y"] + FRONT_REACH + self.climb_zone_lead
        progress = jp.clip((zone_start_y - data.qpos[1]) / (zone_start_y - clear_y), 0.0, 1.0)
        upright_enough = up > self.climb_min_up_z
        best_progress = jp.where(has_obstacle & upright_enough,
                                 jp.maximum(info["best_progress"], progress), info["best_progress"])
        progress_gain = best_progress - info["best_progress"]
        in_zone = (has_obstacle & (progress > 0.0) & (info["cleared"] < 0.5)).astype(jp.float32)

        # Arm penalties are off while tipped (level), blocked, or in the step zone.
        free = level * (1.0 - blocked) * (1.0 - in_zone)

        arm_q = data.qpos[self.arm_qpos_adrs]
        arm_qd = data.qvel[self.arm_qvel_adrs]
        angle_ratio = jp.clip(arm_q / jp.deg2rad(self.arm_angle_scale_deg), -self.arm_ratio_cap, self.arm_ratio_cap)
        speed_ratio = jp.clip(arm_qd / self.arm_speed_scale, -self.arm_ratio_cap, self.arm_ratio_cap)

        # One-time bonus once the rear wheels are past the step's far face.
        past = data.qpos[1] < clear_y
        clear_event = (past & has_obstacle & (up > 0.7)).astype(jp.float32) * (1.0 - info["cleared"])

        terms = {
            "speed": self.reward_speed * level * align * jp.clip(v_forward / self.target_speed, -1.0, 1.0),
            "level": self.reward_level * level,
            "recovery": self.reward_recovery * (up - up_prev),
            "clear_obstacle": self.reward_clear_obstacle * clear_event,
            "climb_progress": self.reward_climb_progress * progress_gain,
            "penalty_arm_pose": -self.penalty_arm_pose * free * jp.mean(angle_ratio ** 2),
            "penalty_arm_speed": -self.penalty_arm_speed * free * jp.mean(speed_ratio ** 2),
            "penalty_action": -self.penalty_action * jp.mean(action ** 2),
            "penalty_action_rate": -self.penalty_action_rate * jp.mean((action - info["last_action"]) ** 2),
            "penalty_lateral_vel": -self.penalty_lateral_vel * level * jp.minimum((v_lateral / self.target_speed) ** 2, 4.0),
            "penalty_body_rate": -self.penalty_body_rate * level * jp.minimum(jp.sum(omega_world ** 2), 25.0),
            "penalty_heading": -self.penalty_heading * level * (1.0 - align),
        }
        task = {
            "up_z": up, "level": level, "align": align, "v_forward": v_forward,
            "forward_position": -(data.qpos[1] - info["start_xy"][1]),
            "stall_s": stall_s, "blocked": blocked, "free": free, "clear_event": clear_event,
            "in_zone": in_zone, "best_progress": best_progress,
        }
        return terms, task

    @staticmethod
    def _metric_names() -> list[str]:
        names = ["reward/speed", "reward/level", "reward/recovery", "reward/clear_obstacle", "reward/climb_progress",
                 "penalty/arm_pose", "penalty/arm_speed", "penalty/action", "penalty/action_rate",
                 "penalty/lateral_vel", "penalty/body_rate", "penalty/heading", "reward", "reward/total"]
        names += ["task/up_z", "task/level", "task/heading_alignment", "task/forward_position",
                  "task/forward_velocity", "task/lateral_position", "task/stall_s", "task/blocked",
                  "task/arm_free_gate", "task/right_arm_angle", "task/left_arm_angle",
                  "task/obstacle_present", "task/obstacle_clear_event", "task/in_step_zone",
                  "task/climb_progress", "task/failure",
                  "diagnostic/wheel_saturation", "diagnostic/arm_saturation"]
        return names

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _proprio_frame(self, data: mjx.Data) -> JaxArray:
        s = data.sensordata
        q = s[self.body_quat_adr:self.body_quat_adr + 4]
        w, x, y, z = q / (jp.linalg.norm(q) + 1e-6)
        # Gravity direction (world -z) expressed in the body frame: -(third row of R).
        gravity_body = -jp.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
        gyro = jp.clip(s[self.body_gyro_adr:self.body_gyro_adr + 3] / 10.0, -5.0, 5.0)
        accel = jp.clip(s[self.body_acc_adr:self.body_acc_adr + 3] / 9.81, -5.0, 5.0)
        current = jp.clip(data.qfrc_actuator[self.actuated_dof_adrs] / self.motor_torque_limits, -3.0, 3.0)
        wheel_vel = jp.clip(s[self.wheel_vel_adrs] / self.wheel_speed_scale, -3.0, 3.0)
        arm_angle = jp.clip(s[self.arm_angle_adrs] / self.arm_limit, -1.0, 1.0)
        arm_vel = jp.clip(s[self.arm_vel_adrs] / self.arm_speed_scale_obs, -3.0, 3.0)
        return jp.concatenate([gravity_body, gyro, accel, current, wheel_vel, arm_angle, arm_vel])

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], frame: JaxArray) -> dict[str, JaxArray]:
        hist = info["frame_hist"]
        actor = jp.concatenate([frame, info["last_action"]] + [hist[k - 1] for k in HIST_OFFSETS])

        rot = data.xmat[self.hub_body_id]
        up = _quat_up_z(data.qpos[3:7])
        level = jp.clip((up - self.level_min_up_z) / (self.level_full_up_z - self.level_min_up_z), 0.0, 1.0)
        heading = -rot[:2, 1]
        heading_norm = jp.linalg.norm(heading)
        align = jp.where(heading_norm > 0.1, jp.clip(-heading[1] / (heading_norm + 1e-8), 0.0, 1.0), 0.0)
        rel_xy = data.qpos[0:2] - info["start_xy"]
        privileged = jp.concatenate([
            jp.clip(jp.array([rel_xy[0], rel_xy[1], data.qpos[2] / 0.1]), -5.0, 5.0),
            data.qpos[self.arm_qpos_adrs] / self.arm_limit,
            rot[:, 0], rot[:, 1],
            jp.clip(data.qvel / self.qvel_scale, -5.0, 5.0),
            info["last_action"],
            jp.array([
                info["has_obstacle"],
                info["obstacle_height"] / 0.05,
                jp.clip((data.qpos[1] - info["obstacle_face_y"]) / 0.5, -2.0, 2.0) * info["has_obstacle"],
                info["cleared"],
                jp.clip(info["stall_s"], 0.0, 2.0) / 2.0,
                jp.clip(info["stall_s"] / self.blocked_ramp_s, 0.0, 1.0),
                level,
                align,
                jp.clip(rel_xy[0] / 0.5, -2.0, 2.0),
            ]),
        ])
        return {"state": actor, "privileged_state": privileged}

    # ------------------------------------------------------------------
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.policy_action_size

    @property
    def mj_model(self) -> MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> MjxModel:
        return self._mjx_model


dm_control_suite.register_environment(
    env_name="TombotRecovery",
    env_class=TransformableWheelMobileRobot,
    cfg_class=default_config,
)
