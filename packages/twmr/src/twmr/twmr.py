from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
from jax import Array as JaxArray
from ml_collections import config_dict
from mujoco import MjModel, mjx  # type: ignore
from mujoco.mjx import Model as MjxModel
from mujoco_playground import MjxEnv, State, dm_control_suite
from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward as reward_utils
from mujoco_playground._src.dm_control_suite import common

ConfigOverridesDict = dict[str, str | int | list]
_XML_PATH = Path(__file__).parent.parent.parent / "assets" / "tombot.xml" 


# def default_vision_config() -> config_dict.ConfigDict:
#     return config_dict.create(
#         gpu_id=0,
#         render_batch_size=512,
#         render_width=64,
#         render_height=64,
#         enable_geom_groups=[0, 1, 2],
#         use_rasterizer=False,
#         history=3,
#     )


# TODO: check all of these default values
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,  # 50 hz control
        sim_dt=0.01,
        episode_length=1000,
        action_repeat=1,  # TODO: should this be a ratio of ctrl_dt / sim_dt?
        vision=False,
        # vision_config=default_vision_config(),
        impl="warp",  # TODO: cartpole uses jax
        naconmax=100,  # allow collisions
        njmax=500,  # allow complex joints
    )


def reward_configs():
    cfg = config_dict.ConfigDict()

    # -------------------------
    # success / failure
    # -------------------------
    cfg.success_x = 2.0
    cfg.reward_progress = 8.0
    cfg.reward_forward_vel = 0.5
    cfg.reward_survival = 0.02
    cfg.reward_success = 50.0

    cfg.min_root_z = 0.10
    cfg.failure_min_up_z = 0.4
    cfg.ext_stall_steps = 80
    cfg.max_vel = 10
    
    cfg.stall_delta_threshold = 0.00005
    cfg.max_stall_steps = 200
    cfg.failure_back_x = -0.20
    cfg.failure_min_root_z = 0.005

    # -------------------------
    # obstacle region
    # -------------------------
    cfg.obstacle_x_min = 1.0
    cfg.obstacle_x_max = 1.4
    cfg.obstacle_margin = 0.001


    # -------------------------
    # penalties
    # -------------------------
    cfg.penalty_control = 0.005
    cfg.penalty_lateral = 0.25
    cfg.penalty_tilt = 0.15
    cfg.penalty_stall = 0.02
    cfg.penalty_failure = 20.0
    cfg.backward_penalty = 2.0

    # -------------------------
    # misc
    # -------------------------
    cfg.terminate_on_success = True

    return cfg

def _quat_up_z(q):
        # q = [qw, qx, qy, qz]
        qw, qx, qy, qz = q
        return 1.0 - 2.0 * (qx * qx + qy * qy)


class TransformableWheelMobileRobot(MjxEnv):
    def __init__(
        self,
        # Task specific config
        config: config_dict.ConfigDict = default_config(),
        config_overrides: ConfigOverridesDict | None = None,
    ):
        super().__init__(config, config_overrides)

        self._xml_path = _XML_PATH.as_posix()
        model_xml = _XML_PATH.read_text()
        self._model_assets = common.get_assets()
        self._mj_model: MjModel = MjModel.from_xml_string(model_xml, self._model_assets)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)  # type: ignore
        self._mj_model.opt.timestep = self.sim_dt
        self.central_hub_body_id = self._mj_model.body("central_hub").id
        
        # MuJoCo actuator dimension from XML.
        # For the mirrored-fold version this should be 6.
        self.nu = int(self._mj_model.nu)

        # Logical RL action dimension.
        # [lower_common_drive, lower_drive_diff, upper_wheel_drive,
        #  delta_upper_target, delta_fold_target]
        self.policy_action_size = 5

        # Control limits / action mapping.
        self.wheel_scale = 35.0
        self.diff_scale = 12.0

        self.upper_min = -1.50
        self.upper_max = 1.50
        self.fold_min = 0.0
        self.fold_max = 1.60

        self.max_upper_delta = 0.06
        self.max_fold_delta = 0.06
        
        def sensor_adr_dim(sensor_name: str) -> tuple[int, int]:
            sid = self._mj_model.sensor(sensor_name).id
            adr = int(self._mj_model.sensor_adr[sid])
            dim = int(self._mj_model.sensor_dim[sid])
            return adr, dim


        # IMU-like sensors
        self.base_quat_adr, self.base_quat_dim = sensor_adr_dim("base_orientation")
        self.base_gyro_adr, self.base_gyro_dim = sensor_adr_dim("base_angular_velocity")
        self.base_accel_adr, self.base_accel_dim = sensor_adr_dim("base_acceleration")

        # Actuator force / current-proxy sensors
        self.front_wheel_torque_adr, _ = sensor_adr_dim("front_lower_wheel_motor_torque")
        self.rear_wheel_torque_adr, _ = sensor_adr_dim("rear_lower_wheel_motor_torque")
        self.upper_wheel_torque_adr, _ = sensor_adr_dim("upper_wheel_motor_torque")
        self.upper_swing_torque_adr, _ = sensor_adr_dim("upper_swing_servo_torque")
        self.front_fold_torque_adr, _ = sensor_adr_dim("front_lower_fold_servo_torque")
        self.rear_fold_torque_adr, _ = sensor_adr_dim("rear_lower_fold_servo_torque")

        # Joint qpos addresses, looked up by name instead of hard-coded.
        def qpos_adr(joint_name: str) -> int:
            jid = self._mj_model.joint(joint_name).id
            return int(self._mj_model.jnt_qposadr[jid])

        def qvel_adr(joint_name: str) -> int:
            jid = self._mj_model.joint(joint_name).id
            return int(self._mj_model.jnt_dofadr[jid])

        self.front_fold_qpos = qpos_adr("front_lower_fold_hinge")
        self.rear_fold_qpos = qpos_adr("rear_lower_fold_hinge")
        self.upper_qpos = qpos_adr("upper_swing_hinge")

        self.front_wheel_qpos = qpos_adr("front_lower_wheel_hinge")
        self.rear_wheel_qpos = qpos_adr("rear_lower_wheel_hinge")
        self.upper_wheel_qpos = qpos_adr("upper_wheel_hinge")

        self.front_fold_qvel = qvel_adr("front_lower_fold_hinge")
        self.rear_fold_qvel = qvel_adr("rear_lower_fold_hinge")
        self.upper_qvel = qvel_adr("upper_swing_hinge")

        self.front_wheel_qvel = qvel_adr("front_lower_wheel_hinge")
        self.rear_wheel_qvel = qvel_adr("rear_lower_wheel_hinge")
        self.upper_wheel_qvel = qvel_adr("upper_wheel_hinge")
        
        self._reward_config = reward_configs()


        # TODO: figure out vision with the madrona batch renderer

        # TODO: what does this do for us exactly?
        # self._root_body_id = self._mj_model.body("root").id

    def reset(self, rng: JaxArray) -> State:
        rng, rng_fold, rng_upper = jax.random.split(rng, 3)

        qpos = self.mjx_model.qpos0
        qvel = jp.zeros(self.mjx_model.nv)

        # Curriculum stage A/B values.
        # Later expand these.
        ng, rng_mode, rng_fold, rng_upper, rng_extreme = jax.random.split(rng, 5)

        mode = jax.random.uniform(rng_mode, ())

        # Medium-radical: enough to force recovery, not completely chaotic.
        fold_medium = jax.random.uniform(rng_fold, (), minval=0.0, maxval=1.0)
        upper_medium = jax.random.uniform(rng_upper, (), minval=-0.6, maxval=1.0)

        # Fully radical.
        fold_radical = jax.random.uniform(rng_fold, (), minval=0.0, maxval=1.55)
        upper_radical = jax.random.uniform(rng_upper, (), minval=-1.4, maxval=1.55)

        # Boundary/extreme cases.
        fold_extreme = jax.random.uniform(rng_fold, (), minval=1.25, maxval=1.60)
        upper_extreme = jax.random.uniform(rng_upper, (), minval=-1.55, maxval=1.55)

        fold_init = jp.where(
            mode < 0.60,
            fold_medium,
            jp.where(mode < 0.90, fold_radical, fold_extreme),
        )

        upper_init = jp.where(
            mode < 0.60,
            upper_medium,
            jp.where(mode < 0.90, upper_radical, upper_extreme),
        )

        wheel_radius = 0.025
        leg_length = 0.150

        # If legs are folded downward, hub must start higher so wheels do not spawn underground.
        max_down_reach = jp.maximum(
            leg_length * jp.sin(fold_init),
            leg_length * jp.maximum(jp.sin(upper_init), 0.0),
        )

        hub_z = wheel_radius + max_down_reach + 0.005

        # Freejoint: x, y, z, qw, qx, qy, qz.
        qpos = qpos.at[0].set(0.0)
        qpos = qpos.at[1].set(0.0)
        qpos = qpos.at[2].set(hub_z)
        qpos = qpos.at[3].set(1.0)
        qpos = qpos.at[4].set(0.0)
        qpos = qpos.at[5].set(0.0)
        qpos = qpos.at[6].set(0.0)

        # Randomized starting morphology.
        qpos = qpos.at[self.front_fold_qpos].set(+fold_init)
        qpos = qpos.at[self.rear_fold_qpos].set(-fold_init)
        qpos = qpos.at[self.upper_qpos].set(upper_init)

        # Wheel angles can start at zero.
        qpos = qpos.at[self.front_wheel_qpos].set(0.0)
        qpos = qpos.at[self.rear_wheel_qpos].set(0.0)
        qpos = qpos.at[self.upper_wheel_qpos].set(0.0)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        # Match servo targets to initial pose so the robot does not snap at reset.
        init_ctrl = jp.array([
            0.0,
            0.0,
            0.0,
            upper_init,
            +fold_init,
            -fold_init,
        ], dtype=jp.float32)

        data = data.replace(ctrl=init_ctrl)
        data = mjx.forward(self.mjx_model, data)
        
        
        info = {"rng": rng}
        info["prev_root_x"] = data.qpos[0]
        info["success"] = jp.array(0.0, dtype=jp.float32)
        info["failure"] = jp.array(0.0, dtype=jp.float32)
        info["stall_steps"] = jp.array(0.0, dtype=jp.float32)

        info["upper_target"] = upper_init
        info["fold_target"] = fold_init

        info["last_action"] = jp.zeros((self.policy_action_size,), dtype=jp.float32)
        info["last_ctrl"] = init_ctrl
        #         metrics = {}
        
        z = jp.array(0.0, dtype=jp.float32)
        metrics = {
        "reward": z,
        "reward/progress_delta": z,
        "reward/survival": z,
        "reward/success_bonus": z,
        "reward/forward_vel_reward": z,

        "penalty/control": z,
        "penalty/action_rate": z,
        "penalty/lateral": z,
        "penalty/tilt": z,
        "penalty/stall": z,
        "penalty/failure": z,
        "penalty/yaw_rate_penalty": z,
        "penalty/backward_penalty": z,
        "penalty/fold": z,
        "penalty/upper_pose": z,

        "failure_back_x": z,

        "task/root_x": z,
        "task/root_vx": z,
        "task/delta_x": z,
        "task/upper_angle": z,
        "task/lower_fold": z,
        "task/success": z,
        "task/failure": z,
        "task/stall_steps": z,
        "task/failure_stall": z,

        "reward/total": z,
    }

        info = {"rng": rng}
        info["prev_root_x"] = data.qpos[0]
        info["success"] = jp.array(0.0, dtype=jp.float32)
        info["failure"] = jp.array(0.0, dtype=jp.float32)
        info["stall_steps"] = jp.array(0.0, dtype=jp.float32)

        # These two are required by _get_obs and _action_to_ctrl.
        info["upper_target"] = upper_init
        info["fold_target"] = fold_init

        # Policy action is 5D, MuJoCo ctrl is 6D.
        info["last_action"] = jp.zeros((self.policy_action_size,), dtype=jp.float32)
        info["last_ctrl"] = init_ctrl

        obs = self._get_obs(data, info)

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info=info,
        )
    
    def _action_to_ctrl(self, action: JaxArray, info: dict[str, Any]) -> tuple[JaxArray, dict[str, Any]]:
        """
        Logical policy action in [-1, 1]^5:

          action[0]: lower common wheel drive
          action[1]: lower front/rear drive difference
          action[2]: upper wheel drive
          action[3]: delta upper-arm target
          action[4]: delta lower-fold target

        MuJoCo ctrl in R^6:

          ctrl[0]: front lower wheel velocity
          ctrl[1]: rear lower wheel velocity
          ctrl[2]: upper wheel velocity
          ctrl[3]: upper arm target angle
          ctrl[4]: front lower fold target
          ctrl[5]: rear lower fold target
        """

        # Ensure action is flat 5D, not accidentally (1, 5) or (5, 1).
        action = jp.reshape(action, (-1,))

        lower_common = action[0] * self.wheel_scale
        lower_diff = action[1] * self.diff_scale

        front_wheel = lower_common - lower_diff
        rear_wheel = lower_common + lower_diff
        upper_wheel = action[2] * self.wheel_scale

        # Make targets scalar.
        old_upper_target = jp.squeeze(info["upper_target"])
        old_fold_target = jp.squeeze(info["fold_target"])

        upper_target = old_upper_target + action[3] * self.max_upper_delta
        fold_target = old_fold_target + action[4] * self.max_fold_delta

        upper_target = jp.clip(upper_target, self.upper_min, self.upper_max)
        fold_target = jp.clip(fold_target, self.fold_min, self.fold_max)

        ctrl = jp.stack([
            jp.squeeze(front_wheel),
            jp.squeeze(rear_wheel),
            jp.squeeze(upper_wheel),
            jp.squeeze(upper_target),
            jp.squeeze(+fold_target),
            jp.squeeze(-fold_target),
        ]).astype(jp.float32)

        new_info = dict(info)
        new_info["upper_target"] = upper_target
        new_info["fold_target"] = fold_target

        return ctrl, new_info

    def step(self, state: State, action: JaxArray) -> State:
        # Convert 5D policy action to 6D MuJoCo ctrl.
        ctrl, mapped_info = self._action_to_ctrl(action, state.info)

        data = mjx_env.step(
            self.mjx_model,
            state.data,
            ctrl,
            self.n_substeps,
        )

        root_x = data.qpos[0]

        reward, success, failure, metrics, failure_stall, stall_steps = (
            self._compute_reward(data, action, ctrl, mapped_info)
        )

        failure = jp.maximum(failure, failure_stall)
        done = jp.maximum(success, failure)

        info = dict(mapped_info)
        info["prev_root_x"] = root_x
        info["success"] = success
        info["failure"] = failure
        info["stall_steps"] = stall_steps
        info["last_action"] = action
        info["last_ctrl"] = ctrl

        obs = self._get_obs(data, info)

        metrics["task/stall_steps"] = stall_steps
        metrics["task/failure_stall"] = failure_stall
        metrics["task/failure"] = failure

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    
    def _get_actor_obs(self, data: mjx.Data, info: dict[str, Any]) -> JaxArray:
        """Hardware-style actor observation.

        No qpos.
        No qvel.
        No root linear velocity.

        Layout:
          orientation estimate: 4
          gyro angular velocity: 3
          accelerometer: 3
          motor current proxies: 5
          targets: 2
          last action: 5

        Total: 22
        """

        sensordata = data.sensordata

        # Orientation estimate.
        # In real hardware this would come from an AHRS/filter, not raw IMU alone.
        base_quat = data.xquat[self.central_hub_body_id]
        base_quat = base_quat / (jp.linalg.norm(base_quat) + 1e-6)

        # Gyroscope: angular velocity, real IMU has this.
        base_gyro = sensordata[
            self.base_gyro_adr : self.base_gyro_adr + self.base_gyro_dim
        ]

        # Accelerometer: acceleration/proper acceleration, real IMU has this.
        base_accel = sensordata[
            self.base_accel_adr : self.base_accel_adr + self.base_accel_dim
        ]

        gyro_obs = jp.clip(base_gyro / 10.0, -5.0, 5.0)
        accel_obs = jp.clip(base_accel / 9.81, -5.0, 5.0)

        # Motor torque/current proxies.
        front_wheel_torque = sensordata[self.front_wheel_torque_adr]
        rear_wheel_torque = sensordata[self.rear_wheel_torque_adr]
        upper_wheel_torque = sensordata[self.upper_wheel_torque_adr]
        upper_swing_torque = sensordata[self.upper_swing_torque_adr]
        front_fold_torque = sensordata[self.front_fold_torque_adr]
        rear_fold_torque = sensordata[self.rear_fold_torque_adr]

        wheel_torque_limit = 0.078
        shape_torque_limit = 0.98

        front_wheel_current = jp.abs(front_wheel_torque) / wheel_torque_limit
        rear_wheel_current = jp.abs(rear_wheel_torque) / wheel_torque_limit
        upper_wheel_current = jp.abs(upper_wheel_torque) / wheel_torque_limit

        upper_swing_current = jp.abs(upper_swing_torque) / shape_torque_limit

        lower_fold_current = 0.5 * (
            jp.abs(front_fold_torque) + jp.abs(rear_fold_torque)
        ) / shape_torque_limit

        current_obs = jp.array([
            front_wheel_current,
            rear_wheel_current,
            upper_wheel_current,
            upper_swing_current,
            lower_fold_current,
        ], dtype=jp.float32)

        current_obs = jp.clip(current_obs, 0.0, 3.0)

        target_obs = jp.array([
            info["upper_target"] / self.upper_max,
            info["fold_target"] / self.fold_max,
        ], dtype=jp.float32)

        return jp.concatenate([
            base_quat,             # 4
            gyro_obs,              # 3 angular velocity, allowed
            accel_obs,             # 3 acceleration, allowed
            current_obs,           # 5 current proxies
            target_obs,            # 2
            info["last_action"],   # 5
        ])
    
    def _get_privileged_obs(self, data: mjx.Data, info: dict[str, Any]) -> JaxArray:
        """Privileged critic observation.

        Layout:
          qpos:          13
          qvel:          12
          targets:       2
          last_action:   5

        Total: 32
        """

        target_obs = jp.array([
            info["upper_target"],
            info["fold_target"],
        ], dtype=jp.float32)

        return jp.concatenate([
            data.qpos,             # 13
            data.qvel,             # 12
            target_obs,            # 2
            info["last_action"],   # 5
        ])


    def _get_obs(self, data: mjx.Data, info: dict[str, Any]):
        return {
            "state": self._get_actor_obs(data, info),
            "privileged_state": self._get_privileged_obs(data, info),
        }
    
    
#     def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> JaxArray:
#         qpos = data.qpos
#         qvel = data.qvel

#         target_obs = jp.array([
#             info["upper_target"],
#             info["fold_target"],
#         ], dtype=jp.float32)

#         return jp.concatenate([
#             qpos,
#             qvel,
#             target_obs,
#             info["last_action"],
#         ])
    
    
    def _compute_task_flags(self, data, info):
        """Flat-ground forward/recovery task flags."""

        root_x = data.qpos[0]
        root_z = data.qpos[2]
        root_quat = data.qpos[3:7]
        up_z = _quat_up_z(root_quat)

        has_nan = (
            (~jp.isfinite(data.qpos)).any()
            | (~jp.isfinite(data.qvel)).any()
            | (~jp.isfinite(data.ctrl)).any()
        ).astype(jp.float32)

        # Success: move forward far enough on flat ground.
        success = (root_x >= self._reward_config.success_x).astype(jp.float32)

        # Stall detection.
        delta_x = root_x - info["prev_root_x"]
        is_stalled = (
            jp.abs(delta_x) < self._reward_config.stall_delta_threshold
        ).astype(jp.float32)

        stall_steps = jp.where(
            is_stalled > 0.5,
            info["stall_steps"] + 1.0,
            0.0,
        )

        failure_stall = (
            stall_steps >= self._reward_config.max_stall_steps
        ).astype(jp.float32)

        # Failure conditions.
        failure_height = (
            root_z < self._reward_config.failure_min_root_z
        ).astype(jp.float32)

        failure_back_x = (
            root_x < self._reward_config.failure_back_x
        ).astype(jp.float32)

        # Optional: keep this weak/off for now, because random leg poses may tilt.
        failure_tilt = jp.array(0.0, dtype=jp.float32)

        failure = jp.maximum(
            has_nan,
            jp.maximum(
                failure_stall,
                jp.maximum(
                    failure_height,
                    jp.maximum(failure_back_x, failure_tilt),
                ),
            ),
        )

        return (
            success,
            failure,
            root_x,
            root_z,
            up_z,
            stall_steps,
            failure_stall,
            failure_back_x,
        )
    
    def _compute_reward(self, data, action, ctrl, info):
        success, failure, root_x, root_z, up_z, stall_steps, failure_stall, failure_back_x = (
            self._compute_task_flags(data, info)
        )

        root_vx = data.qvel[0]
        root_vy = data.qvel[1]
        root_wz = data.qvel[5]
        delta_x = root_x - info["prev_root_x"]

        upper_angle = data.qpos[self.upper_qpos]
        front_fold = data.qpos[self.front_fold_qpos]
        rear_fold = data.qpos[self.rear_fold_qpos]

        # Symmetric fold magnitude.
        lower_fold = 0.5 * (jp.abs(front_fold) + jp.abs(rear_fold))

        tilt_err = 1.0 - jp.clip(up_z, -1.0, 1.0)

        progress_delta_reward = self._reward_config.reward_progress * delta_x
        forward_vel_reward = self._reward_config.reward_forward_vel * jp.clip(
            root_vx, 0.0, self._reward_config.max_vel
        )
        survival_reward = self._reward_config.reward_survival * (1.0 - failure)

        control_penalty = self._reward_config.penalty_control * jp.mean(jp.square(action))
        action_rate_penalty = 0.04 * jp.mean(jp.square(action - info["last_action"]))

        lateral_penalty = self._reward_config.penalty_lateral * jp.square(root_vy)
        tilt_penalty = self._reward_config.penalty_tilt * jp.square(tilt_err)
        yaw_rate_penalty = 0.2 * jp.square(root_wz)
        backward_penalty = self._reward_config.backward_penalty * jp.maximum(-root_vx, 0.0)

        # Weak morphology regularization:
        # For flat recovery, prefer not to stay highly folded forever,
        # but do not make this too strong because reset is randomized.
        fold_penalty = 0.03 * jp.square(lower_fold)

        # Penalize upper arm being deployed downward too much on flat ground.
        # If your upper-positive direction is opposite, change max(upper_angle, 0) accordingly.
        upper_pose_penalty = 0.02 * jp.square(jp.maximum(upper_angle, 0.0))

        is_stalled = (jp.abs(delta_x) < self._reward_config.stall_delta_threshold).astype(jp.float32)

        stall_steps = jp.where(
            is_stalled > 0.5,
            info["stall_steps"] + 1.0,
            0.0,
        )

        failure_stall = (stall_steps >= self._reward_config.max_stall_steps).astype(jp.float32)
        stall_penalty = self._reward_config.penalty_stall * is_stalled

        success_bonus = self._reward_config.reward_success * success
        failure_penalty = self._reward_config.penalty_failure * failure

        reward = (
            progress_delta_reward
            + forward_vel_reward
            + survival_reward
            + success_bonus
            - control_penalty
            - action_rate_penalty
            - lateral_penalty
            - tilt_penalty
            - yaw_rate_penalty
            - backward_penalty
            - fold_penalty
            - upper_pose_penalty
            - stall_penalty
            - failure_penalty
        )

        metrics = {
            "reward": reward,
            "reward/progress_delta": progress_delta_reward,
            "reward/survival": survival_reward,
            "reward/success_bonus": success_bonus,
            "reward/forward_vel_reward": forward_vel_reward,
            "penalty/control": control_penalty,
            "penalty/action_rate": action_rate_penalty,
            "penalty/lateral": lateral_penalty,
            "penalty/tilt": tilt_penalty,
            "penalty/yaw_rate_penalty": yaw_rate_penalty,
            "penalty/backward_penalty": backward_penalty,
            "penalty/fold": fold_penalty,
            "penalty/upper_pose": upper_pose_penalty,
            "penalty/stall": stall_penalty,
            "penalty/failure": failure_penalty,
            "failure_back_x": failure_back_x,
            "task/root_x": root_x,
            "task/root_vx": root_vx,
            "task/delta_x": delta_x,
            "task/upper_angle": upper_angle,
            "task/lower_fold": lower_fold,
            "task/success": success,
            "task/failure": failure,
            "reward/total": reward,
            "task/failure_stall": failure_stall,
            "task/stall_steps": stall_steps,
        }

        return reward, success, failure, metrics, failure_stall, stall_steps
    

#     def _compute_task_flags(self, data, info):
#         root_x = data.qpos[0]
#         root_z = data.qpos[2]
#         root_quat = data.qpos[3:7]
#         up_z = _quat_up_z(root_quat)

#         has_nan = (
#             (~jp.isfinite(data.qpos)).any()
#             | (~jp.isfinite(data.qvel)).any()
#         ).astype(jp.float32)

#         success = (root_x >= self._reward_config.success_x).astype(jp.float32)
            
#         """"
#             & (up_z >= self._reward_config.success_min_up_z 
#             & (root_z >= self._reward_config.min_root_z)
#            )
#         .astype(jp.float32)
#          """
#         delta_x = root_x - info["prev_root_x"]

#         is_stalled = (jp.abs(delta_x) < self._reward_config.stall_delta_threshold).astype(jp.float32)
#         stall_steps = jp.where(
#             is_stalled > 0.5,
#             info["stall_steps"] + 1.0,
#             0.0,
#         )
#         failure_stall = (stall_steps >= self._reward_config.max_stall_steps).astype(jp.float32)
            

#         failure_height = (root_z < self._reward_config.failure_min_root_z).astype(jp.float32)
        
#         failure_back_x = (root_x < self._reward_config.failure_back_x).astype(jp.float32)
#         failure_tilt = 0 # (up_z < self._reward_config.failure_min_up_z).astype(jp.float32)

#         failure = jp.maximum( failure_back_x, jp.maximum(has_nan, jp.maximum(failure_stall, jp.maximum(failure_height, failure_tilt))))
#         return success, failure, root_x, root_z, up_z, stall_steps, failure_stall, failure_back_x


#     def _compute_reward(self, data, action, info):
#         success, failure, root_x, root_z, up_z, stall_steps, failure_stall, failure_back_x  = self._compute_task_flags(data, info)

#         root_vx = data.qvel[0]
#         root_vy = data.qvel[1]
#         root_wz = data.qvel[5]
#         delta_x = root_x - info["prev_root_x"]

#         # module extension state
#         mod_qpos = jp.reshape(data.qpos[7:], (4, 4))
#         ext_pos = mod_qpos[:, 1:4]
#         ext_mean_per_mod = jp.mean(ext_pos, axis=-1)
#         ext_mean_all = jp.mean(ext_mean_per_mod)

#         # actions:
#         # [FL wheel, FR wheel, RL wheel, RR wheel, FL ext0, FR ext0, RL ext0, RR ext0]
#         ext_action = action[4:8]
#         extend_use = jp.mean(jp.maximum(ext_action, 0.0))  

#         # obstacle-local gate
#         in_obstacle_local = (
#             (root_x >= self._reward_config.obstacle_x_min - self._reward_config.obstacle_margin)
#             & (root_x <= self._reward_config.obstacle_x_max + self._reward_config.obstacle_margin)
#         ).astype(jp.float32)

#         tilt_err = 1.0 - jp.clip(up_z, -1.0, 1.0)

#         progress_delta_reward = self._reward_config.reward_progress * delta_x 
#         forward_vel_reward = self._reward_config.reward_forward_vel * jp.clip(root_vx, 0, self._reward_config.max_vel)
#         survival_reward = self._reward_config.reward_survival * (1.0 - failure)

#         control_penalty = self._reward_config.penalty_control * jp.mean(jp.square(action))
#         lateral_penalty = self._reward_config.penalty_lateral * jp.square(root_vy)
#         tilt_penalty = self._reward_config.penalty_tilt * jp.square(tilt_err)
#         yaw_rate_penalty = 0.5 * jp.square(root_wz)
#         backward_penalty = self._reward_config.backward_penalty * jp.maximum(-root_vx, 0.0)
        
#         ext_norm = jp.clip((ext_mean_all ) / self.leg_max, 0.0, 1.0)
#         ext_reward = 1 * ext_norm * (info["stall_steps"] > self._reward_config.ext_stall_steps) * in_obstacle_local
#         ext_penalty = self._reward_config.penalty_ext * ext_norm
        
#         is_stalled = (jp.abs(delta_x) < self._reward_config.stall_delta_threshold).astype(jp.float32)
        
#         stall_excess = jp.maximum(stall_steps - self._reward_config.ext_stall_steps, 0.0)
#         stall_excess_capped = jp.minimum(stall_excess, 20.0)
#         stall_penalty = ( self._reward_config.penalty_stall * is_stalled + self._reward_config.penalty_stall * stall_excess_capped )
        
#         recovery_reward = 5 * jp.maximum(delta_x, 0.0) * ext_norm * (info["stall_steps"] > self._reward_config.ext_stall_steps )
        
#         success_bonus = self._reward_config.reward_success * success
#         success_in_region = 20 * in_obstacle_local
#         failure_penalty = self._reward_config.penalty_failure * failure

#         reward = (
#             progress_delta_reward
#             + survival_reward
#             + forward_vel_reward
#             + ext_reward
#             + recovery_reward
#             + success_bonus
#             + success_in_region
#             - control_penalty
#             - lateral_penalty
#             - tilt_penalty
#             - stall_penalty
#             - ext_penalty
#             - failure_penalty
#             - yaw_rate_penalty
#             - backward_penalty
#         )

#         metrics = {
#             "reward": reward,
#             "reward/progress_delta": progress_delta_reward,
#             "reward/survival": survival_reward,
#             "reward/success_bonus": success_bonus,
#             "reward/forward_vel_reward": forward_vel_reward, 
#             "penalty/control": control_penalty,
#             "penalty/lateral": lateral_penalty,
#             "penalty/tilt": tilt_err,
#             "penalty/stall": stall_penalty,
#             "penalty/failure": failure_penalty,
#             "penalty/yaw_rate_penalty": yaw_rate_penalty,
#             "penalty/backward_penalty": backward_penalty, 
#             "failure_back_x": failure_back_x, 
#             "task/root_x": root_x,
#             "task/root_vx": root_vx,
#             "task/delta_x": delta_x,
#             "task/ext_mean": ext_mean_all,
#             "task/in_obstacle_local": in_obstacle_local,
#             "task/success": success,
#             "task/failure": failure,
#             "reward/total": reward,
#             "task/stall_steps": stall_steps,       
#         }

#         return reward, success, failure, metrics, failure_stall, stall_steps

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
    env_name="TransformableWheelMobileRobot",
    env_class=TransformableWheelMobileRobot,
    cfg_class=default_config,
)
