from __future__ import annotations

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
from mujoco_playground._src.dm_control_suite import common


ConfigOverridesDict = dict[str, str | int | float | bool | list]
_XML_PATH = (
    Path(__file__).parent.parent.parent
    / "assets"
    / "tombot.xml"
)

# These must match tombot_gnn.py.
ACTOR_OBS_SIZE = 31
PRIV_OBS_SIZE = 33
POLICY_ACTION_SIZE = 6


# Exact actuator order in the XML and therefore in the policy action.
ACTUATOR_NAMES = (
    "rear_right_drive",
    "rear_left_drive",
    "front_right_drive",
    "front_left_drive",
    "front_right_transform",
    "front_left_transform",
)


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,       # 50 Hz policy/control loop
        sim_dt=0.002,       # matches the XML; 10 physics steps per action
        episode_length=1000,
        action_repeat=1,
        vision=False,
        impl="warp",
        naconmax=150,
        njmax=700,
        
        # ----------------------------------------------------
        # Reset curriculum
        # ----------------------------------------------------
        reset_arm_common_range=0.45,
        reset_arm_mismatch_range=0.15,

        reset_roll_range=0.1,
        reset_pitch_range=0.1,

        # Robot's local -y points along world +x.
        reset_yaw= 0.0, 

        reset_xy_range=0.0,
        reset_linear_vel_range=0.0,
        reset_angular_vel_range=0.0,

        reset_height_margin=0.0005,
        reset_tilt_height_gain=0.08,
        )


def reward_configs() -> config_dict.ConfigDict:
    cfg = config_dict.ConfigDict()

    # Forward is -world-x in the supplied XML.
    cfg.success_forward = 0.75
    cfg.failure_back_forward = -0.20
    cfg.failure_min_root_z = 0.005
    cfg.failure_min_up_z = 0.05
    cfg.failure_back_forward = -0.40

    cfg.stall_delta_threshold = 2.0e-4
    cfg.max_stall_steps = 250
    cfg.max_forward_vel = 3.0
    cfg.stall_grace_steps = 100

    cfg.reward_progress = 25.0
    cfg.reward_forward_vel = 0.2
    cfg.reward_survival = 0.002
    cfg.reward_success = 50.0

    cfg.penalty_control = 0.002
    cfg.penalty_action_rate = 0.01
    cfg.penalty_lateral_vel = 0.20
    cfg.penalty_lateral_position = 0.05
    cfg.penalty_tilt = 0.150
    cfg.penalty_yaw_rate = 0.05
    cfg.penalty_backward = 3.00
    cfg.penalty_stall = 0.00000002
    cfg.penalty_failure = 20.0

    # The arm motors remain independent, but this weakly encourages the two
    # front transform assemblies to stay synchronized.
    cfg.penalty_arm_sync = 0.05
    cfg.penalty_arm_limit = 0.03

    cfg.terminate_on_success = True
    
    
    return cfg


def _quat_up_z(q: JaxArray) -> JaxArray:
    # MuJoCo quaternion ordering: [qw, qx, qy, qz].
    _, qx, qy, _ = q
    return 1.0 - 2.0 * (qx * qx + qy * qy)


class TransformableWheelMobileRobot(MjxEnv):
    """MJX environment matched to the six-actuator TomBot XML.

    Policy action in [-1, 1]^6 and XML control order:
      0 rear-right wheel
      1 rear-left wheel
      2 front-right wheel
      3 front-left wheel
      4 front-right transform velocity
      5 front-left transform velocity

    Every normalized action is mapped directly to the XML motor-command range
    [-200, 200]. The transform actuators are velocity servos, not angle targets.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: ConfigOverridesDict | None = None,
    ):
        super().__init__(config, config_overrides)

        self._xml_path = _XML_PATH.as_posix()
        model_xml = _XML_PATH.read_text()
        self._model_assets = common.get_assets()

        self._mj_model: MjModel = MjModel.from_xml_string(
            model_xml,
            self._model_assets,
        )
        # Set timestep before copying the model into MJX.
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(
            self._mj_model,
            impl=self._config.impl,
        )  # type: ignore

        if int(self._mj_model.nu) != POLICY_ACTION_SIZE:
            raise ValueError(
                f"Expected 6 actuators from the new XML, got nu={self._mj_model.nu}."
            )
        if int(self._mj_model.nq) != 14 or int(self._mj_model.nv) != 13:
            raise ValueError(
                "The GNN/critic constants assume nq=14 and nv=13, but the loaded "
                f"model has nq={self._mj_model.nq}, nv={self._mj_model.nv}."
            )

        self.policy_action_size = POLICY_ACTION_SIZE
        self.robot_body_id = self._mj_model.body("central_hub").id

        def sensor_adr_dim(sensor_name: str) -> tuple[int, int]:
            sid = self._mj_model.sensor(sensor_name).id
            return (
                int(self._mj_model.sensor_adr[sid]),
                int(self._mj_model.sensor_dim[sid]),
            )

        def qpos_adr(joint_name: str) -> int:
            jid = self._mj_model.joint(joint_name).id
            return int(self._mj_model.jnt_qposadr[jid])

        def qvel_adr(joint_name: str) -> int:
            jid = self._mj_model.joint(joint_name).id
            return int(self._mj_model.jnt_dofadr[jid])

        def actuator_id(actuator_name: str) -> int:
            return int(self._mj_model.actuator(actuator_name).id)

        # IMU/AHRS sensors.
        self.body_quat_adr, self.body_quat_dim = sensor_adr_dim("body_quat")
        self.body_gyro_adr, self.body_gyro_dim = sensor_adr_dim("body_gyro")
        self.body_acc_adr, self.body_acc_dim = sensor_adr_dim("body_acc")

        # Wheel velocity sensors in observation order.
        wheel_sensor_names = (
            "rear_right_wheel_vel",
            "rear_left_wheel_vel",
            "front_right_wheel_vel",
            "front_left_wheel_vel",
            "middle_support_wheel_vel",
        )
        self.wheel_vel_adrs = jp.array(
            [sensor_adr_dim(name)[0] for name in wheel_sensor_names],
            dtype=jp.int32,
        )

        self.front_right_arm_angle_adr, _ = sensor_adr_dim(
            "front_right_arm_angle"
        )
        self.front_left_arm_angle_adr, _ = sensor_adr_dim(
            "front_left_arm_angle"
        )
        self.front_right_arm_vel_adr, _ = sensor_adr_dim(
            "front_right_arm_vel"
        )
        self.front_left_arm_vel_adr, _ = sensor_adr_dim(
            "front_left_arm_vel"
        )

        # Joint addresses.
        self.middle_support_qpos = qpos_adr("middle_support_wheel_spin")
        self.rear_right_wheel_qpos = qpos_adr("rear_right_wheel_spin")
        self.rear_left_wheel_qpos = qpos_adr("rear_left_wheel_spin")
        self.front_right_arm_qpos = qpos_adr("front_right_arm_hinge")
        self.front_right_wheel_qpos = qpos_adr("front_right_wheel_spin")
        self.front_left_arm_qpos = qpos_adr("front_left_arm_hinge")
        self.front_left_wheel_qpos = qpos_adr("front_left_wheel_spin")

        self.middle_support_qvel = qvel_adr("middle_support_wheel_spin")
        self.rear_right_wheel_qvel = qvel_adr("rear_right_wheel_spin")
        self.rear_left_wheel_qvel = qvel_adr("rear_left_wheel_spin")
        self.front_right_wheel_qvel = qvel_adr("front_right_wheel_spin")
        self.front_left_wheel_qvel = qvel_adr("front_left_wheel_spin")
        self.front_right_arm_qvel = qvel_adr("front_right_arm_hinge")
        self.front_left_arm_qvel = qvel_adr("front_left_arm_hinge")

        # Retain exact actuator IDs for startup validation/debugging.
        self.actuator_ids = jp.array(
            [actuator_id(name) for name in ACTUATOR_NAMES],
            dtype=jp.int32,
        )

        # Generalized-force entries in the same order as the six policy actions.
        self.actuated_dof_adrs = jp.array([
            self.rear_right_wheel_qvel,
            self.rear_left_wheel_qvel,
            self.front_right_wheel_qvel,
            self.front_left_wheel_qvel,
            self.front_right_arm_qvel,
            self.front_left_arm_qvel,
        ], dtype=jp.int32)

        # XML units: ctrl [-200, 200] for all six actuators.
        self.ctrl_scale = jp.full((POLICY_ACTION_SIZE,), 200.0, dtype=jp.float32)

        # Joint-torque limits implied by the XML gears/force limits:
        # 0.95 kgf*cm for each wheel and 6 kgf*cm for each arm.
        self.motor_torque_limits = jp.array(
            [0.093163, 0.093163, 0.093163, 0.093163, 0.588399, 0.588399],
            dtype=jp.float32,
        )

        self.arm_limit = 2.09439510
        self.wheel_speed_scale = 10.4719755  # 100 rpm in rad/s
        self.arm_speed_scale = 1.04719755    # 10 rpm in rad/s

        self._reward_config = reward_configs()

    def _safe_root_height(
        self,
        right_arm_angle: JaxArray,
        left_arm_angle: JaxArray,
    ) -> JaxArray:
        """Raises the base enough to avoid initial wheel-floor penetration."""

        front_wheel_radius = 0.0275
        arm_origin_z = 0.0200
        wheel_local_y = -0.1250
        wheel_local_z = -0.0139
        margin = 0.0010

        def required_height(theta):
            wheel_center_z_rel = (
                arm_origin_z
                + wheel_local_y * jp.sin(theta)
                + wheel_local_z * jp.cos(theta)
            )
            return front_wheel_radius - wheel_center_z_rel + margin

        # Fixed rear and passive support wheels require about 22.4 mm.
        fixed_support_height = jp.array(0.0224, dtype=jp.float32)
        return jp.maximum(
            fixed_support_height,
            jp.maximum(
                required_height(right_arm_angle),
                required_height(left_arm_angle),
            ),
        )

    def reset(self, rng: JaxArray) -> State:
        (
            rng,
            rng_common,
            rng_mismatch,
            rng_roll,
            rng_pitch,
            rng_x,
            rng_y,
            rng_linear_vel,
            rng_angular_vel,
        ) = jax.random.split(rng, 9)

        qpos = self.mjx_model.qpos0
        qvel = jp.zeros(
            self.mjx_model.nv,
            dtype=jp.float32,
        )

        def symmetric_uniform(
            key: JaxArray,
            size: float,
            shape: tuple[int, ...] = (),
        ) -> JaxArray:
            size = jp.asarray(size, dtype=jp.float32)
            return jax.random.uniform(
                key,
                shape,
                minval=-size,
                maxval=size,
            )

        # --------------------------------------------------------
        # Arm configuration
        # --------------------------------------------------------

        common_angle = symmetric_uniform(
            rng_common,
            self._config.reset_arm_common_range,
        )

        mismatch = symmetric_uniform(
            rng_mismatch,
            self._config.reset_arm_mismatch_range,
        )

        right_arm_init = jp.clip(
            common_angle + mismatch,
            -2.02,
            2.02,
        )

        left_arm_init = jp.clip(
            common_angle - mismatch,
            -2.02,
            2.02,
        )

        # --------------------------------------------------------
        # Root position and orientation
        # --------------------------------------------------------

        start_x = symmetric_uniform(
            rng_x,
            self._config.reset_xy_range,
        )

        start_y = symmetric_uniform(
            rng_y,
            self._config.reset_xy_range,
        )

        roll = symmetric_uniform(
            rng_roll,
            self._config.reset_roll_range,
        )

        pitch = symmetric_uniform(
            rng_pitch,
            self._config.reset_pitch_range,
        )

        yaw = jp.asarray(
            self._config.reset_yaw,
            dtype=jp.float32,
        )

        # Quaternion for Rz(yaw) Ry(pitch) Rx(roll).
        cr = jp.cos(0.5 * roll)
        sr = jp.sin(0.5 * roll)
        cp = jp.cos(0.5 * pitch)
        sp = jp.sin(0.5 * pitch)
        cy = jp.cos(0.5 * yaw)
        sy = jp.sin(0.5 * yaw)

        root_quat = jp.array(
            [
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                cy * sp * cr + sy * cp * sr,
                sy * cp * cr - cy * sp * sr,
            ],
            dtype=jp.float32,
        )

        root_quat = root_quat / (
            jp.linalg.norm(root_quat) + 1.0e-8
        )

        # Base height required by the arm configuration.
        morphology_root_z = self._safe_root_height(
            right_arm_init,
            left_arm_init,
        )

        # Add clearance as roll/pitch randomization increases.
        initial_up_z = _quat_up_z(root_quat)

        tilt_fraction = jp.clip(
            1.0 - initial_up_z,
            0.0,
            1.0,
        )

        root_z = (
            morphology_root_z
            + self._config.reset_height_margin
            + self._config.reset_tilt_height_gain
            * tilt_fraction
        )

        qpos = qpos.at[0].set(start_x)
        qpos = qpos.at[1].set(start_y)
        qpos = qpos.at[2].set(root_z)
        qpos = qpos.at[3:7].set(root_quat)

        qpos = qpos.at[
            self.front_right_arm_qpos
        ].set(right_arm_init)

        qpos = qpos.at[
            self.front_left_arm_qpos
        ].set(left_arm_init)

        # Wheel phases start at zero.
        for adr in (
            self.middle_support_qpos,
            self.rear_right_wheel_qpos,
            self.rear_left_wheel_qpos,
            self.front_right_wheel_qpos,
            self.front_left_wheel_qpos,
        ):
            qpos = qpos.at[adr].set(0.0)

        # --------------------------------------------------------
        # Initial velocity
        # --------------------------------------------------------

        root_linear_vel = symmetric_uniform(
            rng_linear_vel,
            self._config.reset_linear_vel_range,
            shape=(3,),
        )

        root_angular_vel = symmetric_uniform(
            rng_angular_vel,
            self._config.reset_angular_vel_range,
            shape=(3,),
        )

        qvel = qvel.at[0:3].set(root_linear_vel)
        qvel = qvel.at[3:6].set(root_angular_vel)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        init_ctrl = jp.zeros((POLICY_ACTION_SIZE,), dtype=jp.float32)
        data = data.replace(ctrl=init_ctrl)
        data = mjx.forward(self.mjx_model, data)
        
        root_quat = data.qpos[3:7]
        qw, qx, qy, qz = root_quat

        # Initial yaw of the robot.
        initial_yaw = jp.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        # The XML robot points along local -y.
        forward_dir = jp.array([
            jp.sin(initial_yaw),
            -jp.cos(initial_yaw),
        ], dtype=jp.float32)
        
        # Local +x direction, perpendicular to forward.
        lateral_dir = jp.array([
            jp.cos(initial_yaw),
            jp.sin(initial_yaw),
        ], dtype=jp.float32)

        forward_position = -data.qpos[1]
        initial_up_z = _quat_up_z(data.qpos[3:7])

        info: dict[str, Any] = {
            "rng": rng,

            "start_xy": data.qpos[0:2],
            "forward_dir": forward_dir,
            "lateral_dir": lateral_dir,

            "prev_forward": jp.array(0.0, dtype=jp.float32),
            "prev_up_z": initial_up_z,
            "episode_steps": jp.array(0, dtype=jp.int32),

            "success": jp.array(0.0, dtype=jp.float32),
            "failure": jp.array(0.0, dtype=jp.float32),
            "stall_steps": jp.array(0.0, dtype=jp.float32),

            "last_action": jp.zeros(
                (self.policy_action_size,),
                dtype=jp.float32,
            ),
            "last_ctrl": init_ctrl,
        }
        obs = self._get_obs(data, info)
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0, dtype=jp.float32),
            done=jp.array(0.0, dtype=jp.float32),
            metrics=self._zero_metrics(),
            info=info,
        )

    def _action_to_ctrl(self, action: JaxArray) -> tuple[JaxArray, JaxArray]:
        action = jp.reshape(action, (POLICY_ACTION_SIZE,))
        clipped_action = jp.clip(action, -1.0, 1.0)
        ctrl = clipped_action * self.ctrl_scale
        return ctrl.astype(jp.float32), clipped_action.astype(jp.float32)

    def step(self, state: State, action: JaxArray) -> State:
        ctrl, clipped_action = self._action_to_ctrl(action)

        data = mjx_env.step(
            self.mjx_model,
            state.data,
            ctrl,
            self.n_substeps,
        )

        reward, success, failure, stall_steps, metrics = self._compute_reward(
            data,
            clipped_action,
            state.info,
        )

        done_success = jp.where(
            self._reward_config.terminate_on_success,
            success,
            0.0,
        )
        done = jp.maximum(failure, done_success)

        info = dict(state.info)

        displacement_xy = data.qpos[0:2] - info["start_xy"]
        current_forward = jp.dot(
            displacement_xy,
            info["forward_dir"],
        )
        current_up_z = _quat_up_z(data.qpos[3:7])

        info["prev_forward"] = current_forward
        info["success"] = success
        info["failure"] = failure
        info["stall_steps"] = stall_steps
        info["last_action"] = clipped_action
        info["last_ctrl"] = ctrl
        info["prev_up_z"] = current_up_z
        info["episode_steps"] = (
            state.info["episode_steps"]
            + jp.array(1, dtype=jp.int32)
        )
        

        return mjx_env.State(
            data=data,
            obs=self._get_obs(data, info),
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _sensor_slice(
        self,
        sensordata: JaxArray,
        adr: int,
        dim: int,
    ) -> JaxArray:
        return sensordata[adr: adr + dim]

    def _get_actor_obs(self, data: mjx.Data, info: dict[str, Any]) -> JaxArray:
        sensordata = data.sensordata

        body_quat = self._sensor_slice(
            sensordata,
            self.body_quat_adr,
            self.body_quat_dim,
        )
        body_quat = body_quat / (jp.linalg.norm(body_quat) + 1.0e-6)

        body_gyro = self._sensor_slice(
            sensordata,
            self.body_gyro_adr,
            self.body_gyro_dim,
        )
        body_acc = self._sensor_slice(
            sensordata,
            self.body_acc_adr,
            self.body_acc_dim,
        )

        gyro_obs = jp.clip(body_gyro / 10.0, -5.0, 5.0)
        accel_obs = jp.clip(body_acc / 9.81, -5.0, 5.0)

        # Signed generalized motor torque is the simulation proxy for signed
        # motor current. qfrc_actuator is used rather than raw ctrl so the
        # velocity-servo load is visible to the actor.
        motor_torque = data.qfrc_actuator[self.actuated_dof_adrs]
        current_obs = jp.clip(
            motor_torque / self.motor_torque_limits,
            -3.0,
            3.0,
        )

        wheel_vel = sensordata[self.wheel_vel_adrs]
        wheel_vel_obs = jp.clip(
            wheel_vel / self.wheel_speed_scale,
            -3.0,
            3.0,
        )

        arm_angle = jp.array([
            sensordata[self.front_right_arm_angle_adr],
            sensordata[self.front_left_arm_angle_adr],
        ], dtype=jp.float32)
        arm_angle_obs = jp.clip(arm_angle / self.arm_limit, -1.0, 1.0)

        arm_vel = jp.array([
            sensordata[self.front_right_arm_vel_adr],
            sensordata[self.front_left_arm_vel_adr],
        ], dtype=jp.float32)
        arm_vel_obs = jp.clip(
            arm_vel / self.arm_speed_scale,
            -3.0,
            3.0,
        )

        obs = jp.concatenate([
            body_quat,             # 4
            gyro_obs,              # 3
            accel_obs,             # 3
            current_obs,           # 6
            wheel_vel_obs,         # 5
            arm_angle_obs,         # 2
            arm_vel_obs,           # 2
            info["last_action"],  # 6
        ])

        if obs.shape[-1] != ACTOR_OBS_SIZE:
            raise ValueError(
                f"Actor observation must be {ACTOR_OBS_SIZE}D, got {obs.shape[-1]}."
            )
        return obs

    def _get_privileged_obs(
        self,
        data: mjx.Data,
        info: dict[str, Any],
    ) -> JaxArray:
        # Copy qpos so the original simulator state is untouched.
        qpos = data.qpos

        # Make root x/y relative to the episode spawn position.
        qpos = qpos.at[0:2].set(
            data.qpos[0:2] - info["start_xy"]
        )

        # Unlimited wheel-angle qpos entries in this XML:
        # 7  passive middle wheel
        # 8  rear-right wheel
        # 9  rear-left wheel
        # 11 front-right wheel
        # 13 front-left wheel
        wheel_qpos_indices = jp.array(
            [7, 8, 9, 11, 13],
            dtype=jp.int32,
        )

        wheel_angles = qpos[wheel_qpos_indices]

        # Wrap unlimited angles into [-pi, pi].
        wrapped_wheel_angles = jp.arctan2(
            jp.sin(wheel_angles),
            jp.cos(wheel_angles),
        )

        qpos = qpos.at[wheel_qpos_indices].set(
            wrapped_wheel_angles
        )

        # qpos order:
        # x, y, z, quaternion(4),
        # middle wheel, RR wheel, RL wheel,
        # right arm, FR wheel, left arm, FL wheel
        qpos_scale = jp.array(
            [
                1.0,                # relative x
                1.0,                # relative y
                0.20,               # root z
                1.0, 1.0, 1.0, 1.0,  # quaternion
                jp.pi,              # middle wheel
                jp.pi,              # rear-right wheel
                jp.pi,              # rear-left wheel
                self.arm_limit,     # right arm
                jp.pi,              # front-right wheel
                self.arm_limit,     # left arm
                jp.pi,              # front-left wheel
            ],
            dtype=jp.float32,
        )

        qpos_obs = jp.clip(
            qpos / qpos_scale,
            -5.0,
            5.0,
        )

        # qvel order:
        # root linear(3), root angular(3),
        # middle wheel, RR wheel, RL wheel,
        # right arm, FR wheel, left arm, FL wheel
        qvel_scale = jp.array(
            [
                3.0, 3.0, 3.0,       # root linear velocity
                10.0, 10.0, 10.0,    # root angular velocity
                self.wheel_speed_scale,
                self.wheel_speed_scale,
                self.wheel_speed_scale,
                self.arm_speed_scale,
                self.wheel_speed_scale,
                self.arm_speed_scale,
                self.wheel_speed_scale,
            ],
            dtype=jp.float32,
        )

        qvel_obs = jp.clip(
            data.qvel / qvel_scale,
            -5.0,
            5.0,
        )

        obs = jp.concatenate([
            qpos_obs,              # 14
            qvel_obs,              # 13
            info["last_action"],   # 6
        ])

        if obs.shape[-1] != PRIV_OBS_SIZE:
            raise ValueError(
                f"Privileged observation must be {PRIV_OBS_SIZE}D, "
                f"got {obs.shape[-1]}."
            )

        return obs

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]):
        return {
            "state": self._get_actor_obs(data, info),
            "privileged_state": self._get_privileged_obs(data, info),
        }

    def _compute_task_flags(
        self,
        data: mjx.Data,
        info: dict[str, Any],
    ) -> tuple[JaxArray, ...]:
        # Horizontal displacement relative to the randomized spawn point.
        displacement_xy = data.qpos[0:2] - info["start_xy"]

        # Distance traveled along the robot's initial heading.
        forward_position = jp.dot(
            displacement_xy,
            info["forward_dir"],
        )

        root_z = data.qpos[2]
        up_z = _quat_up_z(data.qpos[3:7])

        delta_forward = forward_position - info["prev_forward"]
        

        has_nan = (
            (~jp.isfinite(data.qpos)).any()
            | (~jp.isfinite(data.qvel)).any()
            | (~jp.isfinite(data.ctrl)).any()
        ).astype(jp.float32)

        success = (
            forward_position >= self._reward_config.success_forward
        ).astype(jp.float32)

        is_stalled = (
            (jp.abs(delta_forward)
             < self._reward_config.stall_delta_threshold)
            & (up_z > 0.80)
            & (
                info["episode_steps"]
                > self._reward_config.stall_grace_steps
            )
        ).astype(jp.float32)
        stall_steps = jp.where(
            is_stalled > 0.5,
            info["stall_steps"] + 1.0,
            0.0,
        )
#         failure_stall = (
#             stall_steps >= self._reward_config.max_stall_steps
#         ).astype(jp.float32)
        failure_stall = jp.array(0.0, dtype=jp.float32)

        failure_height = (
            root_z < self._reward_config.failure_min_root_z
        ).astype(jp.float32)
#         failure_tilt = (
#             up_z < self._reward_config.failure_min_up_z
#         ).astype(jp.float32)
        failure_tilt = jp.array(0.0, dtype=jp.float32)
        failure_back = (
            (forward_position < self._reward_config.failure_back_forward)
            & (
                info["episode_steps"]
                > self._reward_config.stall_grace_steps
            )
        ).astype(jp.float32)
        failure = jp.maximum(
            has_nan,
            jp.maximum(
                failure_stall,
                jp.maximum(
                    failure_height,
                    jp.maximum(failure_tilt, failure_back),
                ),
            ),
        )

        return (
            success,
            failure,
            forward_position,
            delta_forward,
            root_z,
            up_z,
            stall_steps,
            failure_stall,
            failure_back,
            is_stalled,
        )

    def _compute_reward(
        self,
        data: mjx.Data,
        action: JaxArray,
        info: dict[str, Any],
    ) -> tuple[JaxArray, JaxArray, JaxArray, JaxArray, dict[str, JaxArray]]:
        (
            success,
            failure,
            forward_position,
            delta_forward,
            root_z,
            up_z,
            stall_steps,
            failure_stall,
            failure_back,
            is_stalled,
        ) = self._compute_task_flags(data, info)

        displacement_xy = data.qpos[0:2] - info["start_xy"]
        velocity_xy = data.qvel[0:2]

        forward_velocity = jp.dot(
            velocity_xy,
            info["forward_dir"],
        )

        lateral_velocity = jp.dot(
            velocity_xy,
            info["lateral_dir"],
        )

        lateral_position = jp.dot(
            displacement_xy,
            info["lateral_dir"],
        )
        yaw_rate = data.qvel[5]

        right_arm_angle = data.qpos[self.front_right_arm_qpos]
        left_arm_angle = data.qpos[self.front_left_arm_qpos]
        arm_sync_error = (right_arm_angle - left_arm_angle) / self.arm_limit
        arm_limit_fraction = jp.maximum(
            jp.abs(right_arm_angle),
            jp.abs(left_arm_angle),
        ) / self.arm_limit
        
        

        tilt_error = 1.0 - jp.clip(up_z, -1.0, 1.0)
        up_z = _quat_up_z(data.qpos[3:7])        
        upright_gate = jp.clip(
            (up_z - 0.20) / 0.80,
            0.0,
            1.0,
        )
        
        progress_reward = (
            self._reward_config.reward_progress
            * delta_forward
            * upright_gate
        )

        forward_vel_reward = (
            self._reward_config.reward_forward_vel
            * jp.clip(
                forward_velocity,
                0.0,
                self._reward_config.max_forward_vel,
            )
            * upright_gate
        )

        recovery_reward = 4.0 * (
            up_z - info["prev_up_z"]
        )
        survival_reward = (
            self._reward_config.reward_survival * (1.0 - failure)
        )
        success_bonus = self._reward_config.reward_success * success

        control_penalty = (
            self._reward_config.penalty_control
            * jp.mean(jp.square(action))
        )
        action_rate_penalty = (
            self._reward_config.penalty_action_rate
            * jp.mean(jp.square(action - info["last_action"]))
        )
        lateral_vel_penalty = (
            self._reward_config.penalty_lateral_vel
            * jp.square(lateral_velocity)
        )
        lateral_position_penalty = (
            self._reward_config.penalty_lateral_position
            * jp.square(lateral_position)
        )
        tilt_penalty = (
            self._reward_config.penalty_tilt
            * jp.square(tilt_error)
        )
        yaw_rate_penalty = (
            self._reward_config.penalty_yaw_rate
            * jp.square(yaw_rate)
        )
        backward_penalty = (
            self._reward_config.penalty_backward
            * jp.maximum(-forward_velocity, 0.0)
        )
        stall_penalty = self._reward_config.penalty_stall * is_stalled
        failure_penalty = self._reward_config.penalty_failure * failure

        arm_sync_penalty = (
            self._reward_config.penalty_arm_sync
            * jp.square(arm_sync_error)
        )
        arm_limit_penalty = (
            self._reward_config.penalty_arm_limit
            * jp.square(jp.maximum(arm_limit_fraction - 0.90, 0.0))
        )

        reward = (
            progress_reward
            + forward_vel_reward
            + survival_reward
            + recovery_reward
            + success_bonus
            - control_penalty
            - action_rate_penalty
            - lateral_vel_penalty
            - lateral_position_penalty
            - tilt_penalty
            - yaw_rate_penalty
            - backward_penalty
            - stall_penalty
            - arm_sync_penalty
            - arm_limit_penalty
            - failure_penalty
        )
        

        metrics = {
            "reward": reward,
            "reward/total": reward,
            "reward/progress": progress_reward,
            "reward/forward_vel": forward_vel_reward,
            "reward/survival": survival_reward,
            "reward/recovery": recovery_reward,
            "reward/success_bonus": success_bonus,
            "penalty/control": control_penalty,
            "penalty/action_rate": action_rate_penalty,
            "penalty/lateral_vel": lateral_vel_penalty,
            "penalty/lateral_position": lateral_position_penalty,
            "penalty/tilt": tilt_penalty,
            "penalty/yaw_rate": yaw_rate_penalty,
            "penalty/backward": backward_penalty,
            "penalty/stall": stall_penalty,
            "penalty/arm_sync": arm_sync_penalty,
            "penalty/arm_limit": arm_limit_penalty,
            "penalty/failure": failure_penalty,
            "task/forward_position": forward_position,
            "task/forward_velocity": forward_velocity,
            "task/delta_forward": delta_forward,
            "task/lateral_position": lateral_position,
            "task/root_z": root_z,
            "task/up_z": up_z,
            "task/right_arm_angle": right_arm_angle,
            "task/left_arm_angle": left_arm_angle,
            "task/arm_sync_error": arm_sync_error,
            "task/stall_steps": stall_steps,
            "task/failure_stall": failure_stall,
            "task/failure_back": failure_back,
            "task/success": success,
            "task/failure": failure,
        }

        return reward, success, failure, stall_steps, metrics

    def _zero_metrics(self) -> dict[str, JaxArray]:
        z = jp.array(0.0, dtype=jp.float32)
        return {
            "reward": z,
            "reward/total": z,
            "reward/progress": z,
            "reward/forward_vel": z,
            "reward/survival": z,
            "reward/recovery": z,
            "reward/success_bonus": z,
            "penalty/control": z,
            "penalty/action_rate": z,
            "penalty/lateral_vel": z,
            "penalty/lateral_position": z,
            "penalty/tilt": z,
            "penalty/yaw_rate": z,
            "penalty/backward": z,
            "penalty/stall": z,
            "penalty/arm_sync": z,
            "penalty/arm_limit": z,
            "penalty/failure": z,
            "task/forward_position": z,
            "task/forward_velocity": z,
            "task/delta_forward": z,
            "task/lateral_position": z,
            "task/root_z": z,
            "task/up_z": z,
            "task/right_arm_angle": z,
            "task/left_arm_angle": z,
            "task/arm_sync_error": z,
            "task/stall_steps": z,
            "task/failure_stall": z,
            "task/failure_back": z,
            "task/success": z,
            "task/failure": z,
        }

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