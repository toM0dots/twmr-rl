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
        naconmax=16384,
        njmax=700,
        
        
        # ------------------------------------------------------------
        # Arm reset configuration
        # ------------------------------------------------------------

        # "coupled", "independent", "mixed", or "fixed"
        reset_arm_mode="mixed",

        reset_arm_joint_min=-2.02,
        reset_arm_joint_max=2.02,

        # Coupled mode.
        reset_arm_common_range=0.45,
        reset_arm_mismatch_range=0.15,

        # Independent mode.
        reset_right_arm_min=-0.50,
        reset_right_arm_max=0.50,
        reset_left_arm_min=-0.50,
        reset_left_arm_max=0.50,
        
        
        reset_arm_independent_probability=0.0,



        # Fixed mode.
        reset_right_arm_fixed=0.0,
        reset_left_arm_fixed=0.0,

        # ------------------------------------------------------------
        # Root orientation reset configuration
        # ------------------------------------------------------------

        # "upright", "side", "upside_down",
        # "mixed_recovery", "random", or "fixed"
        reset_orientation_mode="upright",

        # Noise around upright, side, and upside-down poses.
        reset_roll_noise=0.05,
        reset_pitch_noise=0.05,
        reset_yaw_noise=0.05,
        reset_yaw=0.0,

        # Mixed recovery probabilities.
        # Remaining probability becomes upside-down.
#         reset_upright_probability=0.20,
#         reset_side_probability=0.40,

        # Fully random mode.
        reset_roll_min=-jp.pi,
        reset_roll_max=jp.pi,
        reset_pitch_min=-0.50 * jp.pi,
        reset_pitch_max=0.50 * jp.pi,
        reset_yaw_min=-jp.pi,
        reset_yaw_max=jp.pi,

        # Fixed mode.
        reset_fixed_roll=0.0,
        reset_fixed_pitch=0.0,
        reset_fixed_yaw=0.0,
        
        # randomized obstacles
        reset_x_range=0.01,
        reset_y_range=0.0,

        reset_upright_probability=0.8,
        reset_side_probability=0.2,

        reset_roll_range=0.03,
        reset_pitch_range=0.03,
        reset_yaw_range=0.0,


        # ------------------------------------------------------------
        # Position and velocity reset configuration
        # ------------------------------------------------------------

        reset_xy_range=0.0,
        reset_height_margin=0.005,
        reset_tilt_height_gain=0.08,
        reset_linear_vel_range=0.0,
        reset_angular_vel_range=0.0,
        
        
        reset_side_roll_min=45.0 * jp.pi / 180.0,
        reset_side_roll_max=85.0 * jp.pi / 180.0,
        
        
    )


def reward_configs() -> config_dict.ConfigDict:
    """Compatibility accessor for the environment class tuning defaults."""
    cls = TransformableWheelMobileRobot
    return config_dict.ConfigDict({
        "success_forward": cls.success_forward,
        "failure_back_forward": cls.failure_back_forward,
        "failure_min_root_z": cls.failure_min_root_z,
        "failure_min_up_z": cls.failure_min_up_z,
        "stall_delta_threshold": cls.stall_delta_threshold,
        "max_stall_steps": cls.max_stall_steps,
        "max_forward_vel": cls.max_forward_vel,
        "stall_grace_steps": cls.stall_grace_steps,
        "reward_progress": cls.reward_progress,
        "reward_forward_vel": cls.reward_forward_vel,
        "reward_survival": cls.reward_survival,
        "reward_success": cls.reward_success,
        "penalty_control": cls.penalty_control,
        "penalty_action_rate": cls.penalty_action_rate,
        "penalty_lateral_vel": cls.penalty_lateral_vel,
        "penalty_lateral_position": cls.penalty_lateral_position,
        "penalty_tilt": cls.penalty_tilt,
        "penalty_yaw_rate": cls.penalty_yaw_rate,
        "penalty_backward": cls.penalty_backward,
        "penalty_stall": cls.penalty_stall,
        "penalty_failure": cls.penalty_failure,
        "penalty_arm_sync": cls.penalty_arm_sync,
        "penalty_arm_limit": cls.penalty_arm_limit,
        "terminate_on_success": cls.terminate_on_success,
        "step_center_forward": cls.step_center_forward,
        "step_half_depth": cls.step_half_depth,
        "step_reached_forward": cls.step_reached_forward,
        "step_crossed_forward": cls.step_crossed_forward,
        "success_min_up_z": cls.success_min_up_z,
        "reward_reach_step": cls.reward_reach_step,
        "reward_cross_step": cls.reward_cross_step,
        "reward_upright": cls.reward_upright,
        "reward_stabilize_after_crossing": cls.reward_stabilize_after_crossing,
        "penalty_post_cross_tilt": cls.penalty_post_cross_tilt,
    })


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

    # Central tuning panel. Edit before constructing/JIT-compiling the environment.
    # Original reward weights and milestone behavior are retained.
    success_forward = 0.75  # Travel-distance goal from spawn, m.
    failure_back_forward = -0.4  # Backward displacement failure boundary, m.
    failure_min_root_z = 0.005  # Minimum root height, m.
    failure_min_up_z = 0.05  # Legacy tilt threshold; tilt failure is disabled.
    stall_delta_threshold = 0.0002  # Maximum absolute forward displacement per control step counted as stalled, m.
    max_stall_steps = 250  # Legacy failure threshold; stall termination remains disabled.
    max_forward_vel = 5.0  # Clip for positive forward-speed reward, m/s.
    stall_grace_steps = 100  # Startup grace period before stall counting and backward failure.
    reward_progress = 12.0  # Weight on forward displacement, gated by orientation and goal.
    reward_forward_vel = 0.1  # Weight on clipped forward speed.
    reward_survival = 0.0002  # Reward per surviving control step.
    reward_success = 150.0  # Bonus on each success step; normally terminates immediately.
    penalty_control = 0.001  # Weight on mean squared applied action.
    penalty_action_rate = 0.005  # Weight on squared change in applied action.
    penalty_lateral_vel = 0.2  # Weight on squared lateral velocity.
    penalty_lateral_position = 0.05  # Weight on squared lateral displacement.
    penalty_tilt = 0.03  # Weight on squared (1 - up_z).
    penalty_yaw_rate = 0.05  # Weight on squared world angular-z velocity.
    penalty_backward = 1.0  # Weight on backward velocity magnitude.
    penalty_stall = 2e-08  # Penalty per stalled control step.
    penalty_failure = 20.0  # Penalty on failure.
    penalty_arm_sync = 0.0  # Weight on squared normalized arm difference; zero disables.
    penalty_arm_limit = 0.03  # Weight on squared excess over soft arm-limit fraction.
    terminate_on_success = True  # End episode when existing success condition is met.
    step_center_forward = 0.3  # Legacy obstacle geometry metadata, m.
    step_half_depth = 0.025  # Legacy obstacle geometry metadata, m.
    step_reached_forward = 0.25  # Existing reach milestone, m from spawn.
    step_crossed_forward = 0.42  # Existing crossing milestone, m from spawn.
    success_min_up_z = 0.7  # Original success/upright threshold; distinct from stricter gate.
    reward_reach_step = 4.0  # One-time reach-milestone bonus.
    reward_cross_step = 10.0  # One-time crossing-milestone bonus.
    reward_upright = 0.0  # Weight on (up_z + 1)/2; zero disables.
    reward_stabilize_after_crossing = 30.0  # One-time post-crossing upright bonus.
    penalty_post_cross_tilt = 0.2  # Weight on squared tilt deficit after crossing.

    # Recovery and stall-based arm gate.
    recovery_weight = 4.0  # Weight on up_z improvement per control step.
    locomotion_min_up_z = 0.2  # Orientation at which forward rewards begin; full weight at up_z=1.
    arm_limit_soft_fraction = 0.85  # Fraction of arm range beyond which arm-limit penalty begins.
    gate_recovery_tilt_deg = 8.0  # Maximum body tilt for closing the arm gate, degrees.
    gate_recovery_gyro_rad_s = 0.3  # Maximum body gyro norm for closing the arm gate, rad/s.
    gate_neutral_arm_deg = 8.0  # Maximum absolute arm angle for closing the gate, degrees.
    gate_reopen_tilt_deg = 15.0  # Body tilt above which recovery reopens the gate, degrees.
    gate_stall_steps = 50  # Consecutive stalled steps needed to unlock arms; 0.6 s at dt=0.02.
    stall_forward_command_min = 0.25  # Minimum mean normalized forward wheel command to count a stall.
    stall_min_up_z = 0.8  # Count stalls only when body up_z exceeds this threshold.
    gate_assist_grace_s = 0.5  # Time allowed to begin an arm maneuver before flat posture can close the gate.
    # Reward for maintaining an upright body; original value was zero.
    reward_upright = 0.001

    # Additional reward per control step for upright, neutral arm posture.
    reward_neutral_posture = 0.002

    # Body orientation where the neutral-posture reward starts increasing.
    neutral_posture_min_up_z = 0.850

    # Arm-angle scale controlling how gradually neutral posture is rewarded.
    neutral_posture_arm_scale_deg = 30.0
    
    diagnostic_action_saturation_threshold = 0.99
    
    
    # Additional penalties per control step, before PPO reward_scaling.
    # Starting weights for tuning; they are not motor parameters.
    arm_quiet_command_weight = 0.010
    arm_quiet_velocity_weight = 0.005
    arm_quiet_posture_weight = 0.002

    # Full quiet-arm preference at <= 8 degrees chassis tilt.
    # Fade to zero by 20 degrees, allowing recovery motions.
    arm_quiet_full_tilt_deg = 8.0
    arm_quiet_off_tilt_deg = 20.0

    # Relax these penalties after persistent stall detection.
    arm_assist_stall_threshold = 40.0
    arm_assist_stall_ramp = 10.0  # Fully relaxed at 40 stalled steps.

    # Desired arm posture and normalization scales.
    arm_quiet_neutral_deg = 0.0
    arm_quiet_angle_scale_deg = 30.0
    arm_quiet_speed_scale_rad_s = 1.04719755  # 10 rpm.

    # Bound these reward terms during large excursions.
    # These caps affect penalties only, not motion or actuator commands.
    arm_quiet_velocity_ratio_cap = 2.0
    arm_quiet_angle_ratio_cap = 2.0

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

        # Reward and gate settings are class attributes in the tuning panel.
        
        
    def _symmetric_uniform(
            self,
            key: JaxArray,
            half_range: float | JaxArray,
            shape: tuple[int, ...] = (),
        ) -> JaxArray:
            """Sample uniformly from [-half_range, +half_range]."""
            half_range = jp.asarray(
                half_range,
                dtype=jp.float32,
            )

            return jax.random.uniform(
                key,
                shape=shape,
                minval=-half_range,
                maxval=half_range,
                dtype=jp.float32,
            )
    

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



    def _uniform_range(
        self,
        key: JaxArray,
        minimum: float | JaxArray,
        maximum: float | JaxArray,
        shape: tuple[int, ...] = (),
    ) -> JaxArray:
        """Sample uniformly from [minimum, maximum]."""
        minimum = jp.asarray(
            minimum,
            dtype=jp.float32,
        )

        maximum = jp.asarray(
            maximum,
            dtype=jp.float32,
        )

        return jax.random.uniform(
            key,
            shape=shape,
            minval=minimum,
            maxval=maximum,
            dtype=jp.float32,
        )


    def _sample_reset_arm_angles(
        self,
        *,
        rng_mode: JaxArray,
        rng_common: JaxArray,
        rng_mismatch: JaxArray,
        rng_right: JaxArray,
        rng_left: JaxArray,
    ) -> tuple[JaxArray, JaxArray]:
        """Sample the two independent transform-arm angles.

        Available modes:

        coupled:
            q_right = common + mismatch
            q_left  = common - mismatch

        independent:
            Right and left angles are sampled independently.

        mixed:
            Select coupled or independent initialization at each reset.

        fixed:
            Use fixed configured arm angles.
        """
        mode = str(
            self._config.reset_arm_mode
        ).lower()

        joint_min = jp.asarray(
            self._config.reset_arm_joint_min,
            dtype=jp.float32,
        )

        joint_max = jp.asarray(
            self._config.reset_arm_joint_max,
            dtype=jp.float32,
        )

        common_angle = self._symmetric_uniform(
            rng_common,
            self._config.reset_arm_common_range,
        )

        mismatch = self._symmetric_uniform(
            rng_mismatch,
            self._config.reset_arm_mismatch_range,
        )

        coupled_right = common_angle + mismatch
        coupled_left = common_angle - mismatch

        independent_right = self._uniform_range(
            rng_right,
            self._config.reset_right_arm_min,
            self._config.reset_right_arm_max,
        )

        independent_left = self._uniform_range(
            rng_left,
            self._config.reset_left_arm_min,
            self._config.reset_left_arm_max,
        )

        fixed_right = jp.asarray(
            self._config.reset_right_arm_fixed,
            dtype=jp.float32,
        )

        fixed_left = jp.asarray(
            self._config.reset_left_arm_fixed,
            dtype=jp.float32,
        )

        if mode == "coupled":
            right_arm_init = coupled_right
            left_arm_init = coupled_left

        elif mode == "independent":
            right_arm_init = independent_right
            left_arm_init = independent_left

        elif mode == "mixed":
            use_independent = jax.random.bernoulli(
                rng_mode,
                p=jp.asarray(
                    self._config.reset_arm_independent_probability,
                    dtype=jp.float32,
                ),
            )

            right_arm_init = jp.where(
                use_independent,
                independent_right,
                coupled_right,
            )

            left_arm_init = jp.where(
                use_independent,
                independent_left,
                coupled_left,
            )

        elif mode == "fixed":
            right_arm_init = fixed_right
            left_arm_init = fixed_left

        else:
            raise ValueError(
                "reset_arm_mode must be one of "
                "'coupled', 'independent', 'mixed', or 'fixed'; "
                f"got {mode!r}"
            )

        right_arm_init = jp.clip(
            right_arm_init,
            joint_min,
            joint_max,
        )

        left_arm_init = jp.clip(
            left_arm_init,
            joint_min,
            joint_max,
        )

        return right_arm_init, left_arm_init


    def _rpy_to_quaternion(
        self,
        roll: JaxArray,
        pitch: JaxArray,
        yaw: JaxArray,
    ) -> JaxArray:
        """Quaternion [w, x, y, z] for Rz(yaw) Ry(pitch) Rx(roll)."""
        cr = jp.cos(0.5 * roll)
        sr = jp.sin(0.5 * roll)

        cp = jp.cos(0.5 * pitch)
        sp = jp.sin(0.5 * pitch)

        cy = jp.cos(0.5 * yaw)
        sy = jp.sin(0.5 * yaw)

        quaternion = jp.array(
            [
                cy * cp * cr + sy * sp * sr,
                cy * cp * sr - sy * sp * cr,
                cy * sp * cr + sy * cp * sr,
                sy * cp * cr - cy * sp * sr,
            ],
            dtype=jp.float32,
        )

        return quaternion / (
            jp.linalg.norm(quaternion) + 1.0e-8
        )


    def _sample_reset_orientation(
        self,
        *,
        rng_mode: JaxArray,
        rng_side: JaxArray,
        rng_roll: JaxArray,
        rng_pitch: JaxArray,
        rng_yaw: JaxArray,
    ) -> JaxArray:
        """Sample the initial robot orientation.

        Available modes:

        upright:
            Upright with configurable roll/pitch/yaw noise.

        side:
            Randomly placed on the left or right side.

        upside_down:
            Approximately 180-degree roll.

        mixed_recovery:
            Mixture of upright, side, and upside-down poses.

        random:
            Roll, pitch, and yaw are sampled from explicit intervals.

        fixed:
            Use one fixed roll/pitch/yaw pose.
        """
        mode = str(
            self._config.reset_orientation_mode
        ).lower()
        
        rng_mode_select, rng_side_magnitude = jax.random.split(
            rng_mode
        )
        roll_noise = self._symmetric_uniform(
            rng_roll,
            self._config.reset_roll_noise,
        )
        
        pitch_noise = self._symmetric_uniform(
            rng_pitch,
            self._config.reset_pitch_noise,
        )

        yaw_noise = self._symmetric_uniform(
            rng_yaw,
            self._config.reset_yaw_noise,
        )

        base_yaw = jp.asarray(
            self._config.reset_yaw,
            dtype=jp.float32,
        )

        side_sign = jp.where(
            jax.random.bernoulli(rng_side),
            jp.asarray(1.0, dtype=jp.float32),
            jp.asarray(-1.0, dtype=jp.float32),
        )

        upright_roll = roll_noise

        side_magnitude = jax.random.uniform(
            rng_side_magnitude,
            (),
            minval=jp.asarray(
                self._config.reset_side_roll_min,
                dtype=jp.float32,
            ),
            maxval=jp.asarray(
                self._config.reset_side_roll_max,
                dtype=jp.float32,
            ),
        )

        side_roll = (
            side_sign * side_magnitude
            + roll_noise
        )

        upside_down_roll = (
            jp.pi + roll_noise
        )

        if mode == "upright":
            roll = upright_roll
            pitch = pitch_noise
            yaw = base_yaw + yaw_noise

        elif mode == "side":
            roll = side_roll
            pitch = pitch_noise
            yaw = base_yaw + yaw_noise

        elif mode == "upside_down":
            roll = upside_down_roll
            pitch = pitch_noise
            yaw = base_yaw + yaw_noise

        elif mode == "mixed_recovery":
            sample = jax.random.uniform(
                rng_mode_select,
                (),
                dtype=jp.float32,
            )

            upright_probability = jp.asarray(
                self._config.reset_upright_probability,
                dtype=jp.float32,
            )

            side_probability = jp.asarray(
                self._config.reset_side_probability,
                dtype=jp.float32,
            )

            upright_boundary = upright_probability

            side_boundary = (
                upright_probability
                + side_probability
            )

            # Remaining probability is assigned to upside-down poses.
            roll = jp.where(
                sample < upright_boundary,
                upright_roll,
                jp.where(
                    sample < side_boundary,
                    side_roll,
                    upside_down_roll,
                ),
            )

            pitch = pitch_noise
            yaw = base_yaw + yaw_noise

        elif mode == "random":
            roll = self._uniform_range(
                rng_roll,
                self._config.reset_roll_min,
                self._config.reset_roll_max,
            )

            pitch = self._uniform_range(
                rng_pitch,
                self._config.reset_pitch_min,
                self._config.reset_pitch_max,
            )

            yaw = self._uniform_range(
                rng_yaw,
                self._config.reset_yaw_min,
                self._config.reset_yaw_max,
            )

        elif mode == "fixed":
            roll = jp.asarray(
                self._config.reset_fixed_roll,
                dtype=jp.float32,
            )

            pitch = jp.asarray(
                self._config.reset_fixed_pitch,
                dtype=jp.float32,
            )

            yaw = jp.asarray(
                self._config.reset_fixed_yaw,
                dtype=jp.float32,
            )

        else:
            raise ValueError(
                "reset_orientation_mode must be one of "
                "'upright', 'side', 'upside_down', "
                "'mixed_recovery', 'random', or 'fixed'; "
                f"got {mode!r}"
            )

        return self._rpy_to_quaternion(
            roll,
            pitch,
            yaw,
        )


    def _heading_vectors_from_quaternion(
        self,
        quaternion: JaxArray,
    ) -> tuple[JaxArray, JaxArray]:
        """Get horizontal forward and lateral directions.

        The robot points along local -y in the XML.
        """
        qw, qx, qy, qz = quaternion

        # World projection of robot local -y.
        forward_dir = jp.array(
            [
                2.0 * (qw * qz - qx * qy),
                -1.0 + 2.0 * (qx * qx + qz * qz),
            ],
            dtype=jp.float32,
        )

        forward_norm = jp.linalg.norm(
            forward_dir
        )

        # Avoid unstable normalization in a nearly vertical orientation.
        default_forward = jp.array(
            [0.0, -1.0],
            dtype=jp.float32,
        )

        forward_dir = jp.where(
            forward_norm > 1.0e-6,
            forward_dir / (
                forward_norm + 1.0e-8
            ),
            default_forward,
        )

        lateral_dir = jp.array(
            [
                -forward_dir[1],
                forward_dir[0],
            ],
            dtype=jp.float32,
        )

        return forward_dir, lateral_dir
    

    def reset(
        self,
        rng: JaxArray,
    ) -> State:
        (
            next_rng,
            rng_arm_mode,
            rng_arm_common,
            rng_arm_mismatch,
            rng_right_arm,
            rng_left_arm,
            rng_orientation_mode,
            rng_side,
            rng_roll,
            rng_pitch,
            rng_yaw,
            rng_x,
            rng_y,
            rng_linear_vel,
            rng_angular_vel,
        ) = jax.random.split(rng, 15)

        qpos = self.mjx_model.qpos0

        qvel = jp.zeros(
            self.mjx_model.nv,
            dtype=jp.float32,
        )
        
        right_arm_init, left_arm_init = self._sample_reset_arm_angles(
            rng_mode=rng_arm_mode,
            rng_common=rng_arm_common,
            rng_mismatch=rng_arm_mismatch,
            rng_right=rng_right_arm,
            rng_left=rng_left_arm,
        )

        
#         # --------------------------------------------------------
#         # Arm configuration
#         # --------------------------------------------------------

#         arm_joint_min = jp.asarray(
#             self._config.reset_arm_joint_min,
#             dtype=jp.float32,
#         )

#         arm_joint_max = jp.asarray(
#             self._config.reset_arm_joint_max,
#             dtype=jp.float32,
#         )

#         # Original coupled-arm initialization.
#         common_angle = self._symmetric_uniform(
#             rng_arm_common,
#             self._config.reset_arm_common_range,
#         )

#         mismatch = self._symmetric_uniform(
#             rng_arm_mismatch,
#             self._config.reset_arm_mismatch_range,
#         )

#         coupled_right_arm = common_angle + mismatch
#         coupled_left_arm = common_angle - mismatch

#         # Fully independent initialization.
#         independent_right_arm = jax.random.uniform(
#             rng_right_arm,
#             (),
#             minval=jp.asarray(
#                 self._config.reset_right_arm_min,
#                 dtype=jp.float32,
#             ),
#             maxval=jp.asarray(
#                 self._config.reset_right_arm_max,
#                 dtype=jp.float32,
#             ),
#             dtype=jp.float32,
#         )

#         independent_left_arm = jax.random.uniform(
#             rng_left_arm,
#             (),
#             minval=jp.asarray(
#                 self._config.reset_left_arm_min,
#                 dtype=jp.float32,
#             ),
#             maxval=jp.asarray(
#                 self._config.reset_left_arm_max,
#                 dtype=jp.float32,
#             ),
#             dtype=jp.float32,
#         )

#         # Choose independent or coupled initialization for this episode.
#         use_independent_arms = jax.random.bernoulli(
#             rng_arm_mode,
#             p=jp.asarray(
#                 self._config.reset_arm_independent_probability,
#                 dtype=jp.float32,
#             ),
#         )

#         right_arm_init = jp.where(
#             use_independent_arms,
#             independent_right_arm,
#             coupled_right_arm,
#         )

#         left_arm_init = jp.where(
#             use_independent_arms,
#             independent_left_arm,
#             coupled_left_arm,
#         )

#         right_arm_init = jp.clip(
#             right_arm_init,
#             arm_joint_min,
#             arm_joint_max,
#         )

#         left_arm_init = jp.clip(
#             left_arm_init,
#             arm_joint_min,
#             arm_joint_max,
#         )
        
        # --------------------------------------------------------
        # Root position and orientation
        # --------------------------------------------------------

        start_x = self._symmetric_uniform(
            rng_x,
            self._config.reset_x_range,
        )

        start_y = self._symmetric_uniform(
            rng_y,
            self._config.reset_y_range,
        )

        root_quat = self._sample_reset_orientation(
            rng_mode=rng_orientation_mode,
            rng_side=rng_side,
            rng_roll=rng_roll,
            rng_pitch=rng_pitch,
            rng_yaw=rng_yaw,
        )

        morphology_root_z = self._safe_root_height(
            right_arm_init,
            left_arm_init,
        )

        initial_up_z = _quat_up_z(
            root_quat
        )

        # 0 upright, 0.5 sideways, 1 upside down.
        tilt_fraction = jp.clip(
            0.5 * (1.0 - initial_up_z),
            0.0,
            1.0,
        )

        root_z = (
            morphology_root_z
            + jp.asarray(
                self._config.reset_height_margin,
                dtype=jp.float32,
            )
            + jp.asarray(
                self._config.reset_tilt_height_gain,
                dtype=jp.float32,
            )
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

        # Reset wheel phases.
        for address in (
            self.middle_support_qpos,
            self.rear_right_wheel_qpos,
            self.rear_left_wheel_qpos,
            self.front_right_wheel_qpos,
            self.front_left_wheel_qpos,
        ):
            qpos = qpos.at[address].set(0.0)

        # --------------------------------------------------------
        # Initial root velocity
        # --------------------------------------------------------

        root_linear_vel = self._symmetric_uniform(
            rng_linear_vel,
            self._config.reset_linear_vel_range,
            shape=(3,),
        )

        root_angular_vel = self._symmetric_uniform(
            rng_angular_vel,
            self._config.reset_angular_vel_range,
            shape=(3,),
        )

        qvel = qvel.at[0:3].set(
            root_linear_vel
        )

        qvel = qvel.at[3:6].set(
            root_angular_vel
        )

        # --------------------------------------------------------
        # Construct MuJoCo state
        # --------------------------------------------------------

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )

        init_ctrl = jp.zeros(
            (self.policy_action_size,),
            dtype=jp.float32,
        )

        data = data.replace(
            ctrl=init_ctrl
        )

        data = mjx.forward(
            self.mjx_model,
            data,
        )

        start_xy = data.qpos[0:2]

        forward_dir, lateral_dir = (
            self._heading_vectors_from_quaternion(
                data.qpos[3:7]
            )
        )

        initial_up_z = _quat_up_z(
            data.qpos[3:7]
        )

        info: dict[str, Any] = {
            "rng": next_rng,

            # Episode-relative coordinate frame.
            "start_xy": start_xy,
            "forward_dir": forward_dir,
            "lateral_dir": lateral_dir,

            # Previous-step quantities.
            "prev_forward": jp.asarray(
                0.0,
                dtype=jp.float32,
            ),
            "prev_up_z": initial_up_z,
            "episode_steps": jp.asarray(
                0,
                dtype=jp.int32,
            ),

            # Reset diagnostics.
            "initial_right_arm_angle": (
                right_arm_init
            ),
            "initial_left_arm_angle": (
                left_arm_init
            ),
            "initial_up_z": initial_up_z,
            
            "stabilized_after_crossing": jp.array(
                0.0,
                dtype=jp.float32,
            ),

            # Task state.
            "success": jp.asarray(
                0.0,
                dtype=jp.float32,
            ),
            "failure": jp.asarray(
                0.0,
                dtype=jp.float32,
            ),
            "stall_steps": jp.asarray(
                0.0,
                dtype=jp.float32,
            ),

            "last_action": jp.zeros(
                (self.policy_action_size,),
                dtype=jp.float32,
            ),
            "last_ctrl": init_ctrl,
            
            "reached_step": jp.array(
                0.0,
                dtype=jp.float32,
            ),

            "crossed_step": jp.array(
                0.0,
                dtype=jp.float32,
            ),
            "upright_steps": jp.array(
                0,
                dtype=jp.int32,
            ),
            
        }

        # Arm gate state only; actor/critic observations remain unchanged.
        info["arm_enabled"] = jp.array(True)
        info["arm_stall_s"] = jp.array(0.0, dtype=jp.float32)
        info["arm_assist_s"] = jp.array(0.0, dtype=jp.float32)

        obs = self._get_obs(
            data,
            info,
        )

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.asarray(
                0.0,
                dtype=jp.float32,
            ),
            done=jp.asarray(
                0.0,
                dtype=jp.float32,
            ),
            metrics=self._zero_metrics(),
            info=info,
        )

    def _action_to_ctrl(self, action: JaxArray) -> tuple[JaxArray, JaxArray]:
        action = jp.reshape(action, (POLICY_ACTION_SIZE,))
        clipped_action = jp.clip(action, -1.0, 1.0)
        ctrl = clipped_action * self.ctrl_scale
        return ctrl.astype(jp.float32), clipped_action.astype(jp.float32)

    def step(self, state: State, action: JaxArray) -> State:
        # Final arm-command gate, before conversion to XML control units.
        up_z = _quat_up_z(state.data.qpos[3:7])
        arm_q = state.data.qpos[jp.array([
            self.front_right_arm_qpos, self.front_left_arm_qpos])]
        gyro = state.data.sensordata[
            self.body_gyro_adr:self.body_gyro_adr + self.body_gyro_dim]
        recovered = ((up_z >= jp.cos(jp.deg2rad(self.gate_recovery_tilt_deg)))
                     & (jp.linalg.norm(gyro) <= self.gate_recovery_gyro_rad_s)
                     & (jp.max(jp.abs(arm_q)) <= jp.deg2rad(self.gate_neutral_arm_deg)))

        # Read the completed-step displacement counter, not wheel speed.
        trying_forward = (
            jp.mean(state.info["last_action"][:4])
            > self.stall_forward_command_min
        )
        blocked = trying_forward & (
            state.info["stall_steps"] >= self.gate_stall_steps
        )
        start_assist = blocked & ~state.info["arm_enabled"]
        arm_stall_s = state.info["stall_steps"] * self.dt
        arm_assist_s = jp.where(
            start_assist, self.gate_assist_grace_s,
            jp.maximum(state.info["arm_assist_s"] - self.dt, 0.0))

        # Stop on this tick, without waiting for arm velocity to become small.
        # Allow the configured grace period for a climbing attempt to leave rest.
        arm_enabled = jp.where(
            recovered & (arm_assist_s <= 0.0), False,
            state.info["arm_enabled"])
        arm_enabled = (arm_enabled | start_assist
                       | (up_z < jp.cos(jp.deg2rad(self.gate_reopen_tilt_deg))))
        action = jp.asarray(action).reshape((POLICY_ACTION_SIZE,))
        requested_arm_action = jp.clip(action[4:6], -1.0, 1.0)
#         action = action.at[4:6].set(jp.where(arm_enabled, action[4:6], 0.0))

        ctrl, clipped_action = self._action_to_ctrl(action)

        data = mjx_env.step(
            self.mjx_model,
            state.data,
            ctrl,
            self.n_substeps,
        )

        reward, success, failure, stall_steps, reached_step, crossed_step, stable_after_crossing,  upright_steps, metrics = self._compute_reward(
            data,
            clipped_action,
            state.info,
        )
        
        # Add diagnostics to the new metrics dictionary returned by _compute_reward.
        # clipped_action contains the actual normalized commands after the arm gate.
        applied_saturated = (
            jp.abs(clipped_action)
            > self.diagnostic_action_saturation_threshold
        ).astype(jp.float32)

        metrics.update({
            "diagnostic/control_steps": jp.array(1.0, dtype=jp.float32),
            "diagnostic/wheel_saturation_steps": jp.mean(applied_saturated[:4]),
            "diagnostic/arm_saturation_steps": jp.mean(applied_saturated[4:6]),
        })

        # Diagnostics only: summed by the existing episode metric logger.
        # Gate decisions describe the command applied over this control interval.
        held = ~arm_enabled
        arm_q_after = data.qpos[jp.array([
            self.front_right_arm_qpos, self.front_left_arm_qpos])]
        metrics.update({
            "gate/enabled_s": arm_enabled.astype(jp.float32) * self.dt,
            "gate/held_s": held.astype(jp.float32) * self.dt,
            "gate/recovered_s": recovered.astype(jp.float32) * self.dt,
            "gate/blocked_s": blocked.astype(jp.float32) * self.dt,
            "gate/assist_starts": start_assist.astype(jp.float32),
            "gate/close_events": (state.info["arm_enabled"] & held).astype(jp.float32),
            "gate/open_events": (~state.info["arm_enabled"] & arm_enabled).astype(jp.float32),
            "gate/held_requested_abs_action_s": jp.where(
                held, jp.mean(jp.abs(requested_arm_action)) * self.dt, 0.0),
            "gate/held_applied_abs_action_s": jp.where(
                held, jp.mean(jp.abs(clipped_action[4:6])) * self.dt, 0.0),
            "gate/held_arm_displacement_rad": jp.where(
                held, jp.sum(jp.abs(arm_q_after - arm_q)), 0.0),
        })

        done_success = jp.where(
            self.terminate_on_success,
            success,
            0.0,
        )
        done = jp.maximum(failure, done_success)

        info = dict(state.info)
        info["arm_enabled"] = arm_enabled
        info["arm_stall_s"] = jp.where(arm_enabled, 0.0, arm_stall_s)
        info["arm_assist_s"] = arm_assist_s

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
        info["stabilized_after_crossing"] = jp.maximum(
            state.info["stabilized_after_crossing"],
            stable_after_crossing,
        )
        info["reached_step"] = jp.maximum(
            state.info["reached_step"],
            reached_step,
        )

        info["crossed_step"] = jp.maximum(
            state.info["crossed_step"],
            crossed_step,
        )

        
        info["upright_steps"] = upright_steps

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
        

        reached_step = (
            forward_position
            >= self.step_reached_forward
        ).astype(jp.float32)

        crossed_step = (
            forward_position
            >= self.step_crossed_forward
        ).astype(jp.float32)

        success = (
            (crossed_step > 0.5)
            & (
                forward_position
                >= self.success_forward
            )
            & (
                up_z
                >= self.success_min_up_z
            )
        ).astype(jp.float32)
        
        # recovery only 
        is_upright = up_z >= self.success_min_up_z

        upright_steps = jp.where(
            is_upright,
            info["upright_steps"] + 1,
            0,
        )
        success = (
                 (forward_position >= self.success_forward)
                & (up_z >= self.success_min_up_z)
            ).astype(jp.float32)

        
        is_stalled = (
            (jp.abs(delta_forward)
             < self.stall_delta_threshold)
            & (up_z > self.stall_min_up_z)
            & (jp.mean(info["last_action"][:4]) > self.stall_forward_command_min)
            & (
                info["episode_steps"]
                > self.stall_grace_steps
            )
        ).astype(jp.float32)
        stall_steps = jp.where(
            is_stalled > 0.5,
            info["stall_steps"] + 1.0,
            0.0,
        )
#         failure_stall = (
#             stall_steps >= self.max_stall_steps
#         ).astype(jp.float32)
        failure_stall = jp.array(0.0, dtype=jp.float32)

        failure_height = (
            root_z < self.failure_min_root_z
        ).astype(jp.float32)
#         failure_tilt = (
#             up_z < self.failure_min_up_z
#         ).astype(jp.float32)
        failure_tilt = jp.array(0.0, dtype=jp.float32)
        failure_back = (
            (forward_position < self.failure_back_forward)
            & (
                info["episode_steps"]
                > self.stall_grace_steps
            )
        ).astype(jp.float32)
        failure = jp.maximum(
            has_nan,
            jp.maximum(
                failure_stall,
                jp.maximum(
                    failure_height,
                    failure_back,
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
            reached_step,
            crossed_step,
             upright_steps,
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
            reached_step,
            crossed_step,
            upright_steps,
        ) = self._compute_task_flags(data, info)
        
        # ------------------------------------------------------------
        # Quiet-arm preference: strong when upright, relaxed when stalled.
        # This changes reward only. It does not overwrite any action.
        # ------------------------------------------------------------
        full_up_z = jp.cos(
            jp.deg2rad(self.arm_quiet_full_tilt_deg)
        )
        off_up_z = jp.cos(
            jp.deg2rad(self.arm_quiet_off_tilt_deg)
        )

        flat_confidence = jp.clip(
            (up_z - off_up_z) / (full_up_z - off_up_z),
            0.0,
            1.0,
        )

        stall_assistance = jp.clip(
            (
                stall_steps
                - self.arm_assist_stall_threshold
            ) / self.arm_assist_stall_ramp,
            0.0,
            1.0,
        )

        arm_quiet_weight = (
            flat_confidence * (1.0 - stall_assistance)
        )

        # Arm coordinates from your existing joint-address attributes.
        arm_angles = data.qpos[jp.array([
            self.front_right_arm_qpos,
            self.front_left_arm_qpos,
        ])]

        arm_velocities = data.qvel[jp.array([
            self.front_right_arm_qvel,
            self.front_left_arm_qvel,
        ])]

        # action is normalized to [-1, 1], before multiplication by 200.
        arm_commands = jp.clip(action[4:6], -1.0, 1.0)

        velocity_ratio = jp.clip(
            arm_velocities / self.arm_quiet_speed_scale_rad_s,
            -self.arm_quiet_velocity_ratio_cap,
            self.arm_quiet_velocity_ratio_cap,
        )

        angle_ratio = jp.clip(
            (
                arm_angles
                - jp.deg2rad(self.arm_quiet_neutral_deg)
            ) / jp.deg2rad(self.arm_quiet_angle_scale_deg),
            -self.arm_quiet_angle_ratio_cap,
            self.arm_quiet_angle_ratio_cap,
        )

        arm_command_penalty = (
            self.arm_quiet_command_weight
            * arm_quiet_weight
            * jp.mean(jp.square(arm_commands))
        )

        arm_velocity_penalty = (
            self.arm_quiet_velocity_weight
            * arm_quiet_weight
            * jp.mean(jp.square(velocity_ratio))
        )

        arm_posture_penalty = (
            self.arm_quiet_posture_weight
            * (1.0 - stall_assistance)
            * jp.mean(jp.square(angle_ratio))
        )


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
        
        reached_goal = (
            forward_position >= self.success_forward
        )

#         stable_now = (
#             reached_goal
#             & (up_z >= 0.90)
#             & (jp.abs(lateral_velocity) < 0.15)
#             & (angular_speed < 0.5)
#         )
        before_goal = (~reached_goal).astype(jp.float32)

        tilt_error = 1.0 - jp.clip(up_z, -1.0, 1.0)
        up_z = _quat_up_z(data.qpos[3:7])        
        upright_gate = jp.clip(
            (up_z - self.locomotion_min_up_z) / (1.0 - self.locomotion_min_up_z),
            0.0,
            1.0,
        )
        
        progress_reward = (
            self.reward_progress
            * delta_forward
            * upright_gate
#             * before_goal
        )

        forward_vel_reward = (
            self.reward_forward_vel
            * jp.clip(
                forward_velocity,
                0.0,
                self.max_forward_vel,
            )
            * upright_gate
        )

        recovery_reward = self.recovery_weight * (
            up_z - info["prev_up_z"]
        )
        survival_reward = (
            self.reward_survival * (1.0 - failure)
        )
        success_bonus = self.reward_success * success

        control_penalty = (
            self.penalty_control
            * jp.mean(jp.square(action))
        )
        action_rate_penalty = (
            self.penalty_action_rate
            * jp.mean(jp.square(action - info["last_action"]))
        )
        lateral_vel_penalty = (
            self.penalty_lateral_vel
            * jp.square(lateral_velocity)
        )
        lateral_position_penalty = (
            self.penalty_lateral_position
            * jp.square(lateral_position)
        )
        tilt_penalty = (
            self.penalty_tilt
            * jp.square(tilt_error)
        )
        yaw_rate_penalty = (
            self.penalty_yaw_rate
            * jp.square(yaw_rate)
        )
        backward_penalty = (
            self.penalty_backward
            * jp.maximum(-forward_velocity, 0.0)
        )
        stall_penalty = self.penalty_stall * is_stalled
        failure_penalty = self.penalty_failure * failure

        arm_sync_penalty = (
            self.penalty_arm_sync
            * jp.square(arm_sync_error)
        )
        arm_limit_excess = jp.maximum(
            arm_limit_fraction - self.arm_limit_soft_fraction,
            0.0,
        )

        arm_limit_penalty = (
            self.penalty_arm_limit
            * jp.square(arm_limit_excess)
        )
        
        reach_event = (
            reached_step
            * (1.0 - info["reached_step"])
        )

        cross_event = (
            crossed_step
            * (1.0 - info["crossed_step"])
        )

        reach_step_bonus = (
            self.reward_reach_step
            * reach_event
        )

        cross_step_bonus = (
            self.reward_cross_step
            * cross_event
        )

        upright_reward = (
            self.reward_upright
            * 0.5
            * (
                jp.clip(up_z, -1.0, 1.0)
                + 1.0
            )
        )
        arm_q = jp.array([right_arm_angle, left_arm_angle])

        upright_quality = jp.clip(
            (up_z - self.neutral_posture_min_up_z)
            / (1.0 - self.neutral_posture_min_up_z),
            0.0,
            1.0,
        )

        neutral_quality = jp.exp(
            -jp.mean(
                jp.square(
                    arm_q / jp.deg2rad(
                        self.neutral_posture_arm_scale_deg
                    )
                )
            )
        )

        neutral_posture_reward = (
            self.reward_neutral_posture
            * upright_quality
            * neutral_quality
        )

        stable_after_crossing = (
            (crossed_step > 0.5)
            & (up_z > self.success_min_up_z)
        ).astype(jp.float32)

        stabilize_event = (
            stable_after_crossing
            * (1.0 - info["stabilized_after_crossing"])
        )

        stabilize_bonus = (
            self.reward_stabilize_after_crossing
            * stabilize_event
        )
        
        post_cross_gate = crossed_step

        post_cross_tilt_penalty = (
            post_cross_gate
            * self.penalty_post_cross_tilt
            * jp.square(
                jp.maximum(
                    self.success_min_up_z - up_z,
                    0.0,
                )
            )
        )
        
        reward = (
            progress_reward
            + forward_vel_reward
            + survival_reward
            + recovery_reward
            + upright_reward
            + cross_step_bonus
            + stabilize_bonus
            + reach_step_bonus
            + neutral_posture_reward
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
            - post_cross_tilt_penalty
            - arm_command_penalty
            - arm_velocity_penalty
            - arm_posture_penalty
        )
        
#         info = dict(State.info)

#         info["reached_step"] = jp.maximum(
#             State.info["reached_step"],
#             reached_step,
#         )
#         info["crossed_step"] = jp.maximum(
#             State.info["crossed_step"],
#             crossed_step,
#         )

        metrics = {
            "reward": reward,
            "reward/total": reward,
            "reward/progress": progress_reward,
            "reward/forward_vel": forward_vel_reward,
            "reward/survival": survival_reward,
            "reward/recovery": recovery_reward,
            "reward/success_bonus": success_bonus,
            "reward/reach_step": reach_step_bonus,
            "reward/cross_step": cross_step_bonus,
            "reward/upright": upright_reward,
            "reward/neutral_posture": neutral_posture_reward,
            "reward/stabilize": stabilize_bonus,
            "task/reached_step": reached_step,
            "task/crossed_step": crossed_step,
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
            "penalty/post_cross_tilt": post_cross_tilt_penalty,
            "penalty/arm_command": arm_command_penalty,
            "penalty/arm_velocity": arm_velocity_penalty,
            "penalty/arm_posture": arm_posture_penalty,
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

        return reward, success, failure, stall_steps, reached_step, crossed_step, stable_after_crossing,  upright_steps,metrics

    def _zero_metrics(self) -> dict[str, JaxArray]:
        z = jp.array(0.0, dtype=jp.float32)
        return {
            "gate/enabled_s": z,
            "gate/held_s": z,
            "gate/recovered_s": z,
            "gate/blocked_s": z,
            "gate/assist_starts": z,
            "gate/close_events": z,
            "gate/open_events": z,
            "gate/held_requested_abs_action_s": z,
            "gate/held_applied_abs_action_s": z,
            "gate/held_arm_displacement_rad": z,
            "reward": z,
            "reward/total": z,
            "reward/progress": z,
            "reward/forward_vel": z,
            "reward/survival": z,
            "reward/recovery": z,
            "reward/stabilize": z,
            "reward/neutral_posture": z,
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
            "penalty/post_cross_tilt": z,
            "penalty/arm_command": z,
            "penalty/arm_velocity": z,
            "penalty/arm_posture": z,
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
            "reward/reach_step": z,
            "reward/cross_step": z,
            "reward/upright": z,

            "task/reached_step": z,
            "task/crossed_step": z,
            
            "diagnostic/control_steps": z,
            "diagnostic/wheel_saturation_steps": z,
            "diagnostic/arm_saturation_steps": z,
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