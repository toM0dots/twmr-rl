from pathlib import Path
from typing import Any

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
_XML_PATH = Path(__file__).parent.parent.parent / "assets" / "trans_wheel_robo2_2BOX.xml" 


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
    cfg.success_x = 1.5
    cfg.success_min_up_z = 0.01

    cfg.min_root_z = 0.10
    cfg.failure_min_root_z = -0.1
    cfg.failure_min_up_z = 0.4
    cfg.max_stall_steps = 150
    cfg.ext_stall_steps = 80
    cfg.max_vel = 10
    cfg.failure_back_x = -0.05

    # -------------------------
    # obstacle region
    # -------------------------
    cfg.obstacle_x_min = 1.0
    cfg.obstacle_x_max = 1.4
    cfg.obstacle_margin = 0.001

    # -------------------------
    # reward scales
    # -------------------------
    cfg.reward_progress = 5.0
    cfg.reward_survival = 0.1
    cfg.reward_forward_vel = 1.0
    cfg.reward_success = 400.0

    # -------------------------
    # penalties
    # -------------------------
    cfg.penalty_control = 0.01
    cfg.penalty_lateral = 0.15
    cfg.penalty_tilt = 0.2
    cfg.penalty_ext = 0.08
    cfg.penalty_stall = 0.1 # 0.005
    cfg.penalty_failure = 40.0
    cfg.backward_penalty = 3

    # -------------------------
    # misc
    # -------------------------
    cfg.stall_delta_threshold = 0.0001
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
        
        self.nu = int(self._mj_model.nu)   # should be 8

        self.x_vel_idx = 0
        self.y_vel_idx = 1

        # wheel joints only
        self.wheel_qvel_idx = jp.array([6, 10, 14, 18], dtype=jp.int32)
        self.wheel_qpos_idx = jp.array([7, 11, 15, 19], dtype=jp.int32)

        # ext0 joints only (the actuated extension joints)
        self.leg_qpos_idx = jp.array([8, 12, 16, 20], dtype=jp.int32)
        self.leg_qvel_idx = jp.array([7, 11, 15, 19], dtype=jp.int32)

        self.leg_center     = (-1.047 + 3.427) / 2
        self.leg_half_range = (3.427 - (-1.047)) / 2
        self.leg_max = 3.427
        self.leg_min = -1.047

        self.torque_limit = 0.6
        self.action_scale = jp.array([0.8] * 4 + [0.6] * 4)
        self.kp = jp.ones(8)                           # tune later
        
        self._reward_config = reward_configs()


        # TODO: figure out vision with the madrona batch renderer

        # TODO: what does this do for us exactly?
        # self._root_body_id = self._mj_model.body("root").id

    def reset(self, rng: JaxArray) -> State:
        # TODO: randomize initial state (qpos, qvel)
        # qpos = qpos.at[2].set(0.2)
#         qpos = qpos + 0.01 * jax.random.normal(rng_init, qpos.shape)

        # Initially reset to the original position
#         qpos = jp.zeros(self.mjx_model.nq)
#         qvel = jp.zeros(self.mjx_model.nv)


        qpos = self.mjx_model.qpos0
        qvel = jp.zeros(self.mjx_model.nv)

        qpos = qpos.at[self.wheel_qpos_idx].set(0.0)   # [7, 11, 15, 19]
        qpos = qpos.at[self.leg_qpos_idx].set(0.0) 

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.naconmax,  # type: ignore
            njmax=self._config.njmax,  # type: ignore
        )

        data = mjx.forward(self.mjx_model, data)
        
        
        data = data.replace(
            qvel=data.qvel.at[0].set(0.2)  # seed forward velocity
        )

        # TODO: initialize metrics to zero once we know what to track
        z = jp.array(0.0, dtype=jp.float32)
#         metrics = {}
        metrics = {
            "reward": z,
            "reward/progress_delta": z,
            "reward/survival": z,
            "reward/success_bonus": z,
            "reward/forward_vel_reward": z, 
            "penalty/control": z,
            "penalty/lateral": z,
            "penalty/tilt": z,
            "penalty/stall": z,
            "penalty/failure": z,
            "penalty/yaw_rate_penalty": z,
            "penalty/backward_penalty": z,
            "failure_back_x": z,
            "task/root_x": z,
            "task/root_vx": z,
            "task/delta_x": z,
            "task/ext_mean": z,
            "task/in_obstacle_local": z,
            "task/success": z,
            "task/failure": z,
            "reward/total": z,
            "task/stall_steps": z,        # add
            "task/failure_stall": z,
        }

        info = {"rng": rng}
        info["prev_root_x"] = data.qpos[0]
        info["success"] = jp.array(0.0, dtype=jp.float32)
        info["failure"] = jp.array(0.0, dtype=jp.float32)
        info["stall_steps"] = jp.array(0.0, dtype=jp.float32)
        info["last_action"] = jp.zeros((self.nu,), dtype=jp.float32)

        obs = self._get_obs(data, info)

        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: JaxArray) -> State:
        scaled_action = action * self.action_scale
#         scaled_action = jp.array([0., 0., 0., 0., 0.6, 0.6, 0, 0])
        data = mjx_env.step(self.mjx_model, state.data, scaled_action, self.n_substeps)

        root_x = data.qpos[0]
        delta_x = root_x - state.info["prev_root_x"]

        reward, success, failure, metrics, failure_stall, stall_steps = self._compute_reward(data, scaled_action, state.info)

        failure = jp.maximum(failure, failure_stall)

        done = jp.maximum(success, failure)

        info = dict(state.info)
        info["prev_root_x"] = root_x
        info["success"] = success
        info["failure"] = failure
        info["stall_steps"] = stall_steps
        info["last_action"] = action

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

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> JaxArray:
        # TODO: center of mass dynamics
        qpos = data.qpos
        # print(f"==>> qpos: {qpos}")
        qvel = data.qvel
        # print(f"==>> qvel: {qvel}")
        return jp.concatenate([qpos, qvel])
    

    def _compute_task_flags(self, data, info):
        root_x = data.qpos[0]
        root_z = data.qpos[2]
        root_quat = data.qpos[3:7]
        up_z = _quat_up_z(root_quat)

        has_nan = (
            (~jp.isfinite(data.qpos)).any()
            | (~jp.isfinite(data.qvel)).any()
        ).astype(jp.float32)

        success = (root_x >= self._reward_config.success_x).astype(jp.float32)
            
        """"
            & (up_z >= self._reward_config.success_min_up_z 
            & (root_z >= self._reward_config.min_root_z)
           )
        .astype(jp.float32)
         """
        delta_x = root_x - info["prev_root_x"]

        is_stalled = (jp.abs(delta_x) < self._reward_config.stall_delta_threshold).astype(jp.float32)
        stall_steps = jp.where(
            is_stalled > 0.5,
            info["stall_steps"] + 1.0,
            0.0,
        )
        failure_stall = (stall_steps >= self._reward_config.max_stall_steps).astype(jp.float32)
            

        failure_height = (root_z < self._reward_config.failure_min_root_z).astype(jp.float32)
        
        failure_back_x = (root_x < self._reward_config.failure_back_x).astype(jp.float32)
        failure_tilt = 0 # (up_z < self._reward_config.failure_min_up_z).astype(jp.float32)

        failure = jp.maximum( failure_back_x, jp.maximum(has_nan, jp.maximum(failure_stall, jp.maximum(failure_height, failure_tilt))))
        return success, failure, root_x, root_z, up_z, stall_steps, failure_stall, failure_back_x


    def _compute_reward(self, data, action, info):
        success, failure, root_x, root_z, up_z, stall_steps, failure_stall, failure_back_x  = self._compute_task_flags(data, info)

        root_vx = data.qvel[0]
        root_vy = data.qvel[1]
        root_wz = data.qvel[5]
        delta_x = root_x - info["prev_root_x"]

        # module extension state
        mod_qpos = jp.reshape(data.qpos[7:], (4, 4))
        ext_pos = mod_qpos[:, 1:4]
        ext_mean_per_mod = jp.mean(ext_pos, axis=-1)
        ext_mean_all = jp.mean(ext_mean_per_mod)

        # actions:
        # [FL wheel, FR wheel, RL wheel, RR wheel, FL ext0, FR ext0, RL ext0, RR ext0]
        ext_action = action[4:8]
        extend_use = jp.mean(jp.maximum(ext_action, 0.0))  

        # obstacle-local gate
        in_obstacle_local = (
            (root_x >= self._reward_config.obstacle_x_min - self._reward_config.obstacle_margin)
            & (root_x <= self._reward_config.obstacle_x_max + self._reward_config.obstacle_margin)
        ).astype(jp.float32)

        tilt_err = 1.0 - jp.clip(up_z, -1.0, 1.0)

        progress_delta_reward = self._reward_config.reward_progress * delta_x 
        forward_vel_reward = self._reward_config.reward_forward_vel * jp.clip(root_vx, 0, self._reward_config.max_vel)
        survival_reward = self._reward_config.reward_survival * (1.0 - failure)

        control_penalty = self._reward_config.penalty_control * jp.mean(jp.square(action))
        lateral_penalty = self._reward_config.penalty_lateral * jp.square(root_vy)
        tilt_penalty = self._reward_config.penalty_tilt * jp.square(tilt_err)
        yaw_rate_penalty = 0.5 * jp.square(root_wz)
        backward_penalty = self._reward_config.backward_penalty * jp.maximum(-root_vx, 0.0)
        
        ext_norm = jp.clip((ext_mean_all ) / self.leg_max, 0.0, 1.0)
        ext_reward = 1 * ext_norm * (info["stall_steps"] > self._reward_config.ext_stall_steps) * in_obstacle_local
        ext_penalty = self._reward_config.penalty_ext * ext_norm
        
        is_stalled = (jp.abs(delta_x) < self._reward_config.stall_delta_threshold).astype(jp.float32)
        
        stall_excess = jp.maximum(stall_steps - self._reward_config.ext_stall_steps, 0.0)
        stall_excess_capped = jp.minimum(stall_excess, 20.0)
        stall_penalty = ( self._reward_config.penalty_stall * is_stalled + self._reward_config.penalty_stall * stall_excess_capped )
        
        recovery_reward = 5 * jp.maximum(delta_x, 0.0) * ext_norm * (info["stall_steps"] > self._reward_config.ext_stall_steps )
        
        success_bonus = self._reward_config.reward_success * success
        success_in_region = 20 * in_obstacle_local
        failure_penalty = self._reward_config.penalty_failure * failure

        reward = (
            progress_delta_reward
            + survival_reward
            + forward_vel_reward
            + ext_reward
            + recovery_reward
            + success_bonus
            + success_in_region
            - control_penalty
            - lateral_penalty
            - tilt_penalty
            - stall_penalty
            - ext_penalty
            - failure_penalty
            - yaw_rate_penalty
            - backward_penalty
        )

        metrics = {
            "reward": reward,
            "reward/progress_delta": progress_delta_reward,
            "reward/survival": survival_reward,
            "reward/success_bonus": success_bonus,
            "reward/forward_vel_reward": forward_vel_reward, 
            "penalty/control": control_penalty,
            "penalty/lateral": lateral_penalty,
            "penalty/tilt": tilt_err,
            "penalty/stall": stall_penalty,
            "penalty/failure": failure_penalty,
            "penalty/yaw_rate_penalty": yaw_rate_penalty,
            "penalty/backward_penalty": backward_penalty, 
            "failure_back_x": failure_back_x, 
            "task/root_x": root_x,
            "task/root_vx": root_vx,
            "task/delta_x": delta_x,
            "task/ext_mean": ext_mean_all,
            "task/in_obstacle_local": in_obstacle_local,
            "task/success": success,
            "task/failure": failure,
            "reward/total": reward,
            "task/stall_steps": stall_steps,       
        }

        return reward, success, failure, metrics, failure_stall, stall_steps

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

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
