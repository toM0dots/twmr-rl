from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Sequence

import flax
from flax import linen as nn
import jax
import jax.numpy as jp

from brax.training import distribution
from brax.training import networks
from brax.training import types


# ============================================================
# Observation layouts shared with tombot_env.py
# ============================================================

# Actor observation: hardware-style and matched to the new XML.
#
# Layout:
#   0:4      body quaternion from body_quat
#   4:7      body gyro
#   7:10     body accelerometer
#   10:16    normalized signed motor-effort/current proxies
#             [rear-right wheel, rear-left wheel,
#              front-right wheel, front-left wheel,
#              front-right transform, front-left transform]
#   16:21    normalized wheel velocities
#             [rear-right, rear-left, front-right, front-left,
#              passive middle support]
#   21:23    normalized arm angles [front-right, front-left]
#   23:25    normalized arm velocities [front-right, front-left]
#   25:31    previous normalized 6D policy action
#
# Total = 31
ACTOR_OBS_SIZE = 31

QUAT_START = 0
GYRO_START = 4
ACCEL_START = 7
CURRENT_START = 10
WHEEL_VEL_START = 16
ARM_ANGLE_START = 21
ARM_VEL_START = 23
LAST_ACTION_START = 25

# Critic observation:
#   qpos:        14
#   qvel:        13
#   last_action:  6
#
# Total = 33
PRIV_OBS_SIZE = 33


# ============================================================
# Graph matching the new robot morphology
# ============================================================

# One body node, one passive wheel node, four driven wheel nodes,
# and two independently actuated transform-arm nodes.
HUB = 0
MIDDLE_SUPPORT = 1
REAR_RIGHT_WHEEL = 2
REAR_LEFT_WHEEL = 3
FRONT_RIGHT_WHEEL = 4
FRONT_LEFT_WHEEL = 5
FRONT_RIGHT_ARM = 6
FRONT_LEFT_ARM = 7

NUM_NODES = 8

DRIVE_WHEEL_NODES = jp.array(
    [REAR_RIGHT_WHEEL, REAR_LEFT_WHEEL, FRONT_RIGHT_WHEEL, FRONT_LEFT_WHEEL],
    dtype=jp.int32,
)
ARM_NODES = jp.array([FRONT_RIGHT_ARM, FRONT_LEFT_ARM], dtype=jp.int32)


EDGE_PAIRS = [
    # Chassis/hub connections.
    (HUB, MIDDLE_SUPPORT), (MIDDLE_SUPPORT, HUB),
    (HUB, REAR_RIGHT_WHEEL), (REAR_RIGHT_WHEEL, HUB),
    (HUB, REAR_LEFT_WHEEL), (REAR_LEFT_WHEEL, HUB),
    (HUB, FRONT_RIGHT_WHEEL), (FRONT_RIGHT_WHEEL, HUB),
    (HUB, FRONT_LEFT_WHEEL), (FRONT_LEFT_WHEEL, HUB),
    (HUB, FRONT_RIGHT_ARM), (FRONT_RIGHT_ARM, HUB),
    (HUB, FRONT_LEFT_ARM), (FRONT_LEFT_ARM, HUB),

    # Same-side front/rear wheel coordination.
    (REAR_RIGHT_WHEEL, FRONT_RIGHT_WHEEL),
    (FRONT_RIGHT_WHEEL, REAR_RIGHT_WHEEL),
    (REAR_LEFT_WHEEL, FRONT_LEFT_WHEEL),
    (FRONT_LEFT_WHEEL, REAR_LEFT_WHEEL),

    # Left/right axle coordination.
    (REAR_RIGHT_WHEEL, REAR_LEFT_WHEEL),
    (REAR_LEFT_WHEEL, REAR_RIGHT_WHEEL),
    (FRONT_RIGHT_WHEEL, FRONT_LEFT_WHEEL),
    (FRONT_LEFT_WHEEL, FRONT_RIGHT_WHEEL),

    # Each transform arm carries its corresponding front wheel.
    (FRONT_RIGHT_ARM, FRONT_RIGHT_WHEEL),
    (FRONT_RIGHT_WHEEL, FRONT_RIGHT_ARM),
    (FRONT_LEFT_ARM, FRONT_LEFT_WHEEL),
    (FRONT_LEFT_WHEEL, FRONT_LEFT_ARM),

    # Bilateral transform synchronization.
    (FRONT_RIGHT_ARM, FRONT_LEFT_ARM),
    (FRONT_LEFT_ARM, FRONT_RIGHT_ARM),

    # The passive support wheel sits near the front hinge line.
    (MIDDLE_SUPPORT, FRONT_RIGHT_ARM),
    (FRONT_RIGHT_ARM, MIDDLE_SUPPORT),
    (MIDDLE_SUPPORT, FRONT_LEFT_ARM),
    (FRONT_LEFT_ARM, MIDDLE_SUPPORT),

    # Self loops.
    (HUB, HUB),
    (MIDDLE_SUPPORT, MIDDLE_SUPPORT),
    (REAR_RIGHT_WHEEL, REAR_RIGHT_WHEEL),
    (REAR_LEFT_WHEEL, REAR_LEFT_WHEEL),
    (FRONT_RIGHT_WHEEL, FRONT_RIGHT_WHEEL),
    (FRONT_LEFT_WHEEL, FRONT_LEFT_WHEEL),
    (FRONT_RIGHT_ARM, FRONT_RIGHT_ARM),
    (FRONT_LEFT_ARM, FRONT_LEFT_ARM),
]

SENDERS = jp.array([i for i, _ in EDGE_PAIRS], dtype=jp.int32)
RECEIVERS = jp.array([j for _, j in EDGE_PAIRS], dtype=jp.int32)
NUM_EDGES = SENDERS.shape[0]

# Abstract graph coordinates: x is left/right, y is front/rear.
# The XML's forward direction is -y.
NODE_POS = jp.array([
    [0.0, 0.0],    # HUB
    [0.0, -0.15],  # MIDDLE_SUPPORT
    [+1.0, +1.0],  # REAR_RIGHT_WHEEL
    [-1.0, +1.0],  # REAR_LEFT_WHEEL
    [+1.0, -1.0],  # FRONT_RIGHT_WHEEL
    [-1.0, -1.0],  # FRONT_LEFT_WHEEL
    [+0.8, -0.25], # FRONT_RIGHT_ARM
    [-0.8, -0.25], # FRONT_LEFT_ARM
], dtype=jp.float32)


def _unordered_pair(i: int, j: int, a: int, b: int) -> bool:
    return (i == a and j == b) or (i == b and j == a)


def _build_edge_attr() -> jp.ndarray:
    attrs = []

    for i, j in EDGE_PAIRS:
        dpos = NODE_POS[j] - NODE_POS[i]

        is_hub_edge = float((i == HUB or j == HUB) and i != j)
        is_same_side = float(
            _unordered_pair(i, j, REAR_RIGHT_WHEEL, FRONT_RIGHT_WHEEL)
            or _unordered_pair(i, j, REAR_LEFT_WHEEL, FRONT_LEFT_WHEEL)
        )
        is_axle_pair = float(
            _unordered_pair(i, j, REAR_RIGHT_WHEEL, REAR_LEFT_WHEEL)
            or _unordered_pair(i, j, FRONT_RIGHT_WHEEL, FRONT_LEFT_WHEEL)
        )
        is_arm_wheel = float(
            _unordered_pair(i, j, FRONT_RIGHT_ARM, FRONT_RIGHT_WHEEL)
            or _unordered_pair(i, j, FRONT_LEFT_ARM, FRONT_LEFT_WHEEL)
        )
        is_arm_pair = float(
            _unordered_pair(i, j, FRONT_RIGHT_ARM, FRONT_LEFT_ARM)
        )
        is_passive_edge = float(i == MIDDLE_SUPPORT or j == MIDDLE_SUPPORT)
        is_self = float(i == j)

        attrs.append(jp.array([
            dpos[0],
            dpos[1],
            is_hub_edge,
            is_same_side,
            is_axle_pair,
            is_arm_wheel,
            is_arm_pair,
            is_passive_edge,
            is_self,
        ], dtype=jp.float32))

    return jp.stack(attrs, axis=0)


EDGE_ATTR = _build_edge_attr()
RECV_ONEHOT = jax.nn.one_hot(RECEIVERS, NUM_NODES, dtype=jp.float32)
RECV_DEG = jp.maximum(jp.sum(RECV_ONEHOT, axis=0), 1.0)


# ============================================================
# Utility modules
# ============================================================

class MLP(nn.Module):
    widths: Sequence[int]
    activation: Any = nn.swish
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for i, width in enumerate(self.widths):
            x = nn.Dense(width)(x)
            if i < len(self.widths) - 1 or self.activate_final:
                x = self.activation(x)
        return x


class SmallActionHead(nn.Module):
    """Small-initialized scalar head, broadcast over any node axis."""

    hidden_dim: int
    activation: Any = nn.swish

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = self.activation(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = self.activation(x)
        x = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(x)
        return x


# ============================================================
# Actor observation parser
# ============================================================

def parse_actor_obs_to_nodes(obs: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray]:
    """Converts the 31D actor observation into eight graph nodes.

    Args:
      obs: shape [..., 31]

    Returns:
      node_features: shape [..., 8, node_dim]
      global_features: shape [..., 10], containing only IMU information
    """

    if obs.shape[-1] != ACTOR_OBS_SIZE:
        raise ValueError(
            f"Graph actor expects {ACTOR_OBS_SIZE}D actor obs, got {obs.shape[-1]}."
        )

    quat = obs[..., QUAT_START:QUAT_START + 4]
    gyro = obs[..., GYRO_START:GYRO_START + 3]
    accel = obs[..., ACCEL_START:ACCEL_START + 3]
    currents = obs[..., CURRENT_START:CURRENT_START + 6]
    wheel_vel = obs[..., WHEEL_VEL_START:WHEEL_VEL_START + 5]
    arm_angle = obs[..., ARM_ANGLE_START:ARM_ANGLE_START + 2]
    arm_vel = obs[..., ARM_VEL_START:ARM_VEL_START + 2]
    last_action = obs[..., LAST_ACTION_START:LAST_ACTION_START + 6]

    batch_shape = obs.shape[:-1]

    node_id = jp.eye(NUM_NODES, dtype=jp.float32)
    node_id = jp.broadcast_to(node_id, (*batch_shape, NUM_NODES, NUM_NODES))

    # Types: hub, passive wheel, driven wheel, transform arm.
    node_type = jp.array([
        [1, 0, 0, 0],  # HUB
        [0, 1, 0, 0],  # MIDDLE_SUPPORT
        [0, 0, 1, 0],  # REAR_RIGHT_WHEEL
        [0, 0, 1, 0],  # REAR_LEFT_WHEEL
        [0, 0, 1, 0],  # FRONT_RIGHT_WHEEL
        [0, 0, 1, 0],  # FRONT_LEFT_WHEEL
        [0, 0, 0, 1],  # FRONT_RIGHT_ARM
        [0, 0, 0, 1],  # FRONT_LEFT_ARM
    ], dtype=jp.float32)
    node_type = jp.broadcast_to(node_type, (*batch_shape, NUM_NODES, 4))

    # Role: [side, longitudinal position, actuated, front assembly].
    node_role = jp.array([
        [0.0, 0.0, 0.0, 0.0],   # HUB
        [0.0, -0.2, 0.0, 1.0],  # MIDDLE_SUPPORT
        [+1.0, +1.0, 1.0, 0.0], # REAR_RIGHT_WHEEL
        [-1.0, +1.0, 1.0, 0.0], # REAR_LEFT_WHEEL
        [+1.0, -1.0, 1.0, 1.0], # FRONT_RIGHT_WHEEL
        [-1.0, -1.0, 1.0, 1.0], # FRONT_LEFT_WHEEL
        [+1.0, -0.2, 1.0, 1.0], # FRONT_RIGHT_ARM
        [-1.0, -0.2, 1.0, 1.0], # FRONT_LEFT_ARM
    ], dtype=jp.float32)
    node_role = jp.broadcast_to(node_role, (*batch_shape, NUM_NODES, 4))

    imu_global = jp.concatenate([quat, gyro, accel], axis=-1)
    imu_tiled = jp.broadcast_to(
        imu_global[..., None, :],
        (*batch_shape, NUM_NODES, imu_global.shape[-1]),
    )

    z = jp.zeros_like(currents[..., 0:1])

    # Current/action order exactly matches the XML actuator order.
    node_current = jp.stack([
        z,                    # HUB
        z,                    # MIDDLE_SUPPORT
        currents[..., 0:1],  # REAR_RIGHT_WHEEL
        currents[..., 1:2],  # REAR_LEFT_WHEEL
        currents[..., 2:3],  # FRONT_RIGHT_WHEEL
        currents[..., 3:4],  # FRONT_LEFT_WHEEL
        currents[..., 4:5],  # FRONT_RIGHT_ARM
        currents[..., 5:6],  # FRONT_LEFT_ARM
    ], axis=-2)

    node_position = jp.stack([
        z, z, z, z, z, z,
        arm_angle[..., 0:1],
        arm_angle[..., 1:2],
    ], axis=-2)

    node_velocity = jp.stack([
        z,
        wheel_vel[..., 4:5],
        wheel_vel[..., 0:1],
        wheel_vel[..., 1:2],
        wheel_vel[..., 2:3],
        wheel_vel[..., 3:4],
        arm_vel[..., 0:1],
        arm_vel[..., 1:2],
    ], axis=-2)

    node_relevant_action = jp.stack([
        z,
        z,
        last_action[..., 0:1],
        last_action[..., 1:2],
        last_action[..., 2:3],
        last_action[..., 3:4],
        last_action[..., 4:5],
        last_action[..., 5:6],
    ], axis=-2)

    last_action_tiled = jp.broadcast_to(
        last_action[..., None, :],
        (*batch_shape, NUM_NODES, last_action.shape[-1]),
    )

    node_features = jp.concatenate([
        node_id,                 # 8
        node_type,               # 4
        node_role,               # 4
        imu_tiled,               # 10
        node_current,            # 1
        node_position,           # 1
        node_velocity,           # 1
        node_relevant_action,    # 1
        last_action_tiled,       # 6
    ], axis=-1)

    return node_features, imu_global


# ============================================================
# Message passing
# ============================================================

class MessagePassingLayer(nn.Module):
    hidden_dim: int
    activation: Any = nn.swish
    use_residual_gate: bool = True
    init_residual_alpha: float = 0.10

    @nn.compact
    def __call__(self, h_nodes: jp.ndarray, node_features: jp.ndarray) -> jp.ndarray:
        h_s = h_nodes[:, SENDERS, :]
        h_r = h_nodes[:, RECEIVERS, :]
        h_diff = h_s - h_r

        x_s = node_features[:, SENDERS, :]
        x_r = node_features[:, RECEIVERS, :]
        x_diff = x_s - x_r

        static_edge = jp.broadcast_to(
            EDGE_ATTR[None, :, :],
            (h_nodes.shape[0], NUM_EDGES, EDGE_ATTR.shape[-1]),
        )

        m_in = jp.concatenate([
            h_s,
            h_r,
            h_diff,
            x_s,
            x_r,
            x_diff,
            static_edge,
        ], axis=-1)

        messages = MLP(
            (self.hidden_dim, self.hidden_dim),
            activation=self.activation,
            activate_final=True,
        )(m_in)

        agg = jp.einsum("beh,en->bnh", messages, RECV_ONEHOT)
        agg = agg / RECV_DEG[None, :, None]

        u_in = jp.concatenate([h_nodes, agg, node_features], axis=-1)
        delta = MLP(
            (self.hidden_dim, self.hidden_dim),
            activation=self.activation,
            activate_final=True,
        )(u_in)

        if self.use_residual_gate:
            init_alpha = jp.clip(self.init_residual_alpha, 1e-4, 1.0 - 1e-4)
            init_logit = jp.log(init_alpha / (1.0 - init_alpha))
            logit_alpha = self.param(
                "logit_alpha",
                nn.initializers.constant(init_logit),
                (1,),
            )
            out = h_nodes + nn.sigmoid(logit_alpha) * delta
        else:
            out = h_nodes + delta

        return nn.LayerNorm()(out)


# ============================================================
# Graph actor: 31D observation -> six XML-matched controls
# ============================================================

class GraphActorModule(nn.Module):
    action_size: int
    hidden_dim: int = 64
    num_mp_layers: int = 1
    activation: Any = nn.swish

    noise_std_type: Literal["scalar", "log"] = "scalar"
    init_noise_std: float = 0.10
    min_noise_std: float = 0.03
    max_noise_std: float = 0.25

    use_residual_gate: bool = True
    init_residual_alpha: float = 0.10

    @nn.compact
    def __call__(self, obs_flat: jp.ndarray) -> jp.ndarray:
        if self.action_size != 6:
            raise ValueError(
                "GraphActorModule expects the six independent actuators in the new XML; "
                f"got action_size={self.action_size}."
            )

        node_features, global_features = parse_actor_obs_to_nodes(obs_flat)

        h = MLP(
            (self.hidden_dim, self.hidden_dim),
            activation=self.activation,
            activate_final=True,
        )(node_features)

        leading_shape = h.shape[:-2]
        hidden_size = h.shape[-1]
        h_flat = jp.reshape(h, (-1, NUM_NODES, hidden_size))
        x_flat = jp.reshape(
            node_features,
            (-1, NUM_NODES, node_features.shape[-1]),
        )

        for i in range(self.num_mp_layers):
            h_flat = MessagePassingLayer(
                hidden_dim=self.hidden_dim,
                activation=self.activation,
                use_residual_gate=self.use_residual_gate,
                init_residual_alpha=self.init_residual_alpha,
                name=f"message_passing_{i}",
            )(h_flat, x_flat)

        h = jp.reshape(h_flat, (*leading_shape, NUM_NODES, hidden_size))
        h_hub = h[..., HUB, :]

        # Four wheel outputs share one readout network, preserving left/right and
        # front/rear structural symmetry while still producing independent actions.
        h_wheels = h[..., DRIVE_WHEEL_NODES, :]
        hub_for_wheels = jp.broadcast_to(
            h_hub[..., None, :],
            (*h_wheels.shape[:-1], h_hub.shape[-1]),
        )

        zero_context = jp.zeros_like(h[..., REAR_RIGHT_WHEEL, :])
        attached_arm_context = jp.stack([
            zero_context,
            zero_context,
            h[..., FRONT_RIGHT_ARM, :],
            h[..., FRONT_LEFT_ARM, :],
        ], axis=-2)

        global_for_wheels = jp.broadcast_to(
            global_features[..., None, :],
            (*h_wheels.shape[:-1], global_features.shape[-1]),
        )

        wheel_readout = jp.concatenate([
            h_wheels,
            hub_for_wheels,
            attached_arm_context,
            global_for_wheels,
        ], axis=-1)
        wheel_mean = SmallActionHead(
            self.hidden_dim,
            self.activation,
            name="shared_wheel_action_head",
        )(wheel_readout)[..., 0]

        # Two transform outputs also share one readout network, but remain
        # independent actuators. This lets the policy correct left/right mismatch.
        h_arms = h[..., ARM_NODES, :]
        attached_wheels = jp.stack([
            h[..., FRONT_RIGHT_WHEEL, :],
            h[..., FRONT_LEFT_WHEEL, :],
        ], axis=-2)
        hub_for_arms = jp.broadcast_to(
            h_hub[..., None, :],
            (*h_arms.shape[:-1], h_hub.shape[-1]),
        )
        global_for_arms = jp.broadcast_to(
            global_features[..., None, :],
            (*h_arms.shape[:-1], global_features.shape[-1]),
        )

        arm_readout = jp.concatenate([
            h_arms,
            attached_wheels,
            hub_for_arms,
            global_for_arms,
        ], axis=-1)
        arm_mean = SmallActionHead(
            self.hidden_dim,
            self.activation,
            name="shared_transform_action_head",
        )(arm_readout)[..., 0]

        # Exact XML actuator order:
        #   rear_right_drive, rear_left_drive,
        #   front_right_drive, front_left_drive,
        #   front_right_transform, front_left_transform.
        mean = jp.concatenate([wheel_mean, arm_mean], axis=-1)

        init_log_std = jp.log(self.init_noise_std)
        if self.noise_std_type == "scalar":
            log_std = self.param(
                "log_std",
                nn.initializers.constant(init_log_std),
                (1,),
            )
            log_std = jp.broadcast_to(log_std, mean.shape)
        elif self.noise_std_type == "log":
            log_std = self.param(
                "log_std",
                nn.initializers.constant(init_log_std),
                (self.action_size,),
            )
            log_std = jp.broadcast_to(log_std, mean.shape)
        else:
            raise ValueError(f"Unsupported noise_std_type={self.noise_std_type}")

        log_std = jp.clip(
            log_std,
            jp.log(self.min_noise_std),
            jp.log(self.max_noise_std),
        )
        return jp.concatenate([mean, log_std], axis=-1)


# ============================================================
# Privileged critic: qpos/qvel + previous action
# ============================================================

class PrivilegedValueModule(nn.Module):
    hidden_layer_sizes: Sequence[int] = (256, 256, 256)
    activation: Any = nn.swish

    @nn.compact
    def __call__(self, obs_flat: jp.ndarray) -> jp.ndarray:
        if obs_flat.shape[-1] != PRIV_OBS_SIZE:
            raise ValueError(
                f"Privileged critic expects {PRIV_OBS_SIZE}D privileged obs, "
                f"got {obs_flat.shape[-1]}."
            )

        value = MLP(
            (*self.hidden_layer_sizes, 1),
            activation=self.activation,
            activate_final=False,
        )(obs_flat)
        return jp.squeeze(value, axis=-1)


# ============================================================
# Brax PPO wrappers
# ============================================================

def _make_ffn(
    module: nn.Module,
    observation_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn,
    obs_key: str,
):
    del preprocess_observations_fn

    def _dummy_from_obs_size(obs_size):
        if isinstance(obs_size, Mapping):
            obs_size = obs_size[obs_key]

        if isinstance(obs_size, int):
            shape = (1, obs_size)
        else:
            shape = (1, *tuple(obs_size))
        return jp.zeros(shape, dtype=jp.float32)

    def init(key):
        return module.init(key, _dummy_from_obs_size(observation_size))

    def apply(normalizer_params, params, observations):
        # The actor observation is normalized explicitly by the environment.
        # The critic uses modestly scaled simulator state and the same wrapper.
        del normalizer_params
        x = observations[obs_key] if isinstance(observations, Mapping) else observations
        return module.apply(params, x)

    return networks.FeedForwardNetwork(init=init, apply=apply)


@flax.struct.dataclass
class PPONetworks:
    policy_network: networks.FeedForwardNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
    activation: networks.ActivationFn = nn.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "privileged_state",
    distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    noise_std_type: Literal["scalar", "log"] = "scalar",
    init_noise_std: float = 0.10,
    state_dependent_std: bool = False,
    **kwargs: Any,
) -> PPONetworks:
    del policy_hidden_layer_sizes
    del state_dependent_std

    if action_size != 6:
        raise ValueError(
            "The new robot has six independently controlled XML actuators; "
            f"got action_size={action_size}."
        )

    if distribution_type == "normal":
        action_distribution = distribution.NormalDistribution(event_size=action_size)
    elif distribution_type == "tanh_normal":
        action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    else:
        raise ValueError(f"Unsupported distribution_type={distribution_type}")

    if action_distribution.param_size != 2 * action_size:
        raise ValueError(
            f"Expected distribution param_size={2 * action_size}, "
            f"got {action_distribution.param_size}."
        )

    hidden_dim = kwargs.pop("hidden_dim", 64)
    num_mp_layers = kwargs.pop("num_mp_layers", 1)
    min_noise_std = kwargs.pop("min_noise_std", 0.03)
    max_noise_std = kwargs.pop("max_noise_std", 0.25)
    use_residual_gate = kwargs.pop("use_residual_gate", True)
    init_residual_alpha = kwargs.pop("init_residual_alpha", 0.10)

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown GNN keyword arguments: {unknown}")

    policy_module = GraphActorModule(
        action_size=action_size,
        hidden_dim=hidden_dim,
        num_mp_layers=num_mp_layers,
        activation=activation,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        min_noise_std=min_noise_std,
        max_noise_std=max_noise_std,
        use_residual_gate=use_residual_gate,
        init_residual_alpha=init_residual_alpha,
    )

    value_module = PrivilegedValueModule(
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
    )

    return PPONetworks(
        policy_network=_make_ffn(
            policy_module,
            observation_size,
            preprocess_observations_fn,
            policy_obs_key,
        ),
        value_network=_make_ffn(
            value_module,
            observation_size,
            preprocess_observations_fn,
            value_obs_key,
        ),
        parametric_action_distribution=action_distribution,
    )
