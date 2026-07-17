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
# Observation layouts
# ============================================================

# Actor observation: hardware-style, deployable.
#
# Layout:
#   0:4      base orientation quaternion / AHRS estimate
#   4:7      gyro angular velocity
#   7:10     accelerometer
#   10:15    motor current proxies
#             [front wheel, rear wheel, upper wheel,
#              upper swing, lower fold]
#   15:17    normalized targets
#             [upper_target / upper_max, fold_target / fold_max]
#   17:22    last 5D logical action
#
# Total = 22
ACTOR_OBS_SIZE = 22

QUAT_START = 0
GYRO_START = 4
ACCEL_START = 7
CURRENT_START = 10
TARGET_START = 15
LAST_ACTION_START = 17

# Critic observation: privileged simulator state.
#
# Layout:
#   qpos:        13
#   qvel:        12
#   targets:      2
#   last_action:  5
#
# Total = 32
PRIV_OBS_SIZE = 32


# ============================================================
# Graph constants for IMU/current actor
# ============================================================

# The graph is no longer "body + four modules".
# It is now a sensor/motor graph:
#
#   HUB node:
#       IMU information
#
#   FRONT_WHEEL node:
#       front wheel current
#
#   REAR_WHEEL node:
#       rear wheel current
#
#   UPPER_WHEEL node:
#       upper wheel current
#
#   UPPER_ARM node:
#       upper swing motor current and upper target
#
#   LOWER_FOLD node:
#       logical lower fold current and fold target
#
# This is intentionally hardware-oriented. The actor does not see qpos/qvel.

HUB = 0
FRONT_WHEEL = 1
REAR_WHEEL = 2
UPPER_WHEEL = 3
UPPER_ARM = 4
LOWER_FOLD = 5

NUM_NODES = 6

EDGE_PAIRS = [
    # Hub-to-actuator star.
    (HUB, FRONT_WHEEL), (FRONT_WHEEL, HUB),
    (HUB, REAR_WHEEL), (REAR_WHEEL, HUB),
    (HUB, UPPER_WHEEL), (UPPER_WHEEL, HUB),
    (HUB, UPPER_ARM), (UPPER_ARM, HUB),
    (HUB, LOWER_FOLD), (LOWER_FOLD, HUB),

    # Lower wheel coordination.
    (FRONT_WHEEL, REAR_WHEEL), (REAR_WHEEL, FRONT_WHEEL),

    # Upper wheel coordinates with lower wheels for climbing/contact.
    (UPPER_WHEEL, FRONT_WHEEL), (FRONT_WHEEL, UPPER_WHEEL),
    (UPPER_WHEEL, REAR_WHEEL), (REAR_WHEEL, UPPER_WHEEL),

    # Shape coordination.
    (UPPER_ARM, UPPER_WHEEL), (UPPER_WHEEL, UPPER_ARM),
    (LOWER_FOLD, FRONT_WHEEL), (FRONT_WHEEL, LOWER_FOLD),
    (LOWER_FOLD, REAR_WHEEL), (REAR_WHEEL, LOWER_FOLD),
    (LOWER_FOLD, UPPER_ARM), (UPPER_ARM, LOWER_FOLD),

    # Self loops.
    (HUB, HUB),
    (FRONT_WHEEL, FRONT_WHEEL),
    (REAR_WHEEL, REAR_WHEEL),
    (UPPER_WHEEL, UPPER_WHEEL),
    (UPPER_ARM, UPPER_ARM),
    (LOWER_FOLD, LOWER_FOLD),
]

SENDERS = jp.array([i for i, _ in EDGE_PAIRS], dtype=jp.int32)
RECEIVERS = jp.array([j for _, j in EDGE_PAIRS], dtype=jp.int32)
NUM_EDGES = SENDERS.shape[0]


# Abstract graph coordinates, not physical meters.
# Used only to give the message function a fixed relational cue.
NODE_POS = jp.array([
    [0.0, 0.0],    # HUB
    [1.0, -0.4],   # FRONT_WHEEL
    [-1.0, 0.4],   # REAR_WHEEL
    [0.7, 0.5],    # UPPER_WHEEL
    [0.35, 0.5],   # UPPER_ARM
    [0.0, -0.6],   # LOWER_FOLD
], dtype=jp.float32)


def _build_edge_attr() -> jp.ndarray:
    attrs = []

    for i, j in EDGE_PAIRS:
        dpos = NODE_POS[j] - NODE_POS[i]

        is_hub_edge = 1.0 if (i == HUB or j == HUB) and i != j else 0.0

        is_lower_pair = 1.0 if (
            {i, j} == {FRONT_WHEEL, REAR_WHEEL}
        ) else 0.0

        is_upper_lower_pair = 1.0 if (
            {i, j} == {UPPER_WHEEL, FRONT_WHEEL}
            or {i, j} == {UPPER_WHEEL, REAR_WHEEL}
        ) else 0.0

        is_shape_edge = 1.0 if (
            {i, j} == {UPPER_ARM, UPPER_WHEEL}
            or {i, j} == {LOWER_FOLD, FRONT_WHEEL}
            or {i, j} == {LOWER_FOLD, REAR_WHEEL}
            or {i, j} == {LOWER_FOLD, UPPER_ARM}
        ) else 0.0

        is_self = 1.0 if i == j else 0.0

        attrs.append(jp.array([
            dpos[0],
            dpos[1],
            is_hub_edge,
            is_lower_pair,
            is_upper_lower_pair,
            is_shape_edge,
            is_self,
        ], dtype=jp.float32))

    return jp.stack(attrs, axis=0)


EDGE_ATTR = _build_edge_attr()
RECV_ONEHOT = jax.nn.one_hot(RECEIVERS, NUM_NODES, dtype=jp.float32)
RECV_DEG = jp.maximum(jp.sum(RECV_ONEHOT, axis=0), 1.0)


# ============================================================
# Utility MLPs
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
    """Small-initialized action head.

    This is important for GNN PPO stability. Without small final
    initialization, the graph policy can start with saturated actions.
    """
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
# Actor observation parser: 22D IMU/current -> graph nodes
# ============================================================

def parse_actor_obs_to_nodes(obs: jp.ndarray) -> tuple[jp.ndarray, jp.ndarray]:
    """Parses 22D hardware-style actor observation into graph node features.

    Args:
      obs: shape [..., 22]

    Returns:
      node_features: shape [..., 6, node_dim]
      global_features: shape [..., global_dim]
    """

    if obs.shape[-1] != ACTOR_OBS_SIZE:
        raise ValueError(
            f"Graph actor expects {ACTOR_OBS_SIZE}D actor obs, got {obs.shape[-1]}."
        )

    quat = obs[..., QUAT_START:QUAT_START + 4]
    gyro = obs[..., GYRO_START:GYRO_START + 3]
    accel = obs[..., ACCEL_START:ACCEL_START + 3]
    currents = obs[..., CURRENT_START:CURRENT_START + 5]
    targets = obs[..., TARGET_START:TARGET_START + 2]
    last_action = obs[..., LAST_ACTION_START:LAST_ACTION_START + 5]

    # Current layout:
    #   0 front wheel
    #   1 rear wheel
    #   2 upper wheel
    #   3 upper swing
    #   4 lower fold
    front_current = currents[..., 0:1]
    rear_current = currents[..., 1:2]
    upper_wheel_current = currents[..., 2:3]
    upper_arm_current = currents[..., 3:4]
    lower_fold_current = currents[..., 4:5]

    upper_target = targets[..., 0:1]
    fold_target = targets[..., 1:2]

    # Last action layout:
    #   0 lower common drive
    #   1 lower drive difference
    #   2 upper wheel drive
    #   3 delta upper arm target
    #   4 delta lower fold target
    lower_common = last_action[..., 0:1]
    lower_diff = last_action[..., 1:2]
    upper_wheel_action = last_action[..., 2:3]
    upper_delta = last_action[..., 3:4]
    fold_delta = last_action[..., 4:5]

    batch_shape = obs.shape[:-1]

    # Node identity one-hot: shape [..., 6, 6]
    node_id = jp.eye(NUM_NODES, dtype=jp.float32)
    node_id = jp.broadcast_to(node_id, (*batch_shape, NUM_NODES, NUM_NODES))

    # Node type one-hot:
    #   hub, wheel, upper_arm, lower_fold
    node_type = jp.array([
        [1, 0, 0, 0],  # HUB
        [0, 1, 0, 0],  # FRONT_WHEEL
        [0, 1, 0, 0],  # REAR_WHEEL
        [0, 1, 0, 0],  # UPPER_WHEEL
        [0, 0, 1, 0],  # UPPER_ARM
        [0, 0, 0, 1],  # LOWER_FOLD
    ], dtype=jp.float32)
    node_type = jp.broadcast_to(node_type, (*batch_shape, NUM_NODES, 4))

    # Abstract side/role features.
    # lower_side: front +1, rear -1, others 0
    # upper_role: upper wheel/arm +1, others 0
    role = jp.array([
        [0.0, 0.0],   # HUB
        [+1.0, 0.0],  # FRONT_WHEEL
        [-1.0, 0.0],  # REAR_WHEEL
        [0.0, +1.0],  # UPPER_WHEEL
        [0.0, +1.0],  # UPPER_ARM
        [0.0, 0.0],   # LOWER_FOLD
    ], dtype=jp.float32)
    role = jp.broadcast_to(role, (*batch_shape, NUM_NODES, 2))

    # IMU is global body information. We tile it to every node so that
    # every local motor node can condition on body attitude/acceleration.
    imu_global = jp.concatenate([quat, gyro, accel], axis=-1)  # [..., 10]
    imu_tiled = jp.broadcast_to(
        imu_global[..., None, :],
        (*batch_shape, NUM_NODES, imu_global.shape[-1]),
    )

    # Per-node motor current.
    node_current = jp.stack([
        jp.zeros_like(front_current),   # HUB
        front_current,
        rear_current,
        upper_wheel_current,
        upper_arm_current,
        lower_fold_current,
    ], axis=-2)

    # Per-node target. Only upper arm and lower fold have meaningful targets.
    node_target = jp.stack([
        jp.zeros_like(upper_target),  # HUB
        jp.zeros_like(upper_target),  # FRONT_WHEEL
        jp.zeros_like(upper_target),  # REAR_WHEEL
        jp.zeros_like(upper_target),  # UPPER_WHEEL
        upper_target,                 # UPPER_ARM
        fold_target,                  # LOWER_FOLD
    ], axis=-2)

    # Per-node relevant previous action.
    #
    # For lower wheels, convert logical action into approximate wheel command
    # direction:
    #   front command ~= lower_common - lower_diff
    #   rear command  ~= lower_common + lower_diff
    front_last_cmd = lower_common - lower_diff
    rear_last_cmd = lower_common + lower_diff

    node_relevant_action = jp.stack([
        jp.zeros_like(lower_common),  # HUB
        front_last_cmd,               # FRONT_WHEEL
        rear_last_cmd,                # REAR_WHEEL
        upper_wheel_action,           # UPPER_WHEEL
        upper_delta,                  # UPPER_ARM
        fold_delta,                   # LOWER_FOLD
    ], axis=-2)

    # Also give every node the whole previous logical action.
    last_action_tiled = jp.broadcast_to(
        last_action[..., None, :],
        (*batch_shape, NUM_NODES, last_action.shape[-1]),
    )

    node_features = jp.concatenate([
        node_id,                # 6
        node_type,              # 4
        role,                   # 2
        imu_tiled,              # 10
        node_current,           # 1
        node_target,            # 1
        node_relevant_action,   # 1
        last_action_tiled,      # 5
    ], axis=-1)

    global_features = jp.concatenate([
        quat,
        gyro,
        accel,
        currents,
        targets,
        last_action,
    ], axis=-1)

    return node_features, global_features


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
        """One graph message-passing layer.

        Args:
          h_nodes: shape [Bflat, N, H]
          node_features: shape [Bflat, N, F]

        Returns:
          updated hidden nodes, shape [Bflat, N, H]
        """

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
            # alpha initialized around init_residual_alpha.
            init_alpha = jp.clip(self.init_residual_alpha, 1e-4, 1.0 - 1e-4)
            init_logit = jp.log(init_alpha / (1.0 - init_alpha))
            logit_alpha = self.param(
                "logit_alpha",
                nn.initializers.constant(init_logit),
                (1,),
            )
            alpha = nn.sigmoid(logit_alpha)
            out = h_nodes + alpha * delta
        else:
            out = h_nodes + delta

        out = nn.LayerNorm()(out)
        return out


# ============================================================
# Graph actor: 22D IMU/current -> 5D action distribution
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
        if self.action_size != 5:
            raise ValueError(
                f"GraphActorModule expects action_size=5 for the logical robot action, "
                f"got {self.action_size}."
            )

        node_features, global_features = parse_actor_obs_to_nodes(obs_flat)

        # Node encoder.
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
        h_front = h[..., FRONT_WHEEL, :]
        h_rear = h[..., REAR_WHEEL, :]
        h_upper_wheel = h[..., UPPER_WHEEL, :]
        h_upper_arm = h[..., UPPER_ARM, :]
        h_lower_fold = h[..., LOWER_FOLD, :]

        # Mechanism-aware action readout.
        #
        # action[0]: lower common wheel drive
        # action[1]: lower front/rear drive difference
        # action[2]: upper wheel drive
        # action[3]: delta upper arm target
        # action[4]: delta lower fold target

        lower_common_in = jp.concatenate([
            h_hub,
            h_front + h_rear,
            h_lower_fold,
            global_features,
        ], axis=-1)

        lower_diff_in = jp.concatenate([
            h_front - h_rear,
            h_hub,
            h_lower_fold,
            global_features,
        ], axis=-1)

        upper_wheel_in = jp.concatenate([
            h_upper_wheel,
            h_upper_arm,
            h_hub,
            global_features,
        ], axis=-1)

        upper_delta_in = jp.concatenate([
            h_upper_arm,
            h_upper_wheel,
            h_hub,
            global_features,
        ], axis=-1)

        fold_delta_in = jp.concatenate([
            h_lower_fold,
            h_front + h_rear,
            h_hub,
            h_upper_arm,
            global_features,
        ], axis=-1)

        lower_common = SmallActionHead(self.hidden_dim, self.activation)(lower_common_in)
        lower_diff = SmallActionHead(self.hidden_dim, self.activation)(lower_diff_in)
        upper_wheel = SmallActionHead(self.hidden_dim, self.activation)(upper_wheel_in)
        upper_delta = SmallActionHead(self.hidden_dim, self.activation)(upper_delta_in)
        fold_delta = SmallActionHead(self.hidden_dim, self.activation)(fold_delta_in)

        mean = jp.concatenate([
            lower_common,
            lower_diff,
            upper_wheel,
            upper_delta,
            fold_delta,
        ], axis=-1)

        # Trainable log std, clipped for hardware/current-based control.
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

        min_log_std = jp.log(self.min_noise_std)
        max_log_std = jp.log(self.max_noise_std)
        log_std = jp.clip(log_std, min_log_std, max_log_std)

        return jp.concatenate([mean, log_std], axis=-1)


# ============================================================
# Privileged critic: 32D qpos/qvel state -> scalar value
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

        v = MLP(
            (*self.hidden_layer_sizes, 1),
            activation=self.activation,
            activate_final=False,
        )(obs_flat)

        return jp.squeeze(v, axis=-1)


# ============================================================
# FeedForwardNetwork wrappers for Brax PPO
# ============================================================

def _make_ffn(
    module: nn.Module,
    observation_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn,
    obs_key: str,
):
    """Wraps a Flax module into brax.training.networks.FeedForwardNetwork."""

    def _dummy_from_obs_size(obs_size):
        # obs_size can be:
        #   22
        #   (22,)
        #   {"state": (22,), "privileged_state": (32,)}
        if isinstance(obs_size, Mapping):
            obs_size = obs_size[obs_key]

        if isinstance(obs_size, int):
            shape = (1, obs_size)
        else:
            shape = (1, *tuple(obs_size))

        return jp.zeros(shape, dtype=jp.float32)

    def init(key):
        dummy_obs = _dummy_from_obs_size(observation_size)
        return module.init(key, dummy_obs)

    def apply(normalizer_params, params, observations):
        # We intentionally ignore Brax's generic observation normalizer here.
        # Actor obs is already manually normalized in the environment.
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
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,  # kept for API compatibility
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
    activation: networks.ActivationFn = nn.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "privileged_state",
    distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    noise_std_type: Literal["scalar", "log"] = "scalar",
    init_noise_std: float = 0.10,
    state_dependent_std: bool = False,  # unused
    **kwargs: Any,
) -> PPONetworks:
    del policy_hidden_layer_sizes
    del state_dependent_std

    if action_size != 5:
        raise ValueError(
            f"This network is written for the 5D logical action space, got {action_size}."
        )

    if distribution_type == "normal":
        pad = distribution.NormalDistribution(event_size=action_size)
    elif distribution_type == "tanh_normal":
        pad = distribution.NormalTanhDistribution(event_size=action_size)
    else:
        raise ValueError(f"Unsupported distribution_type={distribution_type}")

    if pad.param_size != 2 * action_size:
        raise ValueError(
            f"Expected distribution param_size={2 * action_size}, got {pad.param_size}."
        )

    hidden_dim = kwargs.pop("hidden_dim", 64)
    num_mp_layers = kwargs.pop("num_mp_layers", 1)
    min_noise_std = kwargs.pop("min_noise_std", 0.03)
    max_noise_std = kwargs.pop("max_noise_std", 0.25)
    use_residual_gate = kwargs.pop("use_residual_gate", True)
    init_residual_alpha = kwargs.pop("init_residual_alpha", 0.10)

    # Actor sees only deployable hardware-style observation.
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

    # Critic sees privileged full simulator state.
    value_module = PrivilegedValueModule(
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
    )

    policy_net = _make_ffn(
        policy_module,
        observation_size,
        preprocess_observations_fn,
        policy_obs_key,
    )

    value_net = _make_ffn(
        value_module,
        observation_size,
        preprocess_observations_fn,
        value_obs_key,
    )

    return PPONetworks(
        policy_network=policy_net,
        value_network=value_net,
        parametric_action_distribution=pad,
    )