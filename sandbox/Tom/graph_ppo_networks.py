from typing import Any, Literal, Mapping, Sequence
import flax
from flax import linen as nn
import jax
import jax.numpy as jp

from brax.training import distribution
from brax.training import networks
from brax.training import types


# --------------------------
# Robot graph constants
# --------------------------
# --------------------------
# Robot graph constants: new 3-wheel nested-arm robot
# --------------------------

HUB = 0
FRONT_LEG = 1
REAR_LEG = 2
UPPER_ARM = 3
FRONT_WHEEL = 4
REAR_WHEEL = 5
UPPER_WHEEL = 6

NUM_NODES = 7

ACTOR_OBS_SIZE = 22

QUAT_START = 0
GYRO_START = 4
ACCEL_START = 7
CURRENT_START = 10
TARGET_START = 15
LAST_ACTION_START = 17

EDGE_PAIRS = [
    # hub-limb tree
    (HUB, FRONT_LEG), (FRONT_LEG, HUB),
    (HUB, REAR_LEG), (REAR_LEG, HUB),
    (HUB, UPPER_ARM), (UPPER_ARM, HUB),

    # limb-wheel edges
    (FRONT_LEG, FRONT_WHEEL), (FRONT_WHEEL, FRONT_LEG),
    (REAR_LEG, REAR_WHEEL), (REAR_WHEEL, REAR_LEG),
    (UPPER_ARM, UPPER_WHEEL), (UPPER_WHEEL, UPPER_ARM),

    # symmetry / coordination edges
    (FRONT_LEG, REAR_LEG), (REAR_LEG, FRONT_LEG),
    (FRONT_WHEEL, REAR_WHEEL), (REAR_WHEEL, FRONT_WHEEL),
    (UPPER_WHEEL, FRONT_WHEEL), (FRONT_WHEEL, UPPER_WHEEL),
    (UPPER_WHEEL, REAR_WHEEL), (REAR_WHEEL, UPPER_WHEEL),

    # self-loops
    (HUB, HUB),
    (FRONT_LEG, FRONT_LEG),
    (REAR_LEG, REAR_LEG),
    (UPPER_ARM, UPPER_ARM),
    (FRONT_WHEEL, FRONT_WHEEL),
    (REAR_WHEEL, REAR_WHEEL),
    (UPPER_WHEEL, UPPER_WHEEL),
]

SENDERS = jp.array([i for i, j in EDGE_PAIRS], dtype=jp.int32)
RECEIVERS = jp.array([j for i, j in EDGE_PAIRS], dtype=jp.int32)
NUM_EDGES = SENDERS.shape[0]

# Rough abstract graph coordinates, not physical meters.
NODE_XZ = jp.array([
    [0.0, 0.0],    # HUB
    [1.0, 0.0],    # FRONT_LEG
    [-1.0, 0.0],   # REAR_LEG
    [0.5, 0.5],    # UPPER_ARM
    [1.5, 0.0],    # FRONT_WHEEL
    [-1.5, 0.0],   # REAR_WHEEL
    [1.5, 0.5],    # UPPER_WHEEL
], dtype=jp.float32)


def _build_edge_attr():
    attrs = []
    for i, j in EDGE_PAIRS:
        dxz = NODE_XZ[j] - NODE_XZ[i]

        is_hub_limb = 1.0 if (
            (i == HUB and j in [FRONT_LEG, REAR_LEG, UPPER_ARM])
            or (j == HUB and i in [FRONT_LEG, REAR_LEG, UPPER_ARM])
        ) else 0.0

        is_limb_wheel = 1.0 if (
            (i == FRONT_LEG and j == FRONT_WHEEL)
            or (i == FRONT_WHEEL and j == FRONT_LEG)
            or (i == REAR_LEG and j == REAR_WHEEL)
            or (i == REAR_WHEEL and j == REAR_LEG)
            or (i == UPPER_ARM and j == UPPER_WHEEL)
            or (i == UPPER_WHEEL and j == UPPER_ARM)
        ) else 0.0

        is_lower_sym = 1.0 if (
            {i, j} == {FRONT_LEG, REAR_LEG}
            or {i, j} == {FRONT_WHEEL, REAR_WHEEL}
        ) else 0.0

        is_upper_lower = 1.0 if (
            {i, j} == {UPPER_WHEEL, FRONT_WHEEL}
            or {i, j} == {UPPER_WHEEL, REAR_WHEEL}
        ) else 0.0

        is_self = 1.0 if i == j else 0.0

        attrs.append(jp.array([
            dxz[0],
            dxz[1],
            is_hub_limb,
            is_limb_wheel,
            is_lower_sym,
            is_upper_lower,
            is_self,
        ], dtype=jp.float32))

    return jp.stack(attrs, axis=0)


EDGE_ATTR = _build_edge_attr()
RECV_ONEHOT = jax.nn.one_hot(RECEIVERS, NUM_NODES, dtype=jp.float32)
RECV_DEG = jp.maximum(jp.sum(RECV_ONEHOT, axis=0), 1.0)

# --------------------------
# obs -> (body,node) features
# --------------------------
# Observation layout for current XML/env.
NQ = 13
NV = 12
TARGET_START = NQ + NV
LAST_ACTION_START = TARGET_START + 2

LEG_LENGTH = 0.150
WHEEL_SPEED_SCALE = 45.0
JOINT_VEL_SCALE = 10.0
ANGLE_SCALE = 1.65
ROOT_VEL_SCALE = 2.0


def parse_obs_to_nodes(obs_batch: jp.ndarray):
    qpos = obs_batch[..., :NQ]
    qvel = obs_batch[..., NQ:NQ + NV]

    upper_target = obs_batch[..., TARGET_START:TARGET_START + 1]
    fold_target = obs_batch[..., TARGET_START + 1:TARGET_START + 2]
    last_action = obs_batch[..., LAST_ACTION_START:LAST_ACTION_START + 5]

    root_qpos = qpos[..., :7]
    root_qvel = qvel[..., :6]

    root_vx = root_qvel[..., 0:1] / ROOT_VEL_SCALE
    root_vz = root_qvel[..., 2:3] / ROOT_VEL_SCALE
    root_wy = root_qvel[..., 4:5] / ROOT_VEL_SCALE

    # qpos order after freejoint:
    # 7  front_lower_fold_hinge
    # 8  front_lower_wheel_hinge
    # 9  rear_lower_fold_hinge
    # 10 rear_lower_wheel_hinge
    # 11 upper_swing_hinge
    # 12 upper_wheel_hinge
    front_fold = qpos[..., 7:8]
    rear_fold = qpos[..., 9:10]
    upper_angle = qpos[..., 11:12]

    # qvel order after freejoint:
    # 6  front_fold_vel
    # 7  front_wheel_vel
    # 8  rear_fold_vel
    # 9  rear_wheel_vel
    # 10 upper_angle_vel
    # 11 upper_wheel_vel
    front_fold_vel = qvel[..., 6:7]
    front_wheel_vel = qvel[..., 7:8]
    rear_fold_vel = qvel[..., 8:9]
    rear_wheel_vel = qvel[..., 9:10]
    upper_vel = qvel[..., 10:11]
    upper_wheel_vel = qvel[..., 11:12]

    # Approximate relative positions from joint angles.
    # Convention: positive lower fold moves downward.
    # Front lower wheel: +x, -y side.
    front_wheel_x = LEG_LENGTH * jp.cos(front_fold)
    front_wheel_y = -0.062 * jp.ones_like(front_wheel_x)
    front_wheel_z = -LEG_LENGTH * jp.sin(front_fold)

    # Rear lower wheel: -x, +y side.
    # rear_fold is negative when folded downward, so z = L * sin(rear_fold).
    rear_wheel_x = -LEG_LENGTH * jp.cos(-rear_fold)
    rear_wheel_y = +0.062 * jp.ones_like(rear_wheel_x)
    rear_wheel_z = LEG_LENGTH * jp.sin(rear_fold)

    # Upper wheel: +x when stowed; positive upper_angle swings downward.
    upper_wheel_x = LEG_LENGTH * jp.cos(upper_angle)
    upper_wheel_y = jp.zeros_like(upper_wheel_x)
    upper_wheel_z = -LEG_LENGTH * jp.sin(upper_angle)

    # Limb centers are halfway to wheel centers.
    hub_pos = jp.concatenate([
        jp.zeros_like(front_wheel_x),
        jp.zeros_like(front_wheel_x),
        jp.zeros_like(front_wheel_x),
    ], axis=-1)

    front_leg_pos = 0.5 * jp.concatenate([front_wheel_x, front_wheel_y, front_wheel_z], axis=-1)
    rear_leg_pos = 0.5 * jp.concatenate([rear_wheel_x, rear_wheel_y, rear_wheel_z], axis=-1)
    upper_arm_pos = 0.5 * jp.concatenate([upper_wheel_x, upper_wheel_y, upper_wheel_z], axis=-1)

    front_wheel_pos = jp.concatenate([front_wheel_x, front_wheel_y, front_wheel_z], axis=-1)
    rear_wheel_pos = jp.concatenate([rear_wheel_x, rear_wheel_y, rear_wheel_z], axis=-1)
    upper_wheel_pos = jp.concatenate([upper_wheel_x, upper_wheel_y, upper_wheel_z], axis=-1)

    rel_pos = jp.stack([
        hub_pos,
        front_leg_pos,
        rear_leg_pos,
        upper_arm_pos,
        front_wheel_pos,
        rear_wheel_pos,
        upper_wheel_pos,
    ], axis=-2) / LEG_LENGTH

    # Type one-hot: hub, leg, arm, wheel.
    node_type = jp.array([
        [1, 0, 0, 0],  # hub
        [0, 1, 0, 0],  # front leg
        [0, 1, 0, 0],  # rear leg
        [0, 0, 1, 0],  # upper arm
        [0, 0, 0, 1],  # front wheel
        [0, 0, 0, 1],  # rear wheel
        [0, 0, 0, 1],  # upper wheel
    ], dtype=jp.float32)

    batch_shape = obs_batch.shape[:-1]
    node_type = jp.broadcast_to(node_type, (*batch_shape, NUM_NODES, 4))

    side_sign = jp.array([
        [0.0],   # hub
        [+1.0],  # front leg
        [-1.0],  # rear leg
        [0.0],   # upper arm
        [+1.0],  # front wheel
        [-1.0],  # rear wheel
        [0.0],   # upper wheel
    ], dtype=jp.float32)
    side_sign = jp.broadcast_to(side_sign, (*batch_shape, NUM_NODES, 1))

    joint_angle = jp.stack([
        jp.zeros_like(front_fold),
        front_fold,
        rear_fold,
        upper_angle,
        jp.zeros_like(front_fold),
        jp.zeros_like(front_fold),
        jp.zeros_like(front_fold),
    ], axis=-2) / ANGLE_SCALE

    joint_vel = jp.stack([
        jp.zeros_like(front_fold_vel),
        front_fold_vel,
        rear_fold_vel,
        upper_vel,
        jp.zeros_like(front_fold_vel),
        jp.zeros_like(front_fold_vel),
        jp.zeros_like(front_fold_vel),
    ], axis=-2) / JOINT_VEL_SCALE

    wheel_vel = jp.stack([
        jp.zeros_like(front_wheel_vel),
        jp.zeros_like(front_wheel_vel),
        jp.zeros_like(front_wheel_vel),
        jp.zeros_like(front_wheel_vel),
        front_wheel_vel,
        rear_wheel_vel,
        upper_wheel_vel,
    ], axis=-2) / WHEEL_SPEED_SCALE

    target_value = jp.stack([
        jp.zeros_like(fold_target),
        +fold_target,
        -fold_target,
        upper_target,
        jp.zeros_like(fold_target),
        jp.zeros_like(fold_target),
        jp.zeros_like(fold_target),
    ], axis=-2) / ANGLE_SCALE

    root_features = jp.concatenate([
        root_vx,
        root_vz,
        root_wy,
    ], axis=-1)
    root_features = jp.broadcast_to(
        root_features[..., None, :],
        (*batch_shape, NUM_NODES, root_features.shape[-1]),
    )

    node_features = jp.concatenate([
        node_type,       # 4
        side_sign,       # 1
        rel_pos,         # 3
        joint_angle,     # 1
        joint_vel,       # 1
        wheel_vel,       # 1
        target_value,    # 1
        root_features,   # 3
    ], axis=-1)

    global_features = jp.concatenate([
        root_qpos,
        root_qvel / 2.0,
        upper_target / ANGLE_SCALE,
        fold_target / ANGLE_SCALE,
        last_action,
    ], axis=-1)

    return node_features, global_features

# --------------------------
# Small MLP + message passing
# --------------------------
class MLP(nn.Module):
    widths: Sequence[int]
    activation: callable = nn.swish
    activate_final: bool = False
        

    @nn.compact
    def __call__(self, x):
        for i, w in enumerate(self.widths):
            x = nn.Dense(w)(x)
            if i < len(self.widths) - 1 or self.activate_final:
                x = self.activation(x)
        return x


class MessagePassingLayer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, h_nodes, node_features):
        # h_nodes: [Bflat, N, H]
        # node_features: [Bflat, N, F]

        h_s = h_nodes[:, SENDERS, :]
        h_r = h_nodes[:, RECEIVERS, :]
        h_diff = h_s - h_r

        x_s = node_features[:, SENDERS, :]
        x_r = node_features[:, RECEIVERS, :]
        x_diff = x_s - x_r

        static_edge = jp.broadcast_to(
            EDGE_ATTR[None],
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

        m = MLP((self.hidden_dim, self.hidden_dim), activate_final=True)(m_in)

        agg = jp.einsum("beh,en->bnh", m, RECV_ONEHOT)
        agg = agg / RECV_DEG[None, :, None]

        u_in = jp.concatenate([h_nodes, agg, node_features], axis=-1)
        delta = MLP((self.hidden_dim, self.hidden_dim), activate_final=True)(u_in)

        out = h_nodes + delta
        out = nn.LayerNorm()(out)
        return out


# --------------------------
# Policy / value modules
# --------------------------
class GraphPolicyModule(nn.Module):
    action_size: int
    hidden_dim: int = 64
    num_mp_layers: int = 2
    noise_std_type: Literal["scalar", "log"] = "log"
    init_noise_std: float = 0.5

    @nn.compact
    def __call__(self, obs_flat):
        node_features, global_features = parse_obs_to_nodes(obs_flat)

        h = MLP((self.hidden_dim, self.hidden_dim), activate_final=True)(node_features)

        leading_shape = h.shape[:-2]
        hidden_size = h.shape[-1]

        h_flat = jp.reshape(h, (-1, NUM_NODES, hidden_size))
        x_flat = jp.reshape(node_features, (-1, NUM_NODES, node_features.shape[-1]))

        for _ in range(self.num_mp_layers):
            h_flat = MessagePassingLayer(self.hidden_dim)(h_flat, x_flat)

        h = jp.reshape(h_flat, (*leading_shape, NUM_NODES, hidden_size))

        h_hub = h[..., HUB, :]
        h_front_leg = h[..., FRONT_LEG, :]
        h_rear_leg = h[..., REAR_LEG, :]
        h_upper_arm = h[..., UPPER_ARM, :]
        h_front_wheel = h[..., FRONT_WHEEL, :]
        h_rear_wheel = h[..., REAR_WHEEL, :]
        h_upper_wheel = h[..., UPPER_WHEEL, :]

        # 1. lower common drive: shared lower-wheel propulsion
        lower_common_in = jp.concatenate([
            h_hub,
            h_front_wheel + h_rear_wheel,
            global_features,
        ], axis=-1)

        # 2. lower drive difference: front/rear imbalance
        lower_diff_in = jp.concatenate([
            h_front_wheel - h_rear_wheel,
            h_front_leg - h_rear_leg,
            h_hub,
        ], axis=-1)

        # 3. upper wheel drive
        upper_wheel_in = jp.concatenate([
            h_upper_wheel,
            h_upper_arm,
            h_hub,
        ], axis=-1)

        # 4. upper arm target delta
        upper_arm_in = jp.concatenate([
            h_upper_arm,
            h_upper_wheel,
            h_hub,
            global_features,
        ], axis=-1)

        # 5. lower symmetric fold target delta
        fold_in = jp.concatenate([
            h_front_leg + h_rear_leg,
            h_front_wheel + h_rear_wheel,
            h_hub,
            global_features,
        ], axis=-1)

        lower_common = MLP((self.hidden_dim, self.hidden_dim, 1))(lower_common_in)
        lower_diff = MLP((self.hidden_dim, self.hidden_dim, 1))(lower_diff_in)
        upper_wheel = MLP((self.hidden_dim, self.hidden_dim, 1))(upper_wheel_in)
        upper_delta = MLP((self.hidden_dim, self.hidden_dim, 1))(upper_arm_in)
        fold_delta = MLP((self.hidden_dim, self.hidden_dim, 1))(fold_in)

        mean = jp.concatenate([
            lower_common,
            lower_diff,
            upper_wheel,
            upper_delta,
            fold_delta,
        ], axis=-1)

        if self.action_size != 5:
            raise ValueError(f"New robot graph policy expects action_size=5, got {self.action_size}")

        if self.noise_std_type == "scalar":
            log_std = self.param(
                "log_std",
                nn.initializers.constant(jp.log(self.init_noise_std)),
                (1,),
            )
            log_std = jp.broadcast_to(log_std, mean.shape)
        else:
            log_std = self.param(
                "log_std",
                nn.initializers.constant(jp.log(self.init_noise_std)),
                (self.action_size,),
            )
            log_std = jp.broadcast_to(log_std, mean.shape)

        return jp.concatenate([mean, log_std], axis=-1)


class GraphValueModule(nn.Module):
    hidden_dim: int = 64
    num_mp_layers: int = 3

    @nn.compact
    def __call__(self, obs_flat):
        node_features, global_features = parse_obs_to_nodes(obs_flat)

        h = MLP((self.hidden_dim, self.hidden_dim), activate_final=True)(node_features)

        leading_shape = h.shape[:-2]
        hidden_size = h.shape[-1]

        h_flat = jp.reshape(h, (-1, NUM_NODES, hidden_size))
        x_flat = jp.reshape(node_features, (-1, NUM_NODES, node_features.shape[-1]))

        for _ in range(self.num_mp_layers):
            h_flat = MessagePassingLayer(self.hidden_dim)(h_flat, x_flat)

        h = jp.reshape(h_flat, (*leading_shape, NUM_NODES, hidden_size))

        h_mean = jp.mean(h, axis=-2)
        h_hub = h[..., HUB, :]

        value_in = jp.concatenate([
            h_hub,
            h_mean,
            global_features,
        ], axis=-1)

        v = MLP((self.hidden_dim, self.hidden_dim, 1), activate_final=False)(value_in)
        return jp.squeeze(v, axis=-1)


# --------------------------
# FeedForwardNetwork wrappers (match brax.training.networks API)
# --------------------------
def _make_ffn(
    module: nn.Module,
    observation_size: types.ObservationSize,
    preprocess_observations_fn: types.PreprocessObservationFn,
    obs_key: str,
):
    """Wraps a flax module into brax.training.networks.FeedForwardNetwork."""

    def _dummy_from_obs_size(obs_size):
        # obs_size may be:
        #   45
        #   (45,)
        #   {"state": (45,), ...}
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
        del normalizer_params
#         observations = preprocess_observations_fn(observations, normalizer_params)
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
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,  # unused; keep for API compatibility
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,  # unused
    activation: networks.ActivationFn = nn.swish,  # unused
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    distribution_type: Literal["normal", "tanh_normal"] = "tanh_normal",
    noise_std_type: Literal["scalar", "log"] = "scalar",
    init_noise_std: float = 0.5,
    state_dependent_std: bool = False,  # unused in this simple implementation
    **kwargs: Any,
) -> PPONetworks:
#     del policy_hidden_layer_sizes, value_hidden_layer_sizes, activation
#     del state_dependent_std, kwargs

    if distribution_type == "normal":
        pad = distribution.NormalDistribution(event_size=action_size)
    elif distribution_type == "tanh_normal":
        pad = distribution.NormalTanhDistribution(event_size=action_size)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

    if pad.param_size != 2 * action_size:
        raise ValueError(
            f"Expected distribution param_size={2 * action_size}, got {pad.param_size}. "
            "Adjust GraphPolicyModule output to match your Brax distribution."
        )
    hidden_dim = kwargs.pop("hidden_dim", 64)
    num_mp_layers = kwargs.pop("num_mp_layers", 2)

    policy_module = GraphPolicyModule(
        action_size=action_size,
        hidden_dim = hidden_dim,
        num_mp_layers = num_mp_layers,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
    )
    value_module = GraphValueModule(hidden_dim=hidden_dim, num_mp_layers=num_mp_layers)

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
