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
BODY, FL, FR, RL, RR = 0, 1, 2, 3, 4
NUM_NODES = 5
NUM_MODULES = 4

EDGE_PAIRS = [
    # body-star
    (BODY, FL), (FL, BODY),
    (BODY, FR), (FR, BODY),
    (BODY, RL), (RL, BODY),
    (BODY, RR), (RR, BODY),
    # module ring
    (FL, FR), (FR, FL),
    (FR, RR), (RR, FR),
    (RR, RL), (RL, RR),
    (RL, FL), (FL, RL),
    # self-loops
    (BODY, BODY),
    (FL, FL),
    (FR, FR),
    (RL, RL),
    (RR, RR),
    # diagonal 
    (FL, RR), (RR, FL),
    (FR, RL), (RL, FR),
]
SENDERS = jp.array([i for i, j in EDGE_PAIRS], dtype=jp.int32)
RECEIVERS = jp.array([j for i, j in EDGE_PAIRS], dtype=jp.int32)
NUM_EDGES = SENDERS.shape[0]

NODE_XY = jp.array([
    [0.0, 0.0],   # BODY
    [1.0, -1.0],   # FL
    [1.0, +1.0],  # FR
    [-1.0, -1.0],  # RL
    [-1.0, +1.0], # RR
], dtype=jp.float32)


BILATERAL_PAIRS = {frozenset([FL, FR]), frozenset([RL, RR])}
LONGITUDINAL_PAIRS = {frozenset([FL, RL]), frozenset([FR, RR])}
DIAGONAL_PAIRS = {frozenset([FL, RR]), frozenset([FR, RL])}

def _build_edge_attr():
    attrs = []
    for i, j in EDGE_PAIRS:
        dxdy = NODE_XY[j] - NODE_XY[i]
        is_body_module = 1.0 if ((i == BODY) ^ (j == BODY)) else 0.0
        is_module_module = 1.0 if (i != BODY and j != BODY and i != j) else 0.0
        is_self = 1.0 if i == j else 0.0
        
        # New symmetry flags
        pair = frozenset([i, j])
        is_bilateral = 1.0 if pair in BILATERAL_PAIRS else 0.0
        is_longitudinal = 1.0 if pair in LONGITUDINAL_PAIRS else 0.0
        
        attrs.append(jp.array(
            [dxdy[0], dxdy[1], is_body_module, is_module_module, 
             is_self, is_bilateral, is_longitudinal],
            dtype=jp.float32,
        ))
    return jp.stack(attrs, axis=0)


EDGE_ATTR = _build_edge_attr()  # [E, Fe]
RECV_ONEHOT = jax.nn.one_hot(RECEIVERS, NUM_NODES, dtype=jp.float32)  # [E, N]
RECV_DEG = jp.sum(RECV_ONEHOT, axis=0)  # [N]


# --------------------------
# obs -> (body,node) features
# --------------------------
def parse_obs_to_nodes(obs_batch: jp.ndarray):
    qpos = obs_batch[..., :23]
    qvel = obs_batch[..., 23:45]

    root_qpos = qpos[..., :7]
    root_qvel = qvel[..., :6]

    mod_qpos = jp.reshape(qpos[..., 7:], (*obs_batch.shape[:-1], 4, 4))
    mod_qvel = jp.reshape(qvel[..., 6:], (*obs_batch.shape[:-1], 4, 4))

    # body features
    body_lin = root_qvel[..., 0:3]
    body_ang = root_qvel[..., 3:6]

    # module features
    wheel_speed = mod_qvel[..., :, 0:1]
    ext_pos = mod_qpos[..., :, 1:4]
    ext_vel = mod_qvel[..., :, 1:4]

    ext_mean = jp.mean(ext_pos, axis=-1, keepdims=True)
    ext_rate = jp.mean(ext_vel, axis=-1, keepdims=True)

    rel_comp = ext_mean - jp.mean(ext_mean, axis=-2, keepdims=True)

    # crude body speed magnitude, broadcast to modules
    body_speed = jp.linalg.norm(body_lin[..., :2], axis=-1, keepdims=True)
    body_speed = body_speed[..., None, :]  # [..., 1, 1]
    body_speed = jp.broadcast_to(body_speed, ext_mean.shape)

    # crude slip proxy
    slip_proxy = jp.abs(wheel_speed) - body_speed

    corner_id = jp.array(
        [[1., -1.], [1., +1.], [-1., -1.], [-1., +1.]],
        dtype=jp.float32
    )
    corner_id_b = jp.broadcast_to(corner_id, (*obs_batch.shape[:-1], 4, 2))
    dummy = jp.zeros((*obs_batch.shape[:-1], 1), dtype=obs_batch.dtype)
    body_raw = jp.concatenate([root_qpos, root_qvel], axis=-1)[..., None, :]

    module_raw = jp.concatenate([
        mod_qpos,
        mod_qvel,
        ext_mean,
        ext_rate,
        rel_comp,
        corner_id_b,
        body_speed,
        slip_proxy
         
    ], axis=-1)

    return body_raw, module_raw

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
    def __call__(self, h_nodes, obs_nodes):
        # h_nodes: [Bflat, N, H]
        EXT_IDX, EXT_RATE_IDX, WHEEL_IDX = 8, 9, 4

        ext       = obs_nodes[:, :, EXT_IDX]        # [B, N]
        ext_rate  = obs_nodes[:, :, EXT_RATE_IDX]
        wheel     = obs_nodes[:, :, WHEEL_IDX]

        ext_s     = ext[:, SENDERS]                  # [B, E]
        ext_r     = ext[:, RECEIVERS]
        rate_s    = ext_rate[:, SENDERS]
        rate_r    = ext_rate[:, RECEIVERS]
        wheel_s   = wheel[:, SENDERS]
        wheel_r   = wheel[:, RECEIVERS]

        # Gate wheel diff to module-module edges only
        is_mm = ((SENDERS != BODY) & (RECEIVERS != BODY)).astype(jp.float32)
        wheel_diff = (wheel_s - wheel_r) * is_mm

        dynamic_edge = jp.stack([
            ext_s - ext_r,
            rate_s - rate_r,
            wheel_diff,
        ], axis=-1)  # [B, E, 3]

        h_s = h_nodes[:, SENDERS, :]
        h_r = h_nodes[:, RECEIVERS, :]
        h_diff = h_s - h_r

        static_edge = jp.broadcast_to(
            EDGE_ATTR[None],
            (h_nodes.shape[0], NUM_EDGES, EDGE_ATTR.shape[-1]),
        )
        edge_features = jp.concatenate([static_edge, dynamic_edge], axis=-1)

        m_in = jp.concatenate([h_s, h_r, h_diff, edge_features], axis=-1)
        m = MLP((self.hidden_dim, self.hidden_dim), activate_final=True)(m_in)

        agg = jp.einsum("beh,en->bnh", m, RECV_ONEHOT)
        agg = agg / RECV_DEG[None, :, None]

        u_in = jp.concatenate([h_nodes, agg], axis=-1)
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
    init_noise_std: float = 1.0

    @nn.compact
    def __call__(self, obs_flat):
        # obs_flat: [B, 45] (already normalized by preprocess fn)
        body_raw, module_raw = parse_obs_to_nodes(obs_flat)

        body_h = MLP((self.hidden_dim, self.hidden_dim), activate_final=True)(body_raw)
        mod_h = MLP((self.hidden_dim, self.hidden_dim), activate_final=True)(module_raw)
        h = jp.concatenate([body_h, mod_h], axis=-2)  # [B, 5, H]
        
        body_obs_pad = jp.zeros(
            (*obs_flat.shape[:-1], 1, module_raw.shape[-1]), dtype=obs_flat.dtype
        )
        obs_nodes = jp.concatenate([body_obs_pad, module_raw], axis=-2)  # [B, 5, obs_dim]

        leading_shape = h.shape[:-2]
        hidden_size = h.shape[-1]
        h_flat = jp.reshape(h, (-1, NUM_NODES, hidden_size))
        obs_flat_nodes = jp.reshape(obs_nodes, (-1, NUM_NODES, obs_nodes.shape[-1]))

        for _ in range(self.num_mp_layers):
            h_flat = MessagePassingLayer(self.hidden_dim)(h_flat, obs_flat_nodes)

        h = jp.reshape(h_flat, (*leading_shape, NUM_NODES, hidden_size))
        
        if self.action_size % NUM_MODULES != 0:
            raise ValueError(
                f"action_size must be divisible by {NUM_MODULES}, got {self.action_size}"
            )

        per_module_action_dim = self.action_size // NUM_MODULES
        h_body = jp.broadcast_to(h[..., 0:1, :], h[..., 1:5, :].shape)
        actor_in = jp.concatenate([h[..., 1:5, :], h_body], axis=-1)

        out_fl = MLP((self.hidden_dim, self.hidden_dim, per_module_action_dim), activate_final=False)(actor_in[..., 0, :])
        out_fr = MLP((self.hidden_dim, self.hidden_dim, per_module_action_dim), activate_final=False)(actor_in[..., 1, :])
        out_rl = MLP((self.hidden_dim, self.hidden_dim, per_module_action_dim), activate_final=False)(actor_in[..., 2, :])
        out_rr = MLP((self.hidden_dim, self.hidden_dim, per_module_action_dim), activate_final=False)(actor_in[..., 3, :])

        out = jp.stack([out_fl, out_fr, out_rl, out_rr], axis=-2)
        
        mean = jp.reshape(jp.swapaxes(out, -2, -1), (*leading_shape, self.action_size))

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
        body_raw, module_raw = parse_obs_to_nodes(obs_flat)

        body_h = MLP((self.hidden_dim, self.hidden_dim), activate_final=True)(body_raw)
        mod_h = MLP((self.hidden_dim, self.hidden_dim), activate_final=True)(module_raw)

        h = jp.concatenate([body_h, mod_h], axis=-2)

        leading_shape = h.shape[:-2]
        hidden_size = h.shape[-1]
        # Build obs_nodes for dynamic edges — pad body with zeros to match module dim
        body_obs_pad = jp.zeros(
            (*obs_flat.shape[:-1], 1, module_raw.shape[-1]), dtype=obs_flat.dtype
        )
        obs_nodes = jp.concatenate([body_obs_pad, module_raw], axis=-2)  # [B, 5, obs_dim]
        h_flat = jp.reshape(h, (-1, NUM_NODES, hidden_size))
        obs_flat_nodes = jp.reshape(obs_nodes, (-1, NUM_NODES, obs_nodes.shape[-1]))

        for _ in range(self.num_mp_layers):
            h_flat = MessagePassingLayer(self.hidden_dim)(h_flat, obs_flat_nodes)

        h = jp.reshape(h_flat, (*leading_shape, NUM_NODES, hidden_size))

        flat = jp.reshape(h, (*leading_shape, NUM_NODES * hidden_size))
        v = MLP((self.hidden_dim, self.hidden_dim, 1), activate_final=False)(flat)
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
