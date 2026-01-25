import os
import math
import random
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except Exception:
    import gym

IS_GYMNASIUM = getattr(gym, "__name__", "") == "gymnasium"


class AntennaToyEnv(gym.Env):
    """
    Env antenne jouet (autonome, pas CartPole):
    - Réseau linéaire de N éléments, phases discrètes
    - Action: choisir la phase de l'élément courant
    - Observation: cos(phases) + sin(phases) + step_norm
    - Fournit la courbe: get_antenna_curve() + info["theta_deg"/"pattern_db"]
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_elements: int = 8,
        n_phase_bins: int = 16,
        steer_deg: float = 20.0,
        spacing_lambda: float = 0.5,
        mainlobe_exclude_deg: float = 10.0,
        theta_step_deg: float = 1.0,
    ):
        super().__init__()
        self.n = int(n_elements)
        self.n_bins = int(n_phase_bins)
        self.steer_deg = float(steer_deg)
        self.d = float(spacing_lambda)
        self.main_excl = float(mainlobe_exclude_deg)

        self.theta_deg = np.arange(0.0, 360.0 + 1e-9, float(theta_step_deg), dtype=np.float32)
        self.theta_rad = np.deg2rad(self.theta_deg.astype(np.float64))

        self.action_space = gym.spaces.Discrete(self.n_bins)
        obs_dim = 2 * self.n + 1
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.phases = np.zeros(self.n, dtype=np.float64)
        self.k = 2.0 * math.pi  # lambda normalisée
        self.idx = 0
        self.prev_score = 0.0

        self._last_pattern_db = None

    def _array_factor_mag(self) -> np.ndarray:
        theta = self.theta_rad
        theta0 = math.radians(self.steer_deg)
        n_idx = np.arange(self.n, dtype=np.float64)

        psi = self.k * self.d * (np.sin(theta) - math.sin(theta0))
        w = np.exp(1j * self.phases)

        af = np.sum(w[None, :] * np.exp(1j * (psi[:, None] * n_idx[None, :])), axis=1)
        mag = np.abs(af).astype(np.float64)
        return mag

    def _pattern_db(self) -> np.ndarray:
        mag = self._array_factor_mag()
        mag = mag / (np.max(mag) + 1e-12)
        pat_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
        pat_db = pat_db.astype(np.float32)
        self._last_pattern_db = pat_db
        return pat_db

    def _score(self) -> float:
        # score = -SLL (plus grand = mieux)
        pat_db = self._pattern_db()

        diff = (self.theta_deg - self.steer_deg + 180.0) % 360.0 - 180.0
        mask = np.abs(diff) > self.main_excl

        sll = float(np.max(pat_db[mask]))  # négatif
        return -sll

    def _obs(self) -> np.ndarray:
        c = np.cos(self.phases).astype(np.float32)
        s = np.sin(self.phases).astype(np.float32)
        step_norm = np.array([self.idx / max(1, self.n - 1)], dtype=np.float32)
        return np.concatenate([c, s, step_norm], axis=0)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.phases[:] = 0.0
        self.idx = 0
        self.prev_score = self._score()
        obs = self._obs()
        info = {"idx": 0, "reward": 0.0}
        return (obs, info) if IS_GYMNASIUM else obs

    def step(self, action: int):
        a = int(action)
        a = max(0, min(self.n_bins - 1, a))
        phase = 2.0 * math.pi * (a / self.n_bins)

        self.phases[self.idx] = phase
        self.idx += 1

        terminated = self.idx >= self.n
        truncated = False

        score = self._score()
        reward = float(score - self.prev_score)
        self.prev_score = score

        obs = self._obs()
        info = {
            "idx": int(self.idx),
            "reward": float(reward),
            "theta_deg": self.theta_deg.copy(),
            "pattern_db": self._last_pattern_db.copy() if self._last_pattern_db is not None else self._pattern_db(),
            "score": float(score),
        }

        if IS_GYMNASIUM:
            return obs, reward, terminated, truncated, info
        return obs, reward, bool(terminated or truncated), info

    def get_antenna_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._last_pattern_db is None:
            _ = self._pattern_db()
        return self.theta_deg.copy(), self._last_pattern_db.copy()


@dataclass
class DQNConfig:
    env_id: str = "AntennaToy-v0"
    env_kwargs: Optional[Dict[str, Any]] = None
    seed: int = 42

    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 64

    buffer_size: int = 50_000
    min_buffer_size: int = 2_000

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000

    update_every: int = 8
    target_update_every: int = 500

    total_steps: int = 30_000
    eval_every_steps: int = 5_000
    eval_episodes: int = 3

    out_dir: str = "artifacts_dqn"
    model_name: str = "dqn.pt"

    @classmethod
    def from_dict(cls, d: dict) -> "DQNConfig":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in (d or {}).items() if k in valid}
        if filtered.get("env_kwargs") is None:
            filtered["env_kwargs"] = {}
        return cls(**filtered)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_reset(reset_out):
    return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def unpack_step(step_out):
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs2, r, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
        return obs2, float(r), done, info
    if isinstance(step_out, tuple) and len(step_out) == 4:
        obs2, r, done, info = step_out
        return obs2, float(r), bool(done), info
    raise ValueError("Unexpected env.step output format")


def make_env(env_id: str, seed: int, env_kwargs: Optional[Dict[str, Any]] = None):
    env_kwargs = env_kwargs or {}
    if env_id == "AntennaToy-v0":
        env = AntennaToyEnv(**env_kwargs)
    else:
        env = gym.make(env_id, **env_kwargs)

    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    try:
        env.action_space.seed(seed)
    except Exception:
        pass
    return env


class RunningMeanStd:
    def __init__(self, shape, init_count: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(init_count)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        bm = x.mean(axis=0)
        bv = x.var(axis=0)
        bc = x.shape[0]
        self._update_from_moments(bm, bv, bc)

    def _update_from_moments(self, bm, bv, bc):
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        m2 = m_a + m_b + (delta ** 2) * self.count * bc / tot
        new_var = m2 / tot
        self.mean = new_mean
        self.var = new_var
        self.count = tot

    def normalize(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean.astype(np.float32)) / (np.sqrt(self.var.astype(np.float32)) + 1e-8)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((self.capacity,), dtype=np.int64)
        self.rews = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def add(self, s, a, r, sp, done):
        i = self.ptr
        self.obs[i] = s
        self.acts[i] = a
        self.rews[i] = r
        self.next_obs[i] = sp
        self.dones[i] = 1.0 if done else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int):
        n = len(self)
        idx = np.random.randint(0, n, size=int(batch_size))
        return (
            self.obs[idx],
            self.acts[idx],
            self.rews[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, cfg: DQNConfig, obs_dim: int, n_actions: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions

        self.q_online = QNetwork(obs_dim, n_actions).to(device)
        self.q_target = QNetwork(obs_dim, n_actions).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.opt = optim.Adam(self.q_online.parameters(), lr=cfg.lr)
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))

    def eps_schedule(self, step: int) -> float:
        if step >= self.cfg.eps_decay_steps:
            return self.cfg.eps_end
        frac = step / float(self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def preprocess(self, obs: np.ndarray, update_stats: bool) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if update_stats:
            self.obs_rms.update(x)
        return self.obs_rms.normalize(x).astype(np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, step: int, greedy: bool = False) -> int:
        if (not greedy) and (random.random() < self.eps_schedule(step)):
            return random.randrange(self.n_actions)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q_online(x)
        return int(torch.argmax(q, dim=1).item())

    def train_step(self, batch) -> float:
        s, a, r, sp, done = batch
        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        sp_t = torch.tensor(sp, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        d_t = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q_online(s_t).gather(1, a_t)

        with torch.no_grad():
            next_a = torch.argmax(self.q_online(sp_t), dim=1, keepdim=True)
            q_next = self.q_target(sp_t).gather(1, next_a)
            y = r_t + (1.0 - d_t) * self.cfg.gamma * q_next

        loss = nn.functional.smooth_l1_loss(q_sa, y)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def update_target(self):
        self.q_target.load_state_dict(self.q_online.state_dict())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "q_online": self.q_online.state_dict(),
            "obs_rms_mean": self.obs_rms.mean,
            "obs_rms_var": self.obs_rms.var,
            "obs_rms_count": self.obs_rms.count,
            "cfg": self.cfg.__dict__,
        }
        torch.save(payload, path)

    def load(self, path: str):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.q_online.load_state_dict(payload["q_online"])
        self.q_target.load_state_dict(payload["q_online"])
        self.obs_rms.mean = payload["obs_rms_mean"]
        self.obs_rms.var = payload["obs_rms_var"]
        self.obs_rms.count = payload["obs_rms_count"]


