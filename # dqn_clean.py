# dqn_clean.py
# DQN "sans dataset" (RL) : les données sont des transitions collectées en ligne.
# Déploiement : charger le modèle et choisir l'action à partir de l'observation.

import os
import random
from dataclasses import dataclass
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# Gym / Gymnasium compat
try:
    import gymnasium as gym
except Exception:
    import gym


# ------------------------- Config -------------------------

@dataclass
class DQNConfig:
    env_id: str = "CartPole-v1"
    seed: int = 42

    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 128

    buffer_size: int = 100_000
    min_buffer_size: int = 5_000

    # Exploration (epsilon, pas le symbole grec)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000  # linéaire

    # Update
    update_every: int = 1
    updates_per_step: int = 1

    # Target net
    tau: float = 0.005  # soft update ; mettre 1.0 + update périodique si tu préfères

    # Stabilisation
    huber_delta: float = 1.0
    grad_clip_norm: float = 10.0
    reward_clip: bool = False  # True si rewards instables/fortes
    reward_clip_value: float = 1.0

    # Entraînement
    total_steps: int = 200_000
    eval_every_steps: int = 10_000
    eval_episodes: int = 10

    # Sauvegarde
    out_dir: str = "artifacts_dqn"
    model_name: str = "dqn.pt"


# ------------------------- Utils -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RunningMeanStd:
    """Normalisation online des observations (pré-traitement à exporter)."""
    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ------------------------- Replay Buffer -------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, sp, done):
        self.buffer.append((s, a, r, sp, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, sp, done = map(np.array, zip(*batch))
        return s, a, r, sp, done

    def __len__(self):
        return len(self.buffer)


# ------------------------- Réseau Q -------------------------

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------- Agent DQN -------------------------

class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n_actions = n_actions

        self.q_online = QNetwork(obs_dim, n_actions).to(device)
        self.q_target = QNetwork(obs_dim, n_actions).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optim = optim.Adam(self.q_online.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss(beta=cfg.huber_delta)  # Huber

        self.rb = ReplayBuffer(cfg.buffer_size)

        self.global_step = 0
        self.eps = cfg.eps_start

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))

    def _eps_schedule(self, step: int) -> float:
        # Décroissance linéaire
        if step >= self.cfg.eps_decay_steps:
            return self.cfg.eps_end
        frac = step / self.cfg.eps_decay_steps
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def preprocess_obs(self, obs: np.ndarray, update_stats: bool) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if update_stats:
            self.obs_rms.update(obs.reshape(1, -1))
        obs_n = self.obs_rms.normalize(obs)
        return obs_n.astype(np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        if (not greedy) and (random.random() < self.eps):
            return random.randrange(self.n_actions)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q_online(x)
        return int(torch.argmax(q, dim=1).item())

    def store(self, s, a, r, sp, done):
        self.rb.push(s, a, r, sp, done)

    def soft_update_target(self):
        tau = self.cfg.tau
        with torch.no_grad():
            for p_t, p in zip(self.q_target.parameters(), self.q_online.parameters()):
                p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    def train_step(self):
        if len(self.rb) < self.cfg.min_buffer_size:
            return None

        s, a, r, sp, done = self.rb.sample(self.cfg.batch_size)

        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        sp_t = torch.tensor(sp, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done.astype(np.float32), dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a) online
        q_sa = self.q_online(s_t).gather(1, a_t)

        with torch.no_grad():
            # Double DQN:
            # action* = argmax_a Q_online(s',a)
            next_actions = torch.argmax(self.q_online(sp_t), dim=1, keepdim=True)
            # Q_target(s', action*)
            q_next = self.q_target(sp_t).gather(1, next_actions)

            # y = r si done else r + gamma*q_next
            y = r_t + (1.0 - done_t) * self.cfg.gamma * q_next

        td_error = (y - q_sa)  # delta (erreur TD)
        loss = self.loss_fn(q_sa, y)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        self.soft_update_target()

        return {
            "loss": float(loss.item()),
            "td_abs_mean": float(td_error.abs().mean().item()),
            "q_mean": float(q_sa.mean().item()),
            "y_mean": float(y.mean().item()),
        }

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
        payload = torch.load(path, map_location=self.device)
        self.q_online.load_state_dict(payload["q_online"])
        self.q_target.load_state_dict(payload["q_online"])
        self.obs_rms.mean = payload["obs_rms_mean"]
        self.obs_rms.var = payload["obs_rms_var"]
        self.obs_rms.count = payload["obs_rms_count"]


# ------------------------- Train / Eval -------------------------

def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.seed(seed)
    return env

@torch.no_grad()
def evaluate(agent: DQNAgent, env_id: str, seed: int, episodes: int):
    env = make_env(env_id, seed + 10_000)
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        obs = agent.preprocess_obs(obs, update_stats=False)
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(obs, greedy=True)
            step_out = env.step(a)
            if len(step_out) == 5:
                obs2, r, term, trunc, _ = step_out
                done = bool(term or trunc)
            else:
                obs2, r, done, _ = step_out
            obs2 = agent.preprocess_obs(obs2, update_stats=False)
            ep_ret += float(r)
            obs = obs2
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    cfg = DQNConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg.env_id, cfg.seed)

    # Obs dim / actions (discret)
    obs, _ = env.reset()
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    agent = DQNAgent(obs_dim, n_actions, cfg, device)

    # Boucle RL : collecte -> buffer -> TD updates
    obs = agent.preprocess_obs(obs, update_stats=True)

    logs = {"best_eval": -1e9}
    save_path = os.path.join(cfg.out_dir, cfg.model_name)

    for step in range(1, cfg.total_steps + 1):
        agent.global_step = step
        agent.eps = agent._eps_schedule(step)

        a = agent.act(obs, greedy=False)
        step_out = env.step(a)

        if len(step_out) == 5:
            obs2, r, term, trunc, _ = step_out
            done = bool(term or trunc)
        else:
            obs2, r, done, _ = step_out

        if cfg.reward_clip:
            r = float(np.clip(r, -cfg.reward_clip_value, cfg.reward_clip_value))

        obs2p = agent.preprocess_obs(obs2, update_stats=True)
        agent.store(obs, a, float(r), obs2p, done)

        # Updates
        if step % cfg.update_every == 0:
            for _ in range(cfg.updates_per_step):
                out = agent.train_step()
                # out contient: loss, td_abs_mean, etc. (l'erreur de prédiction au sens DQN)
                # Rien à "labelliser" : la cible y est calculée par bootstrap.

        # Reset épisode
        if done:
            obs, _ = env.reset()
            obs = agent.preprocess_obs(obs, update_stats=True)
        else:
            obs = obs2p

        # Eval périodique
        if step % cfg.eval_every_steps == 0:
            mean_ret, std_ret = evaluate(agent, cfg.env_id, cfg.seed, cfg.eval_episodes)
            if mean_ret > logs["best_eval"]:
                logs["best_eval"] = mean_ret
                agent.save(save_path)

    env.close()

    # Sauvegarde finale si rien n'a été sauvegardé
    if not os.path.exists(save_path):
        agent.save(save_path)


# ------------------------- Inference (déploiement) -------------------------

class DQNPolicy:
    """
    Politique déployable "sans dataset":
    - load(model.pt)
    - preprocess observation (normalisation exportée)
    - action = argmax Q(s, a)
    """
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        payload = torch.load(model_path, map_location=self.device)
        cfg_dict = payload["cfg"]
        self.cfg = DQNConfig(**cfg_dict)

        # Créer un env juste pour récupérer dimensions (optionnel, mais pratique)
        env = gym.make(self.cfg.env_id)
        obs_dim = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)
        env.close()

        self.net = QNetwork(obs_dim, n_actions).to(self.device)
        self.net.load_state_dict(payload["q_online"])
        self.net.eval()

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.obs_rms.mean = payload["obs_rms_mean"]
        self.obs_rms.var = payload["obs_rms_var"]
        self.obs_rms.count = payload["obs_rms_count"]

    def preprocess(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        obs_n = self.obs_rms.normalize(obs)
        return obs_n.astype(np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.net(x)
        return int(torch.argmax(q, dim=1).item())


if __name__ == "__main__":
    main()
