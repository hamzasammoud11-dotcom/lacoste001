import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from dqn_core import (
    DQNConfig, QNetwork, RunningMeanStd,
    make_env, unpack_reset, unpack_step
)

def plot_radial(theta_deg: np.ndarray, pattern_db: np.ndarray, idx: int, reward: float,
                out_png: str, show: bool = True):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    theta_deg = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
    pattern_db = np.asarray(pattern_db, dtype=np.float64).reshape(-1)

    rmin, rmax = -40.0, 0.0
    pat = np.clip(pattern_db, rmin, rmax)
    th = np.deg2rad(theta_deg)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(th, pat)
    ax.set_title(f"Radiation pattern (idx={idx}, reward={reward:.2f})")

    ax.set_rlim(rmin, rmax)
    ax.set_rticks([-35, -30, -25, -20, -15, -10, -5, 0])
    ax.grid(True)

    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

class DQNPolicy:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)

        payload = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg_dict = payload.get("cfg", {}) or {}
        self.cfg = DQNConfig.from_dict(cfg_dict)

        env = make_env(self.cfg.env_id, self.cfg.seed, self.cfg.env_kwargs)
        obs0 = unpack_reset(env.reset())
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
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return self.obs_rms.normalize(obs).astype(np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.net(x)
        return int(torch.argmax(q, dim=1).item())

def demo_run(model_path: str, episodes: int = 3, show_plots: bool = True):
    policy = DQNPolicy(model_path=model_path, device="cpu")
    env = make_env(policy.cfg.env_id, policy.cfg.seed + 123, policy.cfg.env_kwargs)

    plots_dir = os.path.join(policy.cfg.out_dir, "plots")

    for ep in range(episodes):
        obs = policy.preprocess(unpack_reset(env.reset()))
        done = False
        ep_ret = 0.0
        last_info = {}

        while not done:
            a = policy.act(obs)
            obs2, r, done, info = unpack_step(env.step(a))
            last_info = info if isinstance(info, dict) else {}
            obs = policy.preprocess(obs2)
            ep_ret += float(r)

        idx = int(last_info.get("idx", ep + 1))
        reward = float(last_info.get("reward", 0.0))
        print("Episode return:", ep_ret)

        base = getattr(env, "unwrapped", env)
        if hasattr(base, "get_antenna_curve"):
            theta, pat = base.get_antenna_curve()
        else:
            theta = last_info.get("theta_deg", None)
            pat = last_info.get("pattern_db", None)

        if theta is None or pat is None:
            print("No curve available")
        else:
            out_png = os.path.join(plots_dir, f"radiation_ep{ep+1}.png")
            plot_radial(theta, pat, idx=idx, reward=reward, out_png=out_png, show=show_plots)
            print("Saved:", out_png)

    env.close()

if __name__ == "__main__":
    demo_run(os.path.join("artifacts_dqn", "dqn.pt"), episodes=3, show_plots=True)
