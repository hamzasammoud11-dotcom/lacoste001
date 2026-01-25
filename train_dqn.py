import os
import time
import numpy as np
import torch

from dqn_core import (
    DQNConfig, set_seed, make_env, unpack_reset, unpack_step,
    DQNAgent, ReplayBuffer
)

@torch.no_grad()
def evaluate(agent: DQNAgent, cfg: DQNConfig, device: torch.device) -> float:
    env = make_env(cfg.env_id, cfg.seed + 999, cfg.env_kwargs)
    rets = []

    for _ in range(cfg.eval_episodes):
        obs = unpack_reset(env.reset())
        obs = agent.preprocess(obs, update_stats=False)

        ep_ret = 0.0
        done = False
        while not done:
            a = agent.act(obs, step=cfg.eps_decay_steps + 1, greedy=True)
            obs2, r, done, _info = unpack_step(env.step(a))
            obs2 = agent.preprocess(obs2, update_stats=False)
            ep_ret += float(r)
            obs = obs2

        rets.append(ep_ret)

    env.close()
    return float(np.mean(rets))

def main():
    cfg = DQNConfig()
    cfg.env_kwargs = {
        "n_elements": 8,
        "n_phase_bins": 16,
        "steer_deg": 20.0,
        "theta_step_deg": 1.0,
    }

    cfg.total_steps = 30_000
    cfg.update_every = 8
    cfg.target_update_every = 500
    cfg.batch_size = 64
    cfg.min_buffer_size = 2_000
    cfg.eval_every_steps = 5_000
    cfg.eval_episodes = 3
    cfg.eps_decay_steps = 20_000

    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    except Exception:
        pass

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, cfg.model_name)

    env = make_env(cfg.env_id, cfg.seed, cfg.env_kwargs)

    obs0 = unpack_reset(env.reset())
    obs_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)

    agent = DQNAgent(cfg, obs_dim, n_actions, device)
    rb = ReplayBuffer(cfg.buffer_size, obs_dim)

    obs = agent.preprocess(obs0, update_stats=True)

    best_eval = -1e18
    t0 = time.time()
    last_print = time.time()

    for step in range(1, cfg.total_steps + 1):
        a = agent.act(obs, step=step, greedy=False)

        obs2, r, done, _info = unpack_step(env.step(a))
        obs2p = agent.preprocess(obs2, update_stats=True)

        rb.add(obs, a, float(r), obs2p, done)

        obs = obs2p if not done else agent.preprocess(unpack_reset(env.reset()), update_stats=True)

        if step >= cfg.min_buffer_size and (step % cfg.update_every == 0):
            agent.train_step(rb.sample(cfg.batch_size))

        if step % cfg.target_update_every == 0:
            agent.update_target()

        if time.time() - last_print >= 1.0:
            print(f"[STEP {step}/{cfg.total_steps}] buf={len(rb)}", flush=True)
            last_print = time.time()

        if step % cfg.eval_every_steps == 0:
            ev = evaluate(agent, cfg, device)
            elapsed = time.time() - t0
            print(f"[EVAL step={step}] return_mean={ev:.3f} elapsed_s={elapsed:.1f}", flush=True)
            if ev > best_eval:
                best_eval = ev
                agent.save(save_path)
                print(f"[SAVE] {save_path}", flush=True)

    env.close()

    if not os.path.exists(save_path):
        agent.save(save_path)
        print(f"[SAVE] {save_path}", flush=True)

if __name__ == "__main__":
    main()
