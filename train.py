"""Train a PPO agent to play 3D Tetris.

Usage:
    python train.py --updates 500 --rollout 2048

Checkpoints are written to checkpoints/latest.pt (every save interval) and
checkpoints/best.pt (best mean episode score). The GUI server reads latest.pt.
"""
import argparse
import json
import os
import time

import numpy as np

from tetris3d.env import Tetris3DEnv, NUM_ACTIONS
from tetris3d.ppo import PPO

CKPT_DIR = "checkpoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--rollout", type=int, default=2048, help="steps per update")
    ap.add_argument("--max-ep-steps", type=int, default=3000)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-shaping", action="store_true")
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    np.random.seed(args.seed)

    env = Tetris3DEnv(seed=args.seed, shaping=not args.no_shaping)
    agent = PPO(Tetris3DEnv.obs_dim(), NUM_ACTIONS, lr=args.lr)

    obs = env.reset()
    ep_return, ep_len, ep_score = 0.0, 0, 0
    ep_returns, ep_scores, ep_lines = [], [], []
    best_score = -1e9
    log_path = os.path.join(CKPT_DIR, "train_log.jsonl")

    global_step = 0
    t0 = time.time()
    for update in range(1, args.updates + 1):
        buf = {k: [] for k in ["obs", "actions", "logp", "rewards", "values", "dones"]}
        for _ in range(args.rollout):
            action, logp, value = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            buf["obs"].append(obs)
            buf["actions"].append(action)
            buf["logp"].append(logp)
            buf["rewards"].append(reward)
            buf["values"].append(value)
            buf["dones"].append(float(done))
            obs = next_obs
            ep_return += reward
            ep_len += 1
            ep_score = info["score"]
            global_step += 1

            if done or ep_len >= args.max_ep_steps:
                ep_returns.append(ep_return)
                ep_scores.append(ep_score)
                ep_lines.append(info["cleared_layers"])
                ep_return, ep_len = 0.0, 0
                obs = env.reset()

        # Bootstrap value for the final state.
        _, _, last_value = agent.act(obs)
        rewards = np.array(buf["rewards"], dtype=np.float32)
        values = np.array(buf["values"], dtype=np.float32)
        dones = np.array(buf["dones"], dtype=np.float32)
        adv, returns = agent.compute_gae(rewards, values, dones, last_value)

        batch = {
            "obs": np.array(buf["obs"], dtype=np.float32),
            "actions": np.array(buf["actions"], dtype=np.int64),
            "logp": np.array(buf["logp"], dtype=np.float32),
            "adv": adv,
            "returns": returns,
        }
        stats = agent.update(batch)

        n_ep = max(len(ep_returns), 1)
        mean_ret = float(np.mean(ep_returns[-20:])) if ep_returns else 0.0
        mean_score = float(np.mean(ep_scores[-20:])) if ep_scores else 0.0
        mean_lines = float(np.mean(ep_lines[-20:])) if ep_lines else 0.0
        fps = int(global_step / (time.time() - t0))

        print(f"upd {update:4d} | step {global_step:8d} | fps {fps:5d} | "
              f"ep {len(ep_returns):4d} | ret {mean_ret:8.1f} | "
              f"score {mean_score:7.1f} | lines {mean_lines:5.2f} | "
              f"ent {stats['entropy']:.3f} | vf {stats['vf_loss']:.2f}",
              flush=True)

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "update": update, "step": global_step,
                "mean_return": mean_ret, "mean_score": mean_score,
                "mean_lines": mean_lines, **stats,
            }) + "\n")

        if update % args.save_every == 0:
            agent.save(os.path.join(CKPT_DIR, "latest.pt"))
        if mean_score > best_score and ep_scores:
            best_score = mean_score
            agent.save(os.path.join(CKPT_DIR, "best.pt"))

    agent.save(os.path.join(CKPT_DIR, "latest.pt"))
    print("done. best mean score:", best_score)


if __name__ == "__main__":
    main()
