"""Competitive training: two PPO agents learn 3D Tetris by playing each other.

Each agent uses the afterstate (placement) policy. They play versus matches:
clearing layers sends garbage that raises the opponent's board; topping out
loses. Both agents train on their own experience and co-adapt.

Usage:
    python train.py --updates 5000 --rollout 1024

Checkpoints: checkpoints/latest_p0.pt, latest_p1.pt (for the GUI) and
best_p0.pt / best_p1.pt (best combined performance). Resumes automatically.
"""
import argparse
import json
import os
import time

import numpy as np

from tetris3d.env import Tetris3DEnv
from tetris3d.versus import VersusEnv
from tetris3d.afterstate_agent import AfterstatePPO

CKPT_DIR = "checkpoints"


def make_agent(lr):
    return AfterstatePPO(Tetris3DEnv.state_dim(), Tetris3DEnv.placement_dim(), lr=lr)


def greedy_eval(agents, n_matches=4, max_rounds=250):
    """Play greedy (argmax) versus matches -- the agents' TRUE skill, vs the
    noisy sampled-policy training metric. Returns (avg layers/agent, avg
    garbage attacks/match)."""
    env = VersusEnv(prefill_p=0.0)
    layers = 0
    attacks = 0
    for _ in range(n_matches):
        env.reset()
        for _ in range(max_rounds):
            chosen = []
            ok = True
            for p in (0, 1):
                pls, feats = env.legal(p)
                if not pls:
                    env.envs[p].done = True
                    ok = False
                    break
                idx, _, _ = agents[p].act(env.state(p), feats, greedy=True)
                chosen.append(pls[idx])
            if not ok:
                break
            _, _, done, _ = env.step(chosen)
            if done:
                break
        layers += env.envs[0].cleared_layers + env.envs[1].cleared_layers
        attacks += sum(env.attacks)
    return layers / (2 * n_matches), attacks / n_matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=5000)
    ap.add_argument("--rollout", type=int, default=1024, help="rounds per update")
    ap.add_argument("--save-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--curriculum-updates", type=int, default=200,
                    help="updates over which the prefill curriculum decays to 0")
    ap.add_argument("--eval-every", type=int, default=5,
                    help="run a greedy evaluation (true skill) every N updates")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    np.random.seed(args.seed)

    agents = [make_agent(args.lr), make_agent(args.lr)]
    for p in (0, 1):
        ckpt = os.path.join(CKPT_DIR, f"latest_p{p}.pt")
        if not args.fresh and os.path.exists(ckpt):
            try:
                agents[p].load(ckpt)
                print(f"resumed agent {p} from {ckpt}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"could not resume agent {p} ({e}); fresh", flush=True)

    env = VersusEnv(seed=args.seed)
    states = env.reset()

    # per-agent rollout buffers
    def empty_buf():
        return {k: [] for k in ["state", "cand", "act", "logp", "rew", "val", "done"]}

    ep_stats = []          # per finished game: (winner, cleared0, cleared1, rounds)
    ep_return = [0.0, 0.0]
    best_perf = -1e18
    greedy_lines = 0.0     # true (argmax) skill, refreshed every eval-every updates
    greedy_attacks = 0.0
    log_path = os.path.join(CKPT_DIR, "train_log.jsonl")
    t0 = time.time()
    global_round = 0

    for update in range(1, args.updates + 1):
        prefill = max(0.0, 0.7 * (1.0 - update / max(1, args.curriculum_updates)))
        env.set_prefill(prefill)
        bufs = [empty_buf(), empty_buf()]

        for _ in range(args.rollout):
            chosen = [None, None]
            stepdata = [None, None]
            for p in (0, 1):
                cands, feats = env.legal(p)
                if not cands:                      # safety: treat as forced loss
                    env.envs[p].done = True
                    # pick a dummy; step will end the game
                    chosen[p] = None
                    stepdata[p] = None
                    continue
                idx, logp, val = agents[p].act(states[p], feats)
                chosen[p] = cands[idx]
                stepdata[p] = (states[p], feats, idx, logp, val)

            if chosen[0] is None or chosen[1] is None:
                # one side had no move: end & reset, skip storing this partial round
                states = env.reset()
                continue

            next_states, rewards, done, info = env.step(chosen)
            for p in (0, 1):
                st, feats, idx, logp, val = stepdata[p]
                b = bufs[p]
                b["state"].append(st); b["cand"].append(feats); b["act"].append(idx)
                b["logp"].append(logp); b["val"].append(val)
                b["rew"].append(rewards[p]); b["done"].append(float(done))
                ep_return[p] += rewards[p]
            states = next_states
            global_round += 1

            if done:
                ep_stats.append((info["winner"], info["cleared"][0], info["cleared"][1], info["rounds"]))
                ep_return = [0.0, 0.0]
                states = env.reset()

        # PPO update for each agent
        ustats = []
        for p in (0, 1):
            b = bufs[p]
            if not b["rew"]:
                ustats.append({"pg_loss": 0, "vf_loss": 0, "entropy": 0})
                continue
            last_val = agents[p].value_of(states[p])
            rew = np.array(b["rew"], np.float32)
            val = np.array(b["val"], np.float32)
            dn = np.array(b["done"], np.float32)
            adv, ret = agents[p].compute_gae(rew, val, dn, last_val)
            ustats.append(agents[p].update(
                b["state"], b["cand"], np.array(b["act"], np.int64),
                np.array(b["logp"], np.float32), adv, ret))

        # metrics over recent games
        recent = ep_stats[-50:]
        if recent:
            cleared = [(c0 + c1) / 2.0 for _, c0, c1, _ in recent]
            mean_lines = float(np.mean(cleared))
            mean_rounds = float(np.mean([r for *_, r in recent]))
            wins0 = sum(1 for w, *_ in recent if w == 0)
            wins1 = sum(1 for w, *_ in recent if w == 1)
            winrate0 = wins0 / len(recent)
        else:
            mean_lines = mean_rounds = winrate0 = 0.0
        ent = (ustats[0]["entropy"] + ustats[1]["entropy"]) / 2
        fps = int(global_round / (time.time() - t0))

        # Greedy evaluation = the agents' TRUE skill (the sampled `mean_lines`
        # above is depressed by exploration). This is what we log as the push /
        # best-checkpoint metric so improvements are real.
        if update % args.eval_every == 0 or update == 1:
            greedy_lines, greedy_attacks = greedy_eval(agents)

        print(f"upd {update:4d} | round {global_round:7d} | rps {fps:4d} | "
              f"games {len(ep_stats):4d} | sampled_lines {mean_lines:4.2f} | "
              f"greedy_lines {greedy_lines:5.1f} | greedy_atk {greedy_attacks:4.1f} | "
              f"rounds {mean_rounds:5.1f} | ent {ent:.3f}", flush=True)

        # push_if_improved reads mean_lines/mean_score/mean_return -> tie them to
        # greedy skill so it pushes on genuine improvement.
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "update": update, "round": global_round, "prefill": round(prefill, 3),
                "mean_lines": greedy_lines, "mean_score": greedy_lines * 100.0,
                "mean_return": greedy_attacks, "greedy_attacks": greedy_attacks,
                "sampled_lines": mean_lines, "mean_rounds": mean_rounds,
                "winrate0": winrate0, "entropy": ent,
                "vf_loss": (ustats[0]["vf_loss"] + ustats[1]["vf_loss"]) / 2,
            }) + "\n")

        perf = greedy_lines + greedy_attacks
        if update % args.save_every == 0:
            for p in (0, 1):
                agents[p].save(os.path.join(CKPT_DIR, f"latest_p{p}.pt"))
        if perf > best_perf:
            best_perf = perf
            for p in (0, 1):
                agents[p].save(os.path.join(CKPT_DIR, f"best_p{p}.pt"))

    for p in (0, 1):
        agents[p].save(os.path.join(CKPT_DIR, f"latest_p{p}.pt"))
    print("done. best mean lines:", best_perf)


if __name__ == "__main__":
    main()
