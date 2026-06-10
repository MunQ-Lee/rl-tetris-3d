"""A compact PPO implementation (clipped objective + GAE) for Tetris3DEnv.

The policy/value network is a small MLP over the flattened board + current
piece grid + piece-type one-hots. The board is tiny (5x5x12), so an MLP trains
quickly on CPU.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class PPO:
    def __init__(self, obs_dim, num_actions, device="cpu",
                 lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                 epochs=4, minibatch=256, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5):
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, num_actions).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm

    def act(self, obs_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, logp, value = self.net.act(obs)
        return int(action.item()), float(logp.item()), float(value.item())

    def compute_gae(self, rewards, values, dones, last_value):
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_nonterminal - values[t]
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            adv[t] = gae
        returns = adv + values
        return adv, returns

    def update(self, batch):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        stats = {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "n": 0}
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch):
                mb = idx[start:start + self.minibatch]
                logp, entropy, value = self.net.evaluate(obs[mb], actions[mb])
                ratio = torch.exp(logp - old_logp[mb])
                s1 = ratio * adv[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[mb]
                pg_loss = -torch.min(s1, s2).mean()
                vf_loss = 0.5 * (returns[mb] - value).pow(2).mean()
                ent = entropy.mean()
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                stats["pg_loss"] += pg_loss.item()
                stats["vf_loss"] += vf_loss.item()
                stats["entropy"] += ent.item()
                stats["n"] += 1
        k = max(stats["n"], 1)
        return {"pg_loss": stats["pg_loss"] / k,
                "vf_loss": stats["vf_loss"] / k,
                "entropy": stats["entropy"] / k}

    def save(self, path):
        torch.save({"model": self.net.state_dict()}, path)

    def load(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.net.load_state_dict(ckpt["model"])
