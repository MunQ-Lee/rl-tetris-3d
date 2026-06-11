"""Masked PPO over afterstate placements.

The policy scores each candidate placement by its afterstate features and takes
a softmax over the (variable-size) legal-placement set; a separate value head
estimates the current state's value. This afterstate formulation is what makes
3D Tetris learnable: one decision = one placed piece, and clears are reachable.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class AfterstateAC(nn.Module):
    def __init__(self, state_dim, placement_dim, hidden=128):
        super().__init__()
        # scorer: placement features -> scalar logit (shared across candidates)
        self.scorer = mlp([placement_dim, hidden, hidden, 1])
        # value: current-state features -> scalar value
        self.value = mlp([state_dim, hidden, hidden, 1])

    def score(self, cand):
        return self.scorer(cand).squeeze(-1)

    def value_of(self, state):
        return self.value(state).squeeze(-1)


class AfterstatePPO:
    def __init__(self, state_dim, placement_dim, device="cpu",
                 lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                 epochs=4, minibatch=512, ent_coef=0.01, vf_coef=0.5,
                 max_grad_norm=0.5):
        self.device = torch.device(device)
        self.net = AfterstateAC(state_dim, placement_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def act(self, state_feats, cand_feats, greedy=False):
        """Pick a placement. cand_feats: (n, placement_dim). Returns (idx, logp, value)."""
        c = torch.as_tensor(cand_feats, dtype=torch.float32, device=self.device)
        logits = self.net.score(c)
        dist = Categorical(logits=logits)
        if greedy:
            a = torch.argmax(logits)
        else:
            a = dist.sample()
        s = torch.as_tensor(state_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        v = self.net.value_of(s)
        return int(a.item()), float(dist.log_prob(a).item()), float(v.item())

    @torch.no_grad()
    def value_of(self, state_feats):
        s = torch.as_tensor(state_feats, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.net.value_of(s).item())

    def compute_gae(self, rewards, values, dones, last_value):
        adv = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
        return adv, adv + values

    def update(self, states, cand_list, actions, old_logp, adv, returns):
        device = self.device
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)
        old_logp = torch.as_tensor(old_logp, dtype=torch.float32, device=device)
        adv = torch.as_tensor(adv, dtype=torch.float32, device=device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ncounts = [c.shape[0] for c in cand_list]
        Fdim = cand_list[0].shape[1]

        n = states.shape[0]
        idx = np.arange(n)
        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "k": 0}
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch):
                mb = idx[start:start + self.minibatch]
                maxn = max(ncounts[i] for i in mb)
                C = torch.zeros(len(mb), maxn, Fdim, dtype=torch.float32, device=device)
                M = torch.zeros(len(mb), maxn, dtype=torch.bool, device=device)
                for j, i in enumerate(mb):
                    ni = ncounts[i]
                    C[j, :ni] = torch.as_tensor(cand_list[i], dtype=torch.float32, device=device)
                    M[j, :ni] = True
                logits = self.net.score(C)                 # (mb, maxn)
                logits = logits.masked_fill(~M, -1e9)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()
                ratio = torch.exp(logp - old_logp[mb])
                a_mb = adv[mb]
                s1 = ratio * a_mb
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_mb
                pg_loss = -torch.min(s1, s2).mean()
                value = self.net.value_of(states[mb])
                vf_loss = 0.5 * (returns[mb] - value).pow(2).mean()
                loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                stats["pg"] += pg_loss.item(); stats["vf"] += vf_loss.item()
                stats["ent"] += entropy.item(); stats["k"] += 1
        k = max(stats["k"], 1)
        return {"pg_loss": stats["pg"] / k, "vf_loss": stats["vf"] / k, "entropy": stats["ent"] / k}

    def save(self, path):
        torch.save({"model": self.net.state_dict()}, path)

    def load(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.net.load_state_dict(ckpt["model"])
