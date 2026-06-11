"""Two-player competitive 3D Tetris (versus mode).

Two independent boards play simultaneously. Each round, both players place one
piece; the layers they clear are converted to garbage rows that rise on the
*opponent's* board (attacks cancel out — only the net difference is sent). A
player tops out when garbage (or their own stack) is shoved above the ceiling;
the survivor wins.

Used both for competitive training (train.py) and for the live GUI (server.py).
"""
import numpy as np

from .env import Tetris3DEnv, GARBAGE_SENT

# Competitive reward shaping (added on top of each board's solo placement reward).
# Kept small on purpose: the dominant learning signal must be the per-placement
# clear/fill reward. A large win/lose bonus depends on the opponent, not on the
# chosen placement, so it acts as noise that swamps the real signal (the policy
# would never learn to clear). Win/lose is only a tiebreak nudge here.
ATTACK_REWARD = 6.0     # per net garbage layer sent to the opponent
RECV_PENALTY = -1.5     # per net garbage layer received
WIN_REWARD = 5.0
LOSE_REWARD = -5.0


class VersusEnv:
    def __init__(self, seed=None, prefill_p=0.0, max_rounds=300):
        seed0 = None if seed is None else seed * 2
        seed1 = None if seed is None else seed * 2 + 1
        self.envs = [Tetris3DEnv(seed0), Tetris3DEnv(seed1)]
        self.prefill_p = prefill_p
        self.max_rounds = max_rounds
        self.reset()

    def set_prefill(self, p):
        self.prefill_p = p

    def reset(self):
        for e in self.envs:
            e.reset(self.prefill_p)
        self.done = False
        self.winner = None        # 0, 1, or -1 for draw
        self.rounds = 0
        self.attacks = [0, 0]     # total garbage sent by each player
        return [e.state_features() for e in self.envs]

    def legal(self, p):
        """(placements, feature_matrix) for player p's current piece."""
        return self.envs[p].enumerate_placements()

    def state(self, p):
        return self.envs[p].state_features()

    def step(self, chosen):
        """Apply both players' chosen placements; exchange garbage; score.

        chosen: [placement_for_p0, placement_for_p1]
        Returns (next_states, rewards[2], done, info).
        """
        assert not self.done
        self.rounds += 1
        rewards = [0.0, 0.0]
        lines = [0, 0]
        for p in (0, 1):
            r, _, info = self.envs[p].place(chosen[p])
            rewards[p] += r                      # solo reward (clear + health + place)
            lines[p] = info["lines"]

        # Convert clears to garbage and cancel attacks; send the net to the loser.
        g = [GARBAGE_SENT[min(lines[0], 4)], GARBAGE_SENT[min(lines[1], 4)]]
        net = g[0] - g[1]
        if net > 0:
            self.envs[1].add_garbage(net)
            self.attacks[0] += net
            rewards[0] += ATTACK_REWARD * net
            rewards[1] += RECV_PENALTY * net
        elif net < 0:
            self.envs[0].add_garbage(-net)
            self.attacks[1] += -net
            rewards[1] += ATTACK_REWARD * (-net)
            rewards[0] += RECV_PENALTY * (-net)

        d0, d1 = self.envs[0].done, self.envs[1].done
        if d0 or d1 or self.rounds >= self.max_rounds:
            self.done = True
            if d0 and not d1:
                self.winner = 1
            elif d1 and not d0:
                self.winner = 0
            else:
                # both topped out, or round cap reached: more layers cleared wins
                c0, c1 = self.envs[0].cleared_layers, self.envs[1].cleared_layers
                self.winner = 0 if c0 > c1 else (1 if c1 > c0 else -1)
            for p in (0, 1):
                if self.winner == -1:
                    pass
                elif self.winner == p:
                    rewards[p] += WIN_REWARD
                else:
                    rewards[p] += LOSE_REWARD

        next_states = [self.envs[0].state_features(), self.envs[1].state_features()]
        info = {
            "winner": self.winner,
            "lines": lines,
            "net_garbage": net,
            "rounds": self.rounds,
            "scores": [self.envs[0].score, self.envs[1].score],
            "cleared": [self.envs[0].cleared_layers, self.envs[1].cleared_layers],
        }
        return next_states, rewards, self.done, info

    def render_state(self):
        """Snapshot of both boards for the GUI."""
        return {
            "boards": [self.envs[0].render_state(), self.envs[1].render_state()],
            "winner": self.winner,
            "rounds": self.rounds,
            "attacks": list(self.attacks),
            "done": bool(self.done),
        }
