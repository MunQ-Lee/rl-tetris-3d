"""Live versus GUI: two PPO agents play competitive 3D Tetris in your browser.

A background thread runs versus matches between agent 0 and agent 1, animating
each piece dropping into place and exchanging garbage when layers clear. The
browser polls /api/state (~15 Hz) and renders both boards side by side with
three.js. Checkpoints (latest_p0.pt / latest_p1.pt) are hot-reloaded, so you can
watch the two models improve as they train.

Usage:
    python server.py --port 8000 --speed 6
"""
import argparse
import copy
import os
import threading
import time

import numpy as np
from flask import Flask, jsonify, send_from_directory

from tetris3d.env import Tetris3DEnv, W, D, H
from tetris3d.versus import VersusEnv
from tetris3d.afterstate_agent import AfterstatePPO

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
app = Flask(__name__, static_folder=None)

_lock = threading.Lock()
_shared = {"state": None, "status": "starting"}


class Match:
    def __init__(self, ckpt_dir, speed, reload_every=8.0):
        self.ckpt_dir = ckpt_dir
        self.frame_delay = 1.0 / max(1.0, speed * 6)   # ~6 animation frames per placement
        self.reload_every = reload_every
        self.agents = [AfterstatePPO(Tetris3DEnv.state_dim(), Tetris3DEnv.placement_dim())
                       for _ in range(2)]
        self.env = VersusEnv(prefill_p=0.0)
        self._mtime = [None, None]
        self._last_reload = 0.0
        self._reload(force=True)

    def _reload(self, force=False):
        now = time.time()
        if not force and now - self._last_reload < self.reload_every:
            return
        self._last_reload = now
        msgs = []
        for p in (0, 1):
            path = os.path.join(self.ckpt_dir, f"latest_p{p}.pt")
            if os.path.exists(path):
                mt = os.path.getmtime(path)
                if force or mt != self._mtime[p]:
                    try:
                        self.agents[p].load(path)
                        self._mtime[p] = mt
                        msgs.append(f"p{p}✓")
                    except Exception as e:  # noqa: BLE001
                        msgs.append(f"p{p} err")
            else:
                msgs.append(f"p{p} random")
        with _lock:
            _shared["status"] = "models: " + " ".join(msgs)

    def _publish(self, floating=(None, None)):
        boards = []
        for p in (0, 1):
            boards.append(self.env.envs[p].render_state(piece_cells=floating[p]))
        snap = {
            "boards": boards,
            "winner": self.env.winner,
            "rounds": self.env.rounds,
            "attacks": list(self.env.attacks),
            "done": bool(self.env.done),
        }
        with _lock:
            snap["status"] = _shared["status"]
            _shared["state"] = snap

    def run(self):
        self.env.reset()
        self._publish()
        while True:
            self._reload()
            if self.env.done:
                time.sleep(1.5)
                self.env.reset()
                self._publish()
                continue

            chosen = [None, None]
            anims = [None, None]
            for p in (0, 1):
                cands, feats = self.env.legal(p)
                if not cands:
                    self.env.envs[p].done = True
                    continue
                idx, _, _ = self.agents[p].act(self.env.state(p), feats, greedy=True)
                chosen[p] = cands[idx]
                anims[p] = self._anim_path(chosen[p])

            if chosen[0] is None or chosen[1] is None:
                self.env.done = True
                self.env.winner = 1 if chosen[0] is None else 0
                self._publish()
                continue

            # Animate both pieces dropping simultaneously.
            nframes = max(len(anims[0]), len(anims[1]))
            for f in range(nframes):
                fl = [anims[p][min(f, len(anims[p]) - 1)] for p in (0, 1)]
                self._publish(floating=fl)
                time.sleep(self.frame_delay)

            self.env.step(chosen)
            self._publish()
            time.sleep(self.frame_delay)

    def _anim_path(self, placement):
        """World-cell frames of the piece falling from the top to its rest pos."""
        x, y, zr = int(placement.pos[0]), int(placement.pos[1]), int(placement.pos[2])
        max_oz = int(placement.cells[:, 2].max())
        z_top = H - 1 - max_oz
        frames = []
        for z in range(max(z_top, zr), zr - 1, -1):
            frames.append((placement.cells + (x, y, z)).tolist())
        return frames or [(placement.cells + (x, y, zr)).tolist()]


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/state")
def api_state():
    with _lock:
        state = copy.deepcopy(_shared["state"])
    return jsonify(state or {"boards": [], "done": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--speed", type=float, default=6.0, help="placements per second")
    args = ap.parse_args()

    match = Match(args.ckpt_dir, speed=args.speed)
    threading.Thread(target=match.run, daemon=True).start()
    print(f"Versus GUI at http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
