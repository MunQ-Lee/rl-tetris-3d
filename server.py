"""Web GUI server: the PPO agent plays 3D Tetris live in your browser.

A background thread runs the agent and continuously updates a shared snapshot.
The browser polls /api/state (~10 Hz) and renders the board with three.js.

Usage:
    python server.py --ckpt checkpoints/latest.pt --host 0.0.0.0 --port 8000

If the checkpoint is missing the agent plays with a randomly initialized
policy (so you can watch even before training finishes). The server also
hot-reloads the checkpoint while running, so you can watch training improve.
"""
import argparse
import copy
import os
import threading
import time

from flask import Flask, jsonify, send_from_directory

from tetris3d.env import Tetris3DEnv, NUM_ACTIONS
from tetris3d.ppo import PPO

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

app = Flask(__name__, static_folder=None)

_state_lock = threading.Lock()
_shared = {"state": None, "ckpt_status": "starting", "episode": 0}


class Player:
    def __init__(self, ckpt_path, speed=8.0, reload_every=10.0):
        self.ckpt_path = ckpt_path
        self.delay = 1.0 / speed
        self.reload_every = reload_every
        self.agent = PPO(Tetris3DEnv.obs_dim(), NUM_ACTIONS)
        self.env = Tetris3DEnv()
        self._last_mtime = None
        self._last_reload = 0.0
        self._maybe_reload(force=True)

    def _maybe_reload(self, force=False):
        now = time.time()
        if not force and now - self._last_reload < self.reload_every:
            return
        self._last_reload = now
        if os.path.exists(self.ckpt_path):
            mtime = os.path.getmtime(self.ckpt_path)
            if force or mtime != self._last_mtime:
                try:
                    self.agent.load(self.ckpt_path)
                    self._last_mtime = mtime
                    status = f"loaded {os.path.basename(self.ckpt_path)} @ {time.strftime('%H:%M:%S', time.localtime(mtime))}"
                except Exception as e:  # noqa: BLE001
                    status = f"load failed: {e}"
            else:
                status = "up to date"
        else:
            status = "no checkpoint (random policy)"
        with _state_lock:
            _shared["ckpt_status"] = status

    def run(self):
        obs = self.env.reset()
        self._publish()
        while True:
            self._maybe_reload()
            action, _, _ = self.agent.act(obs)
            obs, _, done, _ = self.env.step(action)
            self._publish()
            time.sleep(self.delay)
            if done:
                time.sleep(1.0)  # pause on the final frame
                with _state_lock:
                    _shared["episode"] += 1
                obs = self.env.reset()
                self._publish()

    def _publish(self):
        snap = self.env.render_state()
        with _state_lock:
            snap["ckpt_status"] = _shared["ckpt_status"]
            snap["episode"] = _shared["episode"]
            _shared["state"] = snap


@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.route("/api/state")
def api_state():
    with _state_lock:
        state = copy.deepcopy(_shared["state"])
    return jsonify(state or {"done": True, "cells": [], "piece": []})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--speed", type=float, default=8.0, help="agent steps per second")
    args = ap.parse_args()

    player = Player(args.ckpt, speed=args.speed)
    threading.Thread(target=player.run, daemon=True).start()

    print(f"GUI at http://localhost:{args.port}  (ckpt: {args.ckpt})")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
